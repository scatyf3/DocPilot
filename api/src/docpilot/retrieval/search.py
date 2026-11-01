"""Hybrid retrieval: Postgres full-text + pgvector, fused with RRF, optionally reranked.

Modes (set per request or via RETRIEVAL_MODE) exist so evals can run an ablation table:
  vector  - cosine nearest neighbours on the HNSW index
  fts     - BM25 over the GIN-indexed tsvector (see _fts)
  hybrid  - both lists merged by reciprocal rank fusion
"""

import asyncio
import logging
import math
import time
from dataclasses import asdict, dataclass

import httpx
from psycopg_pool import AsyncConnectionPool

from docpilot.config import RerankerName, RetrievalMode, Settings
from docpilot.embeddings import Embedder
from docpilot.retrieval.fusion import rrf
from docpilot.retrieval.rerank import make_reranker

log = logging.getLogger(__name__)
_COLUMNS = "c.id, c.document_id, c.ord, c.text, d.title"

# BM25's usual parameters: term-frequency saturation and document-length normalisation.
BM25_K1 = 1.2
BM25_B = 0.75
# Scoring a term costs one row per chunk containing it, so stop adding terms past this many.
# Rarest terms go first, and they carry most of the signal: common ones add cost, not ranking.
BM25_MAX_POSTINGS = 20_000
_STATS_TTL_SECONDS = 300


@dataclass
class RetrievedChunk:
    id: int
    document_id: int
    ord: int
    title: str
    text: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def tsquery_literal(lexeme: str) -> str:
    """Quote a lexeme for a ::tsquery cast, which has its own rules, unlike SQL string quoting."""
    return "'" + lexeme.replace("\\", "\\\\").replace("'", "''") + "'"


def select_terms(doc_freqs: dict[str, int], max_postings: int = BM25_MAX_POSTINGS) -> list[str]:
    """Terms that occur in the corpus, rarest first, within the postings budget (at least one)."""
    selected, postings = [], 0
    for term, df in sorted(doc_freqs.items(), key=lambda kv: (kv[1], kv[0])):
        if df == 0:
            continue
        if selected and postings + df > max_postings:
            break
        selected.append(term)
        postings += df
    return selected


def bm25_idf(n_chunks: int, df: int) -> float:
    n_chunks = max(n_chunks, df)  # corpus stats are cached and may lag behind a fresh upload
    return math.log((n_chunks - df + 0.5) / (df + 0.5) + 1)


class Retriever:
    def __init__(self, pool: AsyncConnectionPool, embedder: Embedder, settings: Settings):
        self._pool = pool
        self._embedder = embedder
        self._settings = settings
        self._stats: tuple[float, int, float] | None = None  # (fetched at, chunks, mean length)

    async def search(
        self, query: str, *, top_k: int = 5,
        mode: RetrievalMode | None = None, reranker: RerankerName | None = None,
    ) -> list[RetrievedChunk]:
        mode = mode or self._settings.retrieval_mode
        reranker = reranker or self._settings.reranker
        n = self._settings.candidate_k

        if mode == "vector":
            candidates = await self._vector(query, n)
        elif mode == "fts":
            candidates = await self._fts(query, n)
        else:
            # Two independent queries on separate pooled connections: run them concurrently.
            vec, fts = await asyncio.gather(self._vector(query, n), self._fts(query, n),
                                            return_exceptions=True)
            vec, fts = _survivors(vec, fts)
            by_id = {c.id: c for c in [*fts, *vec]}
            fused = rrf([[c.id for c in vec], [c.id for c in fts]])
            candidates = [_with_score(by_id[cid], score) for cid, score in fused]

        if reranker != "none" and candidates:
            head = candidates[: max(top_k * 4, 20)]
            try:
                order = await make_reranker(reranker, self._settings).rerank(
                    query, [c.text for c in head]
                )
            except httpx.HTTPError as e:  # a worse ranking beats no answer
                log.warning("reranker failed, keeping first-stage order: %s", type(e).__name__)
            else:
                candidates = [_with_score(head[i], score) for i, score in order]
        return candidates[:top_k]

    async def _vector(self, query: str, limit: int) -> list[RetrievedChunk]:
        [embedding] = await self._embedder.embed([query])
        sql = f"""SELECT {_COLUMNS}, 1 - (c.embedding <=> %(q)s) AS score
                  FROM chunks c JOIN documents d ON d.id = c.document_id
                  ORDER BY c.embedding <=> %(q)s LIMIT %(limit)s"""
        return await self._fetch(sql, {"q": embedding, "limit": limit})

    async def _fts(self, query: str, limit: int) -> list[RetrievedChunk]:
        """BM25, computed from the tsvector.

        Postgres's own ts_rank has no inverse document frequency: in a question, "the" and
        "Keystone" would count the same. Ranking by it scored about a quarter of the published
        BM25 baseline on BEIR FiQA; this version matches that baseline.
        """
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(
                "SELECT lexeme FROM unnest(to_tsvector('english', %s))", (query,)
            )).fetchall()
            lexemes = [r["lexeme"] for r in rows]
            if not lexemes:
                return []
            rows = await (await conn.execute(
                """SELECT t.lexeme, (SELECT count(*) FROM chunks c WHERE c.tsv @@ t.q::tsquery) AS df
                   FROM unnest(%s::text[], %s::text[]) AS t(lexeme, q)""",
                (lexemes, [tsquery_literal(x) for x in lexemes]),
            )).fetchall()
            doc_freqs = {r["lexeme"]: r["df"] for r in rows}
            terms = select_terms(doc_freqs)
            if not terms:
                return []
            n_chunks, mean_length = await self._corpus_stats(conn)
        sql = f"""WITH scored AS (
                      SELECT c.id, sum(w.idf * cardinality(u.positions) * (%(k1)s + 1)
                                       / (cardinality(u.positions) + %(k1)s * (1 - %(b)s + %(b)s
                                          * length(c.tsv) / %(mean_length)s))) AS score
                      FROM chunks c
                      CROSS JOIN LATERAL unnest(c.tsv) AS u
                      JOIN unnest(%(terms)s::text[], %(idf)s::float8[]) AS w(lexeme, idf)
                        ON w.lexeme = u.lexeme
                      WHERE c.tsv @@ %(any_term)s::tsquery
                      GROUP BY c.id ORDER BY score DESC, c.id LIMIT %(limit)s
                  )
                  SELECT {_COLUMNS}, s.score
                  FROM scored s JOIN chunks c ON c.id = s.id JOIN documents d ON d.id = c.document_id
                  ORDER BY s.score DESC, c.id"""
        return await self._fetch(sql, {
            "k1": BM25_K1, "b": BM25_B, "mean_length": mean_length, "limit": limit, "terms": terms,
            "idf": [bm25_idf(n_chunks, doc_freqs[t]) for t in terms],
            "any_term": " | ".join(tsquery_literal(t) for t in terms),
        })

    async def _corpus_stats(self, conn) -> tuple[int, float]:
        """Chunk count and mean length. Scanning for them on every query would be wasteful."""
        now = time.monotonic()
        if self._stats is None or now - self._stats[0] > _STATS_TTL_SECONDS:
            row = await (await conn.execute(
                "SELECT count(*) AS n, avg(length(tsv))::float8 AS mean FROM chunks"
            )).fetchone()
            self._stats = (now, row["n"], row["mean"] or 1.0)
        return self._stats[1], max(self._stats[2], 1.0)

    async def _fetch(self, sql: str, params: dict) -> list[RetrievedChunk]:
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(sql, params)).fetchall()
        return [RetrievedChunk(**{**r, "score": float(r["score"])}) for r in rows]


def _survivors(vec, fts) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """If one retriever fails (say, the embedding API is down) serve from the other."""
    if isinstance(vec, BaseException) and isinstance(fts, BaseException):
        raise vec
    for name, result in (("vector", vec), ("fts", fts)):
        if isinstance(result, BaseException):
            log.warning("%s retrieval failed, degrading: %s", name, type(result).__name__)
    return ([] if isinstance(vec, BaseException) else vec,
            [] if isinstance(fts, BaseException) else fts)


def _with_score(chunk: RetrievedChunk, score: float) -> RetrievedChunk:
    return RetrievedChunk(**{**chunk.to_dict(), "score": float(score)})
