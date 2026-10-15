"""Hybrid retrieval: Postgres full-text + pgvector, fused with RRF, optionally reranked.

Modes (set per request or via RETRIEVAL_MODE) exist so evals can run an ablation table:
  vector  - cosine nearest neighbours on the HNSW index
  fts     - full-text match on the GIN-indexed tsvector, ranked by ts_rank_cd
  hybrid  - both lists merged by reciprocal rank fusion
"""

import asyncio
import re
from dataclasses import asdict, dataclass

from psycopg_pool import AsyncConnectionPool

from docpilot.config import RerankerName, RetrievalMode, Settings
from docpilot.embeddings import Embedder
from docpilot.retrieval.fusion import rrf
from docpilot.retrieval.rerank import make_reranker

_COLUMNS = "c.id, c.document_id, c.ord, c.text, d.title"


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


def fts_query(question: str) -> str:
    """OR the words together: plainto_tsquery ANDs them, which misses most natural questions."""
    words = re.findall(r"[A-Za-z0-9]+", question)
    return " | ".join(words)


class Retriever:
    def __init__(self, pool: AsyncConnectionPool, embedder: Embedder, settings: Settings):
        self._pool = pool
        self._embedder = embedder
        self._settings = settings

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
            vec, fts = await asyncio.gather(self._vector(query, n), self._fts(query, n))
            by_id = {c.id: c for c in [*fts, *vec]}
            fused = rrf([[c.id for c in vec], [c.id for c in fts]])
            candidates = [_with_score(by_id[cid], score) for cid, score in fused]

        if reranker != "none" and candidates:
            head = candidates[: max(top_k * 4, 20)]
            order = await make_reranker(reranker, self._settings).rerank(
                query, [c.text for c in head]
            )
            candidates = [_with_score(head[i], score) for i, score in order]
        return candidates[:top_k]

    async def _vector(self, query: str, limit: int) -> list[RetrievedChunk]:
        [embedding] = await self._embedder.embed([query])
        sql = f"""SELECT {_COLUMNS}, 1 - (c.embedding <=> %(q)s) AS score
                  FROM chunks c JOIN documents d ON d.id = c.document_id
                  ORDER BY c.embedding <=> %(q)s LIMIT %(limit)s"""
        return await self._fetch(sql, {"q": embedding, "limit": limit})

    async def _fts(self, query: str, limit: int) -> list[RetrievedChunk]:
        tsquery = fts_query(query)
        if not tsquery:
            return []
        sql = f"""SELECT {_COLUMNS}, ts_rank_cd(c.tsv, q) AS score
                  FROM chunks c JOIN documents d ON d.id = c.document_id,
                       to_tsquery('english', %(q)s) q
                  WHERE c.tsv @@ q
                  ORDER BY score DESC, c.id LIMIT %(limit)s"""
        return await self._fetch(sql, {"q": tsquery, "limit": limit})

    async def _fetch(self, sql: str, params: dict) -> list[RetrievedChunk]:
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(sql, params)).fetchall()
        return [RetrievedChunk(**{**r, "score": float(r["score"])}) for r in rows]


def _with_score(chunk: RetrievedChunk, score: float) -> RetrievedChunk:
    return RetrievedChunk(**{**chunk.to_dict(), "score": float(score)})
