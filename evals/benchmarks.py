"""Retrieval benchmarks on public datasets, for what the 14-doc wiki in evals/corpus can't show.

    python -m evals.benchmarks download
    python -m evals.benchmarks run --datasets hotpot,multihop,fiqa,techqa --rerankers none,cohere

  hotpot    HotpotQA dev (distractor config): multi-hop over Wikipedia intros; gold = both
            supporting articles. Its fullwiki config omits gold articles for most questions.
  multihop  MultiHop-RAG: long news articles; gold = verbatim evidence quotes; has null queries
  fiqa      BEIR FiQA test: finance forum, vocabulary gap; human qrels, published BM25 baselines
  techqa    RAGBench TechQA: IBM technotes; gold = documents GPT-4 marked relevant (weak labels)
  qasper    QASPER dev: questions about one NLP paper each, asked from its title and abstract;
            gold = evidence paragraphs (or table/figure captions) chosen by each annotator.
            Questions are only meaningful within their paper ("what dataset do they use?"),
            so each is searched within its paper. Papers come as the dataset's own parsed
            text, so this scores chunking and retrieval, not PDF parsing.
            This is NOT the official protocol (qasper_evaluator.py: the model reads the whole
            paper and predicts an answer plus an evidence set, scored by Answer F1 and Evidence
            F1, each the max over annotators). Differences: recall@k instead of set F1; each
            metric takes its own best annotator; unanswerable annotators of answerable questions,
            bare section-name evidence, and questions with no usable evidence are dropped;
            "FLOAT SELECTED:" evidence matches the caption text.
  qasper_lib  The same questions over the whole 416-paper library, unfiltered, with the paper's
            title prepended ('In the paper "<title>": <question>') so each one is unambiguous.
            A hit needs the right paper and the right paragraph.
  qasper_test, qasper_lib_test  The same on the test split. Its README asks that it be used
            only to evaluate a finished system, so iterate on dev.

Each dataset gets its own Postgres schema (bench_<name>). Questions go through POST /search as in
evals.run. Documents are chunked, embedded and stored with the app's own functions, but in
batches rather than one POST /documents per file, which would take far too long at this size.
Data lives in EVAL_DATA_DIR and is never committed; check each dataset's license before reuse.
"""

import argparse
import asyncio
import glob
import hashlib
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import quantiles

import httpx
import numpy as np

from docpilot import store
from docpilot.app import create_app
from docpilot.cli import run_async
from docpilot.config import Settings
from docpilot.db import init_schema, tenant_connection
from docpilot.ingest.chunk import approx_tokens, chunk_text
from docpilot.retrieval.rerank import CohereReranker
from docpilot.security import DEFAULT_TENANT_ID
from evals.dataset import EVALS_DIR
from evals.metrics import contains, mean_or_none, normalize

DATA_DIR = Path(os.environ.get("EVAL_DATA_DIR", EVALS_DIR / "data"))
RESULTS_DIR = EVALS_DIR / "results"
K_VALUES = (1, 5, 10)
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class BenchItem:
    id: str
    question: str
    type: str
    units: list[str]  # gold documents (keys into docs) or quotes; empty = no answer in the corpus
    answer: str | None = None  # short reference answer, where the dataset has one
    scope: str | None = None  # search only this document (a key into docs); None = everything
    unit_sets: list[list[str]] | None = None  # alternative gold sets (one per annotator); best counts


@dataclass
class Benchmark:
    docs: dict[str, str]
    items: list[BenchItem]
    unit_kind: str  # "doc" or "quote"


def download() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    snapshot_download("hotpotqa/hotpot_qa", repo_type="dataset", local_dir=DATA_DIR / "hotpot",
                      allow_patterns=["distractor/validation*"])
    for name in ("MultiHopRAG.json", "corpus.json"):
        hf_hub_download("yixuantt/MultiHopRAG", name, repo_type="dataset", local_dir=DATA_DIR / "multihop")
    snapshot_download("BeIR/fiqa", repo_type="dataset", local_dir=DATA_DIR / "fiqa",
                      allow_patterns=["corpus/*", "queries/*"])
    hf_hub_download("BeIR/fiqa-qrels", "test.tsv", repo_type="dataset", local_dir=DATA_DIR / "fiqa")
    snapshot_download("galileo-ai/ragbench", repo_type="dataset", local_dir=DATA_DIR / "techqa",
                      allow_patterns=["techqa/*"])
    _download_qasper()
    print(f"downloaded to {DATA_DIR}")


QASPER_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/"
QASPER_FILES = {"qasper-train-dev-v0.3.tgz": ["qasper-dev-v0.3.json"],
                "qasper-test-and-evaluator-v0.3.tgz": ["qasper-test-v0.3.json", "qasper_evaluator.py"]}


def _download_qasper() -> None:
    import io
    import tarfile

    target = DATA_DIR / "qasper"
    target.mkdir(parents=True, exist_ok=True)
    for archive, names in QASPER_FILES.items():
        data = httpx.get(QASPER_URL + archive, timeout=300, follow_redirects=True).raise_for_status().content
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            for name in names:
                (target / name).write_bytes(tar.extractfile(tar.getmember(name)).read())


def qasper_papers(split: str) -> dict:
    return json.loads((DATA_DIR / f"qasper/qasper-{split}-v0.3.json").read_text(encoding="utf-8"))


def _parquet(pattern: str) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(glob.glob(str(DATA_DIR / pattern))[0]).to_pylist()


def load_hotpot() -> Benchmark:
    docs: dict[str, str] = {}
    items = []
    for row in _parquet("hotpot/distractor/validation*"):
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"], strict=True):
            docs.setdefault(title, f"# {title}\n\n{''.join(sentences)}")
        items.append(BenchItem(row["id"], row["question"], row["type"],
                               sorted(set(row["supporting_facts"]["title"])), row["answer"]))
    return Benchmark(docs, items, "doc")


def load_multihop() -> Benchmark:
    corpus = json.loads((DATA_DIR / "multihop/corpus.json").read_text(encoding="utf-8"))
    docs = {c["url"]: f"# {c['title']}\n\n{c['body']}" for c in corpus}
    queries = json.loads((DATA_DIR / "multihop/MultiHopRAG.json").read_text(encoding="utf-8"))
    items = [BenchItem(f"mh{i}", q["query"], q["question_type"],
                       sorted({e["fact"] for e in q["evidence_list"]}), q["answer"])
             for i, q in enumerate(queries)]
    return Benchmark(docs, items, "quote")


def load_fiqa() -> Benchmark:
    docs = {r["_id"]: r["text"] for r in _parquet("fiqa/corpus/*") if r["text"].strip()}
    queries = {r["_id"]: r["text"] for r in _parquet("fiqa/queries/*")}
    gold: dict[str, list[str]] = {}
    for line in (DATA_DIR / "fiqa/test.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        qid, doc_id, score = line.split("\t")
        if int(score) > 0 and doc_id in docs:
            gold.setdefault(qid, []).append(doc_id)
    return Benchmark(docs, [BenchItem(q, queries[q], "fiqa", u) for q, u in gold.items()], "doc")


def _key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def load_techqa() -> Benchmark:
    docs: dict[str, str] = {}
    gold: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        for row in _parquet(f"techqa/techqa/{split}-*"):
            docs.update((_key(d), d) for d in row["documents"])
            if split == "test":
                # Keys look like "0b": document index, then sentence letters.
                gold.setdefault(row["question"], set()).update(
                    _key(row["documents"][int(k[0])]) for k in row["all_relevant_sentence_keys"])
    items = [BenchItem(f"tq{i}", q, "techqa", sorted(u))
             for i, (q, u) in enumerate(gold.items()) if u]
    return Benchmark(docs, items, "doc")


FLOAT_PREFIX = "FLOAT SELECTED: "


def qasper_markdown(paper: dict) -> str:
    """One paper as Markdown. Nested sections arrive as "Model ::: Response Generation"."""
    parts = [f"# {paper['title']}", "## Abstract", paper["abstract"].strip()]
    for section in paper["full_text"]:
        name = [x.strip() for x in (section["section_name"] or "").split(":::") if x.strip()]
        if name:
            parts.append(f"{'#' * min(len(name) + 1, 6)} {name[-1]}")
        parts += [x.strip() for x in section["paragraphs"] if x.strip()]
    captions = [f["caption"].strip() for f in paper["figures_and_tables"] if f["caption"].strip()]
    if captions:
        parts += ["## Figures and tables", *captions]
    return "\n\n".join(parts)


def qasper_items(paper_id: str, paper: dict) -> list[BenchItem]:
    """Null query when every annotator said unanswerable. Some evidence is a bare section name
    (the whole section); a quote can't score that, so it is dropped. Annotators left with no
    evidence are left out, and a question with no evidence at all is dropped."""
    sections = {s["section_name"].strip() for s in paper["full_text"] if s["section_name"]}
    items = []
    for q in paper["qas"]:
        answers = [a["answer"] for a in q["answers"]]
        sets = [sorted({e.removeprefix(FLOAT_PREFIX).strip() for e in a["evidence"]} - sections - {""})
                for a in answers if not a["unanswerable"]]
        sets = [u for u in sets if u]
        if all(a["unanswerable"] for a in answers):
            kind = "unanswerable"
        elif not sets:
            continue
        else:
            kinds = {("yes_no" if a["yes_no"] is not None else "extractive" if a["extractive_spans"]
                      else "abstractive") for a in answers if not a["unanswerable"]}
            kind = kinds.pop() if len(kinds) == 1 else "mixed"
        units = sorted({u for s in sets for u in s})
        items.append(BenchItem(q["question_id"], q["question"], kind, units, scope=paper_id,
                               unit_sets=sets or None))
    return items


def load_qasper(split: str = "dev", library: bool = False) -> Benchmark:
    papers = qasper_papers(split)
    docs = {pid: qasper_markdown(p) for pid, p in papers.items()}
    items = [i for pid, p in papers.items() for i in qasper_items(pid, p)]
    if library:
        for i in items:
            i.question = f'In the paper "{papers[i.scope]["title"].strip()}": {i.question}'
            i.scope = None
    return Benchmark(docs, items, "quote")


LOADERS = {"hotpot": load_hotpot, "multihop": load_multihop, "fiqa": load_fiqa, "techqa": load_techqa,
           "qasper": load_qasper, "qasper_lib": lambda: load_qasper(library=True),
           "qasper_test": lambda: load_qasper("test"),
           "qasper_lib_test": lambda: load_qasper("test", library=True)}


async def ingest(app, settings: Settings, docs: dict[str, str]) -> dict[str, int]:
    """Chunk, embed in batches, store. Returns document key -> document id."""
    keys = list(docs)
    chunks = [chunk_text(docs[k], settings.chunk_tokens, settings.chunk_overlap_tokens) for k in keys]
    keep = [i for i, c in enumerate(chunks) if c]
    flat = [c for i in keep for c in chunks[i]]
    print(f"  {len(keep)} docs, {len(flat)} chunks", flush=True)
    embeddings = await app.state.embedder.embed(flat)
    print("  embedded", flush=True)

    sem, ids, pos = asyncio.Semaphore(8), {}, 0

    async def one(key: str, doc_chunks: list[str], doc_embeddings: list) -> None:
        async with sem:
            ids[key] = await store.insert_document(
                app.state.pool, tenant_id=DEFAULT_TENANT_ID, title=key, source=key, kind="upload",
                chunks=doc_chunks, embeddings=doc_embeddings)

    tasks = []
    for i in keep:
        tasks.append(one(keys[i], chunks[i], embeddings[pos : pos + len(chunks[i])]))
        pos += len(chunks[i])
    await asyncio.gather(*tasks)
    return ids


async def existing_doc_ids(app, bench: Benchmark) -> dict[str, int]:
    """Document key -> id for a schema ingested earlier (ingest stores the key as the title)."""
    async with tenant_connection(app.state.pool, DEFAULT_TENANT_ID) as conn:
        rows = await (await conn.execute("SELECT id, title FROM documents")).fetchall()
    by_title = {r["title"]: r["id"] for r in rows}
    return {k: by_title[k] for k in bench.docs if k in by_title}


def unit_ranks(ranked: list[dict], units: list[str], kind: str, doc_ids: dict[str, int]) -> list[int | None]:
    """1-based rank of the first chunk satisfying each gold unit (None if not retrieved)."""
    def hit(chunk: dict, unit: str) -> bool:
        return (chunk["document_id"] == doc_ids.get(unit) if kind == "doc"
                else contains(chunk["text"], unit))

    return [next((r for r, c in enumerate(ranked, start=1) if hit(c, u)), None) for u in units]


def score_ranks(ranks: list[int | None]) -> dict[str, float]:
    found = [r for r in ranks if r is not None]
    out: dict[str, float] = {}
    for k in K_VALUES:
        out[f"recall@{k}"] = sum(r <= k for r in found) / len(ranks)
    out["complete@5"] = float(all(r is not None and r <= 5 for r in ranks))
    out["mrr@10"] = 1 / min(found) if found and min(found) <= 10 else 0.0
    # Binary nDCG over gold units. A chunk holding two units counts for both, so cap at 1.
    dcg = sum(1 / math.log2(r + 1) for r in found if r <= 10)
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(len(ranks), 10) + 1))
    out["ndcg@10"] = min(dcg / ideal, 1.0)
    return out


def best_scores(ranked: list[dict], item: BenchItem, kind: str, doc_ids: dict[str, int]) -> dict[str, float]:
    """Scores against each alternative gold set, keeping the best value of every metric."""
    scores = [score_ranks(unit_ranks(ranked, units, kind, doc_ids))
              for units in (item.unit_sets or [item.units])]
    return {m: max(s[m] for s in scores) for m in scores[0]}


def auc(positives: list[float], negatives: list[float]) -> float | None:
    if not positives or not negatives:
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in positives for n in negatives]))


def paired_ci(a: list[float], b: list[float], seed: int = 0, n: int = 2000) -> tuple[float, float, float]:
    diff = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = [rng.choice(diff, len(diff)).mean() for _ in range(n)]
    return float(diff.mean()), *(float(x) for x in np.percentile(means, [2.5, 97.5]))


async def eval_variant(client, bench: Benchmark, doc_ids, items, mode, reranker, top_k, concurrency) -> dict:
    sem = asyncio.Semaphore(concurrency)

    async def one(item: BenchItem) -> dict:
        async with sem:
            started = time.perf_counter()
            body = {"query": item.question[:8000], "top_k": top_k, "mode": mode, "reranker": reranker}
            if item.scope is not None:
                body["document_ids"] = [doc_ids[item.scope]]
            resp = await client.post("/search", json=body)
            latency = (time.perf_counter() - started) * 1000
        resp.raise_for_status()
        ranked = resp.json()
        row = {"id": item.id, "type": item.type, "latency_ms": latency,
               "context_tokens@5": sum(approx_tokens(c["text"]) for c in ranked[:5]),
               "top1": ranked[0]["score"] if ranked else None, "n": len(ranked)}
        if item.units:
            row.update(best_scores(ranked, item, bench.unit_kind, doc_ids))
        return row

    # A scoped question whose document produced no chunks has nothing to search.
    items = [i for i in items if i.scope is None or i.scope in doc_ids]
    rows = await asyncio.gather(*(one(i) for i in items))
    gold = [r for r in rows if "mrr@10" in r]
    summary = {m: mean_or_none([r[m] for r in gold])
               for m in [f"recall@{k}" for k in K_VALUES] + ["complete@5", "mrr@10", "ndcg@10"]}
    latencies = [r["latency_ms"] for r in rows]
    summary |= {"n": len(gold), "latency_ms_p50": round(quantiles(latencies, n=100)[49]),
                "latency_ms_p95": round(quantiles(latencies, n=100)[94]),
                "no_hits": sum(r["n"] == 0 for r in rows),
                "context_tokens@5": round(mean_or_none([r["context_tokens@5"] for r in rows])),
                "null_auc": auc([r["top1"] or 0.0 for r in gold],
                                [r["top1"] or 0.0 for r in rows if "mrr@10" not in r])}
    return {"summary": summary, "items": rows}


def unreachable_quotes(bench: Benchmark, settings: Settings) -> int:
    """Quotes split across a chunk boundary: no retriever can score them, a ceiling on recall."""
    if bench.unit_kind != "quote":
        return 0
    joined = "\x00".join(normalize(c) for text in bench.docs.values()
                         for c in chunk_text(text, settings.chunk_tokens, settings.chunk_overlap_tokens))
    return sum(normalize(u) not in joined for u in {u for i in bench.items for u in i.units})


async def run_dataset(name: str, args) -> dict:
    print(f"{name}: loading", flush=True)
    bench = LOADERS[name]()
    settings = Settings(db_schema=f"bench_{name}{args.schema_suffix}", llm_provider="fake",
                        rate_limit_enabled=False, environment="development", api_keys="",
                        chunk_tokens=args.chunk_tokens, chunk_overlap_tokens=args.chunk_overlap_tokens)
    items = bench.items
    if args.limit and len(items) > args.limit:
        items = random.Random(args.seed).sample(items, args.limit)
    if not args.reuse:
        await init_schema(settings, reset=True)
    app = create_app(settings)
    out = {"n_docs": len(bench.docs), "n_questions": len(items),
           "unreachable_quotes": unreachable_quotes(bench, settings), "variants": {}}
    async with app.router.lifespan_context(app):
        if args.reuse:
            doc_ids = await existing_doc_ids(app, bench)
        else:
            doc_ids = await ingest(app, settings, bench.docs)
        async with tenant_connection(app.state.pool, DEFAULT_TENANT_ID) as conn:
            out["n_chunks"] = (await (await conn.execute("SELECT count(*) AS n FROM chunks")).fetchone())["n"]
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://bench", timeout=600) as client:
            for mode in args.modes.split(","):
                for reranker in args.rerankers.split(","):
                    # Rerank at the agent's real top_k too: the candidate pool depends on top_k.
                    for top_k in ((10,) if reranker == "none" else (5, 10)):
                        variant = mode if reranker == "none" else f"{mode}+{reranker}@{top_k}"
                        print(f"  {variant}", flush=True)
                        out["variants"][variant] = await eval_variant(
                            client, bench, doc_ids, items, mode, reranker, top_k, args.concurrency)
    return out


def _gold(result: dict, variant: str, metric: str) -> list[float]:
    return [x[metric] for x in result["variants"][variant]["items"] if metric in x]


def to_markdown(results: dict, config: dict) -> str:
    lines = [f"### Public retrieval benchmarks - {config['timestamp']}", "",
             f"embedder `{config['embedder']}`, rerank `{config['rerank_model']}`, "
             f"chunk {config['chunk_tokens']}/{config['chunk_overlap_tokens']} tokens", ""]
    for name, r in results.items():
        lines += [f"**{name}**: {r['n_docs']} docs, {r['n_chunks']} chunks, {r['n_questions']} questions, "
                  f"{r['unreachable_quotes']} quotes split by chunking", "",
                  "| variant | recall@1 | recall@5 | recall@10 | complete@5 | MRR@10 | nDCG@10 "
                  "| p50 ms | null AUC |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for variant, v in r["variants"].items():
            s = v["summary"]
            cols = ["recall@1", "recall@5", "recall@10", "complete@5", "mrr@10", "ndcg@10"]
            auc_text = "-" if s["null_auc"] is None else f"{s['null_auc']:.3f}"
            lines.append(f"| {variant} | " + " | ".join(f"{s[c]:.3f}" for c in cols)
                         + f" | {s['latency_ms_p50']} | {auc_text} |")
        pairs = [("hybrid", "vector"), ("hybrid", "fts"), ("vector", "fts"), ("hybrid+cohere@5", "hybrid")]
        lines.append("")
        for a, b in pairs:
            if a in r["variants"] and b in r["variants"]:
                mean, low, high = paired_ci(_gold(r, a, "recall@5"), _gold(r, b, "recall@5"))
                mrr, mlow, mhigh = paired_ci(_gold(r, a, "mrr@10"), _gold(r, b, "mrr@10"))
                lines.append(f"- {a} - {b}: recall@5 {mean:+.3f} [{low:+.3f}, {high:+.3f}], "
                             f"MRR@10 {mrr:+.3f} [{mlow:+.3f}, {mhigh:+.3f}]")
        lines.append("")
    return "\n".join(lines)


async def run(args) -> None:
    if args.rerank_url:
        CohereReranker.URL = args.rerank_url
    results = {name: await run_dataset(name, args) for name in args.datasets.split(",")}
    settings = Settings()
    config = {"timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ"),
              "embedder": settings.embedding_model, "rerank_model": settings.rerank_model,
              "chunk_tokens": args.chunk_tokens, "chunk_overlap_tokens": args.chunk_overlap_tokens,
              "limit": args.limit, "seed": args.seed}
    out = Path(args.output) if args.output else RESULTS_DIR / f"bench-{config['timestamp']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": config, "results": results}), encoding="utf-8")
    markdown = to_markdown(results, config)
    out.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m evals.benchmarks", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    r = sub.add_parser("run")
    r.add_argument("--datasets", default="hotpot,multihop,fiqa,techqa,qasper,qasper_lib")
    r.add_argument("--modes", default="vector,fts,hybrid")
    r.add_argument("--rerankers", default="none", help="comma list, e.g. none,cohere")
    r.add_argument("--limit", type=int, default=1000, help="sample at most N questions per dataset")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--chunk-tokens", type=int, default=500)
    r.add_argument("--chunk-overlap-tokens", type=int, default=60)
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--reuse", action="store_true", help="keep the bench_<name> schemas already ingested")
    r.add_argument("--schema-suffix", default="", help="keep other ingests apart, e.g. _c300")
    r.add_argument("--rerank-url", help="override the Cohere endpoint, e.g. a local server")
    r.add_argument("--output", help="results JSON path (default: evals/results/bench-<timestamp>.json)")
    args = p.parse_args()
    if args.command == "download":
        download()
    else:
        run_async(run(args))


if __name__ == "__main__":
    main()
