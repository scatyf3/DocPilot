"""Eval runner. Replays the labeled questions through the real API, in process.

    python -m evals.run --retrieval-only                 # recall@5 / MRR per retrieval mode
    python -m evals.run --modes vector,hybrid --rerankers none,cohere   # ablation table
    python -m evals.run                                  # + full answers via /chat + LLM judge

Steps: reset an isolated Postgres schema (DB_SCHEMA=docpilot_eval by default), upload the
corpus through POST /documents, score retrieval through POST /search, and (unless
--retrieval-only) stream answers from POST /chat, the same endpoint the web app uses.
Writes evals/results/<timestamp>.json and a Markdown summary next to it.
"""

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import quantiles

import httpx

from docpilot.agent.prompts import is_refusal
from docpilot.app import create_app
from docpilot.cli import run_async
from docpilot.config import Settings
from docpilot.db import init_schema
from evals.dataset import CORPUS_DIR, EVALS_DIR, Item, load_dataset
from evals.metrics import contains, mean_or_none, recall_at_k, reciprocal_rank

K = 5
logging.getLogger("httpx").setLevel(logging.WARNING)  # one log line per request is noise here
RESULTS_DIR = EVALS_DIR / "results"


async def ingest_corpus(client: httpx.AsyncClient) -> int:
    paths = sorted(CORPUS_DIR.glob("*.md"))
    for path in paths:
        resp = await client.post("/documents", files={"file": (path.name, path.read_bytes())})
        resp.raise_for_status()
    return len(paths)


async def eval_retrieval(client, items: list[Item], mode: str, reranker: str, sem) -> dict:
    async def one(item: Item) -> dict:
        async with sem:
            resp = await client.post("/search", json={"query": item.question, "top_k": 10,
                                                      "mode": mode, "reranker": reranker})
        resp.raise_for_status()
        texts = [c["text"] for c in resp.json()]
        quotes = [e.quote for e in item.evidence]
        return {"id": item.id, "type": item.type, "recall@5": recall_at_k(texts, quotes, K),
                "rr": reciprocal_rank(texts, quotes)}

    rows = await asyncio.gather(*(one(i) for i in items if i.answerable))
    summary = {"recall@5": mean_or_none([r["recall@5"] for r in rows]),
               "mrr": mean_or_none([r["rr"] for r in rows]), "n": len(rows)}
    for t in ("single", "multi"):
        summary[f"recall@5_{t}"] = mean_or_none([r["recall@5"] for r in rows if r["type"] == t])
    return {"summary": summary, "items": rows}


def parse_ui_stream(body: str) -> dict:
    """Rebuild answer text, citations and trace from the UI message stream."""
    parts: dict[str, list[str]] = {}
    out = {"answer": "", "citations": [], "trace": None, "error": None}
    for event in body.split("\n\n"):
        if not event.startswith("data: ") or event == "data: [DONE]":
            continue
        chunk = json.loads(event[len("data: "):])
        kind = chunk["type"]
        if kind == "text-delta":
            parts.setdefault(chunk["id"], []).append(chunk["delta"])
        elif kind == "data-citations":
            out["citations"] = chunk["data"]
        elif kind == "data-trace":
            out["trace"] = chunk["data"]
        elif kind == "error":
            out["error"] = chunk["errorText"]
    out["answer"] = "\n\n".join("".join(p) for p in parts.values())
    return out


async def eval_answers(client, items: list[Item], mode: str, reranker: str, judge, sem) -> dict:
    run_id = int(time.time())

    async def one(item: Item) -> dict:
        async with sem:
            resp = await client.post("/chat", json={
                "id": f"eval-{run_id}-{item.id}",
                "messages": [{"role": "user", "parts": [{"type": "text", "text": item.question}]}],
                "retrieval": {"mode": mode, "reranker": reranker},
            })
            resp.raise_for_status()
            result = parse_ui_stream(resp.text)
            verdict = None
            if judge and not result["error"]:
                try:
                    verdict = await judge.grade(item.question, item.answer if item.answerable else None,
                                                result["answer"], result["citations"])
                except Exception as e:  # one failed grade shouldn't sink the run
                    verdict = {"error": f"{type(e).__name__}: {e}"}
        quotes = [e.quote for e in item.evidence]
        trace = result["trace"] or {}
        refused = is_refusal(result["answer"]) or bool(verdict and verdict.get("refused"))
        return {
            "id": item.id, "type": item.type, "question": item.question,
            "reference": item.answer, "answer": result["answer"], "error": result["error"],
            "citations": [{"id": c["id"], "title": c["title"], "text": c["text"]}
                          for c in result["citations"]],
            "refused": refused,
            "cited_evidence": any(contains(c["text"], q) for c in result["citations"] for q in quotes),
            "verdict": verdict,
            "latency_ms": trace.get("latency_ms"), "cost_usd": trace.get("cost_usd", 0.0),
            "input_tokens": trace.get("input_tokens", 0), "output_tokens": trace.get("output_tokens", 0),
        }

    rows = await asyncio.gather(*(one(i) for i in items))
    return {"summary": summarize_answers(rows), "items": rows}


def summarize_answers(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    attempted = [r for r in answerable if not r["refused"]]

    def judged(rs, key):
        return mean_or_none([float(r["verdict"][key]) for r in rs
                             if r["verdict"] and key in r["verdict"]])

    latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    return {
        "n": len(rows),
        "errors": sum(1 for r in rows if r["error"]),
        "correctness": judged(answerable, "correct"),
        "citation_faithfulness": judged(attempted, "citations_faithful"),
        "refusal_accuracy": mean_or_none([float(r["refused"]) for r in unanswerable]),
        "false_refusal_rate": mean_or_none([float(r["refused"]) for r in answerable]),
        "cited_evidence_rate": mean_or_none([float(r["cited_evidence"]) for r in attempted]),
        "latency_ms_p50": round(quantiles(latencies, n=100)[49]) if len(latencies) > 1 else None,
        "latency_ms_p95": round(quantiles(latencies, n=100)[94]) if len(latencies) > 1 else None,
        "cost_usd_total": round(sum(r["cost_usd"] or 0 for r in rows), 4),
    }


def to_markdown(result: dict) -> str:
    cfg = result["config"]
    lines = [f"### DocPilot eval - {result['timestamp']}", "",
             f"embedder `{cfg['embedder']}`, llm `{cfg['llm']}`, chunk {cfg['chunk_tokens']}/"
             f"{cfg['chunk_overlap_tokens']} tokens, {cfg['n_docs']} docs, {cfg['n_questions']} questions",
             "", "| retrieval | recall@5 | MRR | recall@5 single | recall@5 multi |",
             "|---|---|---|---|---|"]
    for name, r in result["retrieval"].items():
        s = r["summary"]
        lines.append(f"| {name} | {_fmt(s['recall@5'])} | {_fmt(s['mrr'])} | "
                     f"{_fmt(s['recall@5_single'])} | {_fmt(s['recall@5_multi'])} |")
    if result.get("answers"):
        s = result["answers"]["summary"]
        lines += ["", f"Answers ({cfg['answer_variant']}, judge `{cfg['judge']}`):", "",
                  "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {_fmt(v)} |" for k, v in s.items()]
    return "\n".join(lines) + "\n"


def _fmt(v) -> str:
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


async def main_async(args) -> Path:
    overrides = {"db_schema": args.schema}
    if args.embedder:
        overrides["embedding_provider"] = args.embedder
    if args.llm:
        overrides["llm_provider"] = args.llm
    settings = Settings(**overrides)
    items = load_dataset()[: args.limit]
    judge = None
    if not args.retrieval_only and not args.no_judge and os.environ.get("ANTHROPIC_API_KEY"):
        from evals.judge import Judge
        judge = Judge()

    await init_schema(settings, reset=True)
    app = create_app(settings)
    sem = asyncio.Semaphore(args.concurrency)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://evals", timeout=600) as client:
            n_docs = await ingest_corpus(client)
            retrieval = {}
            for mode in args.modes.split(","):
                for reranker in args.rerankers.split(","):
                    name = mode if reranker == "none" else f"{mode}+{reranker}"
                    print(f"retrieval: {name}")
                    retrieval[name] = await eval_retrieval(client, items, mode, reranker, sem)
            answers = None
            if not args.retrieval_only:
                print(f"answers: {len(items)} questions via /chat ({args.answer_mode})")
                answers = await eval_answers(client, items, args.answer_mode,
                                             args.answer_reranker, judge, sem)

    answer_variant = (args.answer_mode if args.answer_reranker == "none"
                      else f"{args.answer_mode}+{args.answer_reranker}")
    result = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ"),
        "config": {
            "embedder": app.state.embedder.name,
            "llm": None if args.retrieval_only else app.state.llm.model,
            "judge": judge.model if judge else None, "answer_variant": answer_variant,
            "chunk_tokens": settings.chunk_tokens, "chunk_overlap_tokens": settings.chunk_overlap_tokens,
            "n_docs": n_docs, "n_questions": len(items),
        },
        "retrieval": retrieval,
        "answers": answers,
    }
    out = Path(args.output) if args.output else RESULTS_DIR / f"{result['timestamp']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    markdown = to_markdown(result)
    out.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--retrieval-only", action="store_true", help="skip /chat answers (no LLM cost)")
    p.add_argument("--modes", default="vector,fts,hybrid", help="retrieval modes to score")
    p.add_argument("--rerankers", default="none", help="comma list, e.g. none,cohere")
    p.add_argument("--answer-mode", default="hybrid", help="retrieval mode used for /chat answers")
    p.add_argument("--answer-reranker", default="none")
    p.add_argument("--embedder", choices=["openai", "fake"], help="override EMBEDDING_PROVIDER")
    p.add_argument("--llm", choices=["anthropic", "openai", "fake"], help="override LLM_PROVIDER")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    p.add_argument("--schema", default="docpilot_eval", help="Postgres schema to (re)create")
    p.add_argument("--limit", type=int, default=None, help="only the first N questions")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--output", help="results JSON path (default: evals/results/<timestamp>.json)")
    run_async(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
