"""QASPER under its official protocol: the DocPilot agent answers every question of a split,
and the official evaluator scores the answers and the evidence.

    python -m evals.qasper_official run --llm openai --out evals/results/qasper-dev-qwen
    python -m evals.qasper_official run --llm openai --out evals/results/qasper-dev-qwen --limit 100
    python -m evals.qasper_official score evals/results/qasper-dev-qwen

Official protocol (qasper_evaluator.py, shipped with the test split): for every question the
system predicts an answer string and a list of evidence paragraphs. Answer F1 is SQuAD token
F1 and Evidence F1 is set F1 over exact paragraph strings, each the max over annotators, then
averaged over all questions; a missing prediction scores 0. An annotator who marked the
question unanswerable is a reference with answer "Unanswerable" and no evidence.

How the agent's turn becomes a prediction:
  answer    the answer text without [c:<id>] markers, or "Unanswerable" if the agent refused
            (its refusal also predicts no evidence). A failed turn predicts "" and no evidence.
  evidence  the dataset paragraphs (stripped, as the gold strings are) and table/figure
            captions ("FLOAT SELECTED: <caption>") behind the cited chunks, two ways:
              chunk  every paragraph that overlaps a cited chunk. A 500-token chunk holds
                     several paragraphs, so this is the evidence the agent actually pointed to,
                     at the granularity it points with.
              cited  per citation marker, only the paragraph of that chunk sharing the most
                     words with the sentence the marker ends. What paragraph-level citations
                     would give, without changing the agent.

  --setting paper    (default, the official task) each question is asked about its own paper:
                     search_docs and read_doc see only that document.
  --setting library  the whole split's papers, unfiltered, with the title in the question.
  --style short      (default) the question ends with an instruction to answer in a short
                     phrase or yes/no, as the references are written. Token F1 punishes length,
                     so a full-sentence answer can be right and still score low.
  --style natural    the question as written.

Turns are appended to <out>/turns.jsonl as they finish; rerunning with the same --out skips
questions already answered. `score` writes predictions-{chunk,cited}.jsonl, runs the official
evaluator on each, and writes <out>/summary.md. Papers are the dataset's parsed text, so this
measures the agent and retrieval, not PDF parsing. Uses the bench_qasper[_test] schema from
`python -m evals.benchmarks run --datasets qasper`, and ingests only if that schema is empty.
"""

import argparse
import asyncio
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, quantiles

import httpx

from docpilot.agent.citations import CITATION
from docpilot.agent.prompts import is_refusal
from docpilot.cli import run_async
from docpilot.config import Settings
from docpilot.db import init_schema
from evals.bench_matrix import serve
from evals.benchmarks import (
    DATA_DIR,
    FLOAT_PREFIX,
    Benchmark,
    existing_doc_ids,
    ingest,
    qasper_markdown,
    qasper_papers,
)
from evals.metrics import normalize
from evals.run import parse_ui_stream

SHORT = ("\n\nAnswer as briefly as possible: a short phrase taken from the paper, or just yes or no "
         "when that answers it. Cite your sources as usual.")
EVIDENCE_MODES = ("chunk", "cited")
_WORD = re.compile(r"\w+")


# --- turning an agent turn into a prediction --------------------------------------------------

def paper_units(paper: dict) -> list[tuple[str, str]]:
    """(normalized text, gold-form string) for every paragraph and caption that can be evidence."""
    units = [(normalize(p), p.strip()) for s in paper["full_text"] for p in s["paragraphs"] if p.strip()]
    units += [(normalize(f["caption"]), FLOAT_PREFIX + f["caption"])
              for f in paper["figures_and_tables"] if f["caption"].strip()]
    return units


def chunk_units(chunk_text: str, units: list[tuple[str, str]]) -> list[str]:
    """Gold-form strings of the paragraphs a chunk holds, whole or in part, in chunk order.

    Chunks are built from qasper_markdown, which puts blank lines between paragraphs, so each
    block of a chunk is a whole paragraph, a heading, or (in a long section) part of one.
    """
    out: list[str] = []
    for block in chunk_text.split("\n\n"):
        text = normalize(block)
        if not text or block.lstrip().startswith("#"):
            continue
        for norm, gold in units:
            if (text in norm or norm in text) and gold not in out:
                out.append(gold)
                break
    return out


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


def cited_sentences(answer: str) -> list[tuple[int, str]]:
    """(chunk id, the text that marker ends) for each citation, in order."""
    out, start = [], 0
    for m in CITATION.finditer(answer):
        span = answer[start:m.start()]
        # The claim is the text since the previous sentence end or citation.
        claim = re.split(r"(?<=[.!?])\s+|\n", CITATION.sub(" ", span).strip())[-1]
        out.append((int(m.group(1)), claim or span))
        start = m.end()
    return out


def predict(turn: dict, units: list[tuple[str, str]]) -> dict[str, dict]:
    """Official prediction rows for one turn, one per evidence mode."""
    if turn["error"] or not turn["answer"].strip():
        answer, evidence = "", {m: [] for m in EVIDENCE_MODES}
    elif is_refusal(turn["answer"]):
        answer, evidence = "Unanswerable", {m: [] for m in EVIDENCE_MODES}
    else:
        answer = " ".join(CITATION.sub(" ", turn["answer"]).split())
        chunks = {c["id"]: chunk_units(c["text"], units) for c in turn["citations"]}
        by_chunk = list(dict.fromkeys(u for c in turn["citations"] for u in chunks[c["id"]]))
        by_claim: list[str] = []
        for chunk_id, claim in cited_sentences(turn["answer"]):
            candidates = chunks.get(chunk_id) or []  # ids the app rejected resolve to nothing
            if candidates:
                claim_words = _words(claim)
                best = max(candidates, key=lambda u: len(claim_words & _words(u)))
                if best not in by_claim:
                    by_claim.append(best)
        evidence = {"chunk": by_chunk, "cited": by_claim}
    return {m: {"question_id": turn["question_id"], "predicted_answer": answer,
                "predicted_evidence": evidence[m]} for m in EVIDENCE_MODES}


# --- running ----------------------------------------------------------------------------------

def questions(papers: dict) -> list[tuple[str, dict]]:
    return [(pid, q) for pid, p in papers.items() for q in p["qas"]]


def load_turns(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    turns = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {t["question_id"]: t for t in turns}


async def run(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out / "config.json"
    config = {"split": args.split, "setting": args.setting, "style": args.style, "mode": args.mode,
              "reranker": args.reranker, "llm": args.llm, "limit": args.limit, "seed": args.seed}
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if {k: old.get(k) for k in config} != config:
            sys.exit(f"{args.out} holds a run with other settings {old}; use another --out")
    papers = qasper_papers(args.split)
    todo = questions(papers)
    if args.limit and len(todo) > args.limit:
        todo = random.Random(args.seed).sample(todo, args.limit)
    turns_path = args.out / "turns.jsonl"
    done = load_turns(turns_path)
    todo = [(pid, q) for pid, q in todo if q["question_id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)

    schema = "bench_qasper" if args.split == "dev" else f"bench_qasper_{args.split}"
    settings = Settings(db_schema=schema, rate_limit_enabled=False, environment="development",
                        api_keys="", **({"llm_provider": args.llm} if args.llm else {}))
    await init_schema(settings)
    docs = {pid: qasper_markdown(p) for pid, p in papers.items()}
    async with serve(settings) as (app, client):
        doc_ids = await existing_doc_ids(app, Benchmark(docs, [], "quote"))
        if not doc_ids:
            print(f"{schema} is empty, ingesting", flush=True)
            doc_ids = await ingest(app, settings, docs)
        config |= {"model": app.state.llm.model, "embedder": settings.embedding_model,
                   "rerank_model": settings.rerank_model, "schema": schema,
                   "chunk_tokens": settings.chunk_tokens}
        config_path.write_text(json.dumps(config, indent=1), encoding="utf-8")
        run_id, sem, lock, finished = int(time.time()), asyncio.Semaphore(args.concurrency), asyncio.Lock(), 0

        async def one(pid: str, q: dict) -> None:
            nonlocal finished
            text = q["question"]
            retrieval: dict = {"mode": args.mode, "reranker": args.reranker}
            if args.setting == "paper":
                retrieval["document_ids"] = [doc_ids[pid]]
            else:
                text = f'In the paper "{papers[pid]["title"].strip()}": {text}'
            if args.style == "short":
                text += SHORT
            async with sem:
                started = time.perf_counter()
                try:
                    resp = await client.post("/chat", json={
                        "id": f"qasper-{run_id}-{q['question_id']}", "retrieval": retrieval,
                        "messages": [{"role": "user", "parts": [{"type": "text", "text": text}]}]})
                    resp.raise_for_status()
                    result = parse_ui_stream(resp.text)
                except httpx.HTTPError as e:  # a failed turn is scored, not dropped
                    result = {"answer": "", "citations": [], "trace": None, "finish_reason": None,
                              "error": f"{type(e).__name__}: {e}"}
            trace = result["trace"] or {}
            turn = {"question_id": q["question_id"], "paper_id": pid, "question": text,
                    "answer": result["answer"], "error": result["error"],
                    "finish_reason": result["finish_reason"],
                    "citations": [{"id": c["id"], "document_id": c["document_id"], "text": c["text"]}
                                  for c in result["citations"]],
                    "tool_calls": [t["name"] for t in trace.get("tool_calls", [])],
                    "latency_ms": trace.get("latency_ms") or round((time.perf_counter() - started) * 1000),
                    "input_tokens": trace.get("input_tokens", 0),
                    "output_tokens": trace.get("output_tokens", 0)}
            async with lock:
                with turns_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(turn) + "\n")
                finished += 1
                if finished % 25 == 0 or finished == len(todo):
                    print(f"  {finished}/{len(todo)}", flush=True)

        await asyncio.gather(*(one(pid, q) for pid, q in todo))
    score(args.out)


# --- scoring ----------------------------------------------------------------------------------

def official(predictions: Path, gold: Path, text_only: bool = False) -> dict:
    """Run the dataset's own evaluator, unchanged."""
    cmd = [sys.executable, "-I", str(DATA_DIR / "qasper/qasper_evaluator.py"),
           "--predictions", str(predictions), "--gold", str(gold)]
    out = subprocess.run(cmd + (["--text_evidence_only"] if text_only else []),
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def gold_subset(split: str, ids: set[str], path: Path) -> Path:
    """The split restricted to the questions that were run (for --limit runs)."""
    papers = qasper_papers(split)
    for p in papers.values():
        p["qas"] = [q for q in p["qas"] if q["question_id"] in ids]
    path.write_text(json.dumps({k: p for k, p in papers.items() if p["qas"]}), encoding="utf-8")
    return path


def score(out: Path) -> None:
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    split = config["split"]
    papers = qasper_papers(split)
    turns = load_turns(out / "turns.jsonl")
    units = {pid: paper_units(p) for pid, p in papers.items()}
    preds = {m: [] for m in EVIDENCE_MODES}
    for t in turns.values():
        for m, row in predict(t, units[t["paper_id"]]).items():
            preds[m].append(row)

    n_total = len(questions(papers))
    # A --limit run is scored against the gold of the questions it ran, not the whole split.
    gold = (DATA_DIR / f"qasper/qasper-{split}-v0.3.json" if len(turns) == n_total
            else gold_subset(split, set(turns), out / "gold-subset.json"))
    results = {}
    for m, rows in preds.items():
        path = out / f"predictions-{m}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        results[m] = official(path, gold)
        results[m + "_text_only"] = official(path, gold, text_only=True)

    # The trivial baseline: answer every question "Unanswerable" with no evidence.
    trivial = out / "predictions-unanswerable.jsonl"
    rows = [{"question_id": q, "predicted_answer": "Unanswerable", "predicted_evidence": []} for q in turns]
    trivial.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    results["all_unanswerable"] = official(trivial, gold)

    t = list(turns.values())
    latencies = [x["latency_ms"] for x in t if x["latency_ms"]]
    diag = {
        "questions": len(t), "of_split": n_total,
        "refused": sum(is_refusal(x["answer"]) for x in t), "errors": sum(bool(x["error"]) for x in t),
        "no_search": sum(not {"search_docs", "read_doc"} & set(x["tool_calls"]) for x in t),
        "read_doc_used": sum("read_doc" in x["tool_calls"] for x in t),
        "answer_words_mean": round(mean(len(CITATION.sub(" ", x["answer"]).split()) for x in t), 1),
        "evidence_per_answer": {m: round(mean(len(r["predicted_evidence"]) for r in preds[m]), 2)
                                for m in EVIDENCE_MODES},
        "latency_ms_p50": round(quantiles(latencies, n=100)[49]) if len(latencies) > 1 else None,
        "latency_ms_p95": round(quantiles(latencies, n=100)[94]) if len(latencies) > 1 else None,
        "finish_reasons": dict(Counter(x["finish_reason"] for x in t)),
    }
    summary = {"config": config, "official": results, "diagnostics": diag,
               "scored_at": datetime.now(UTC).isoformat(timespec="seconds")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    md = to_markdown(summary)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)


def to_markdown(summary: dict) -> str:
    c, r, d = summary["config"], summary["official"], summary["diagnostics"]
    by_type = ["extractive", "abstractive", "boolean", "none"]
    lines = [f"### QASPER {c['split']}, official evaluator - {summary['scored_at']}", "",
             f"setting `{c['setting']}`, style `{c['style']}`, retrieval `{c['mode']}`"
             f"{'' if c['reranker'] == 'none' else '+' + c['reranker']}; LLM `{c.get('model')}`, "
             f"embedder `{c.get('embedder')}`, chunk {c.get('chunk_tokens')} tokens. "
             f"{d['questions']} of {d['of_split']} questions.", "",
             "| prediction | Answer F1 | " + " | ".join(f"F1 {t}" for t in by_type)
             + " | Evidence F1 | Evidence F1 (text only) |",
             "|---|---|" + "---|" * len(by_type) + "---|---|"]
    rows = [(f"agent, evidence={m}", r[m], r[m + "_text_only"]) for m in EVIDENCE_MODES]
    rows.append(("all \"Unanswerable\"", r["all_unanswerable"], None))
    for name, res, text_only in rows:
        types = res["Answer F1 by type"]
        lines.append(f"| {name} | {res['Answer F1']:.3f} | "
                     + " | ".join(f"{types[t]:.3f}" for t in by_type)
                     + f" | {res['Evidence F1']:.3f} | "
                     + ("-" if text_only is None else f"{text_only['Evidence F1']:.3f}") + " |")
    lines += ["", "Answer F1 has one type per question: the type of the annotator it scored best "
                  "against, so the per-type columns are not fixed question subsets.", "",
              f"- refused {d['refused']}, errors {d['errors']}, answered without any tool {d['no_search']}, "
              f"read_doc used {d['read_doc_used']}",
              f"- answer length {d['answer_words_mean']} words on average; evidence paragraphs per "
              f"answer: " + ", ".join(f"{m} {v}" for m, v in d["evidence_per_answer"].items()),
              f"- latency p50 {d['latency_ms_p50']} ms, p95 {d['latency_ms_p95']} ms; "
              f"finish reasons {d['finish_reasons']}", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m evals.qasper_official", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--split", choices=["dev", "test"], default="dev")
    r.add_argument("--setting", choices=["paper", "library"], default="paper")
    r.add_argument("--style", choices=["short", "natural"], default="short")
    r.add_argument("--mode", choices=["vector", "fts", "hybrid"], default="hybrid")
    r.add_argument("--reranker", choices=["none", "cohere"], default="none")
    r.add_argument("--llm", help="LLM provider, e.g. openai (default: LLM_PROVIDER)")
    r.add_argument("--limit", type=int, default=0, help="sample N questions (0 = all)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--concurrency", type=int, default=8)
    s = sub.add_parser("score")
    s.add_argument("out", type=Path)
    args = p.parse_args()
    if args.command == "run":
        run_async(run(args))
    else:
        score(args.out)


if __name__ == "__main__":
    main()
