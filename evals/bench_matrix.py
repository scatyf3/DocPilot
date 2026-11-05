"""End-to-end answer eval on the public benchmarks that come with short reference answers,
crossed with every retrieval setting: one table per dataset with retrieval accuracy, answer
accuracy and latency.

    python -m evals.bench_matrix --llm openai --out evals/results/matrix-qwen
    python -m evals.bench_matrix --llm openai --out evals/results/matrix-qwen --combos hybrid,closed-book
    python -m evals.bench_matrix --out evals/results/matrix-qwen --report-only

Every dataset (multihop, hotpot; FiQA and TechQA have no short answers) is crossed with every
combination of retrieval mode (vector = embeddings only, fts = keyword/BM25 only, hybrid = both
fused) and reranker (none, cohere). All cells use the same seeded question sample, so they can be
compared question by question:

  retrieval  POST /search at the agent's own top_k (5): recall@1/5, complete@5, MRR, latency
  answers    POST /chat, the same agent the web app uses, with that retrieval setting

`closed-book` is an optional extra row: the agent over an empty library, which measures how much
the model answers from memory (HotpotQA is Wikipedia). Add it to --combos to run it.

Answer scoring is deterministic, so no LLM judge is involved:

  answerable  correct if the agent did not refuse and the normalized reference answer appears
              in the normalized answer as whole tokens. For yes/no references the first yes or
              no in the answer must match. Lenient: an answer listing many candidates can match.
  null query  MultiHop-RAG questions with no evidence in the corpus: correct only if refused.

`always_yes_accuracy` is what answering "yes" to everything would score on the same sample;
most comparison and temporal questions are yes/no.

Each cell goes to <out>/<dataset>__<combo>.json as soon as it finishes, and <out>/summary.md is
rebuilt after every cell. Rerunning with the same --out skips finished cells, so an interrupted
run resumes where it stopped. Delete a cell file to rerun it.

Uses the bench_<name> schemas from `python -m evals.benchmarks run`, and ingests only into a schema
that has no documents yet, so it never wipes an existing ingest. Run one matrix at a time: the
cells share the LLM, embedding and rerank servers, and parallel runs would distort the latencies.
"""

import argparse
import asyncio
import json
import random
import string
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from statistics import quantiles

import httpx

from docpilot.agent.citations import CITATION
from docpilot.agent.prompts import is_refusal
from docpilot.app import create_app
from docpilot.cli import run_async
from docpilot.config import Settings
from docpilot.db import init_schema
from docpilot.retrieval.rerank import CohereReranker
from evals.benchmarks import (
    LOADERS,
    RESULTS_DIR,
    BenchItem,
    eval_variant,
    existing_doc_ids,
    ingest,
    paired_ci,
)
from evals.metrics import contains, mean_or_none
from evals.run import parse_ui_stream

ANSWER_DATASETS = ("multihop", "hotpot")
MODES = ("vector", "fts", "hybrid")
RERANKERS = ("none", "cohere")
BASELINE = "hybrid"  # the app's default retrieval; other cells are compared against it
CLOSED_BOOK = "closed-book"
YES_NO = {"yes", "no"}


# --- answer scoring ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """SQuAD-style: drop citation markers, case, punctuation and articles."""
    text = CITATION.sub(" ", text).lower()
    text = "".join(" " if ch in string.punctuation else ch for ch in text)
    return " ".join(w for w in text.split() if w not in ("a", "an", "the"))


def answer_matches(answer: str, reference: str) -> bool:
    ref, got = normalize_answer(reference), normalize_answer(answer)
    if not ref:
        return False
    if ref in YES_NO:
        first = next((w for w in got.split() if w in YES_NO), None)
        return first == ref
    return f" {ref} " in f" {got} "


def llm_ms(trace: dict) -> int | None:
    """Turn time minus tool time: the model's share. Tools that ran in parallel are summed,
    so this slightly undercounts it (by under a second at p50 in the 2026-11 runs)."""
    if trace.get("latency_ms") is None:
        return None
    return trace["latency_ms"] - sum(t.get("duration_ms", 0) for t in trace.get("tool_calls", []))


def cited_gold(citations: list[dict], item: BenchItem, kind: str) -> bool:
    if kind == "doc":  # ingest stores the document key as the title
        return any(c["title"] in item.units for c in citations)
    return any(contains(c["text"], u) for c in citations for u in item.units)


async def answer_items(client, items: list[BenchItem], kind: str, mode: str, reranker: str,
                       concurrency: int) -> list[dict]:
    sem, run_id = asyncio.Semaphore(concurrency), int(time.time())

    async def one(item: BenchItem) -> dict:
        async with sem:
            try:
                resp = await client.post("/chat", json={
                    "id": f"bench-{run_id}-{item.id}",
                    "messages": [{"role": "user", "parts": [{"type": "text", "text": item.question}]}],
                    "retrieval": {"mode": mode, "reranker": reranker},
                })
                resp.raise_for_status()
                result = parse_ui_stream(resp.text)
            except httpx.HTTPError as e:  # count it as a failed turn rather than sinking the run
                result = {"answer": "", "citations": [], "trace": None, "finish_reason": None,
                          "error": f"{type(e).__name__}: {e}"}
        trace = result["trace"] or {}
        answerable = bool(item.units)
        refused = is_refusal(result["answer"])
        correct = (not refused and answer_matches(result["answer"], item.answer)) if answerable else refused
        return {
            "id": item.id, "type": item.type, "answerable": answerable, "question": item.question,
            "reference": item.answer, "answer": result["answer"], "error": result["error"],
            "refused": refused, "correct": correct,
            "cited_gold": cited_gold(result["citations"], item, kind) if answerable else None,
            "citations": [{"id": c["id"], "title": c["title"]} for c in result["citations"]],
            "finish_reason": result["finish_reason"],
            "tool_calls": [t["name"] for t in trace.get("tool_calls", [])],
            "ungrounded": trace.get("ungrounded", False),
            "latency_ms": trace.get("latency_ms"), "llm_ms": llm_ms(trace),
            "cost_usd": trace.get("cost_usd", 0.0),
            "input_tokens": trace.get("input_tokens", 0), "output_tokens": trace.get("output_tokens", 0),
        }

    return await asyncio.gather(*(one(i) for i in items))


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    null = [r for r in rows if not r["answerable"]]
    attempted = [r for r in answerable if not r["refused"]]
    latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    return {
        "n": len(rows), "n_null": len(null), "errors": sum(1 for r in rows if r["error"]),
        "accuracy": mean_or_none([float(r["correct"]) for r in answerable]),
        "accuracy_by_type": {t: mean_or_none([float(r["correct"]) for r in answerable if r["type"] == t])
                             for t in sorted({r["type"] for r in answerable})},
        "always_yes_accuracy": mean_or_none([float(normalize_answer(r["reference"]) == "yes")
                                             for r in answerable]),
        "refusal_accuracy": mean_or_none([float(r["refused"]) for r in null]),
        "false_refusal_rate": mean_or_none([float(r["refused"]) for r in answerable]),
        "cited_gold_rate": mean_or_none([float(r["cited_gold"]) for r in attempted]),
        "no_search_rate": mean_or_none([float("search_docs" not in r["tool_calls"]) for r in rows]),
        "incomplete_rate": mean_or_none([float(bool(r["error"]) or r["finish_reason"] not in (None, "stop"))
                                         for r in rows]),
        "ungrounded_rate": mean_or_none([float(r["ungrounded"]) for r in rows]),
        "latency_ms_p50": round(quantiles(latencies, n=100)[49]) if len(latencies) > 1 else None,
        "latency_ms_p95": round(quantiles(latencies, n=100)[94]) if len(latencies) > 1 else None,
        "cost_usd_total": round(sum(r["cost_usd"] or 0 for r in rows), 4),
    }


# --- running cells ----------------------------------------------------------------------------

def combo_name(mode: str, reranker: str) -> str:
    return mode if reranker == "none" else f"{mode}+{reranker}"


ALL_COMBOS = [combo_name(m, r) for r in RERANKERS for m in MODES]
ROWS = [*ALL_COMBOS, CLOSED_BOOK]  # table order


def parse_combo(name: str) -> tuple[str, str]:
    mode, _, reranker = name.partition("+")
    if mode not in MODES or (reranker or "none") not in RERANKERS:
        raise ValueError(f"unknown combination {name!r}; expected one of {ROWS}")
    return mode, reranker or "none"


def cell_path(out: Path, dataset: str, combo: str) -> Path:
    return out / f"{dataset}__{combo}.json"


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)  # a crash mid-write must not leave a cell that looks finished


def bench_settings(name: str, suffix: str, args) -> Settings:
    overrides = {"db_schema": f"bench_{name}{suffix}", "rate_limit_enabled": False,
                 "environment": "development", "api_keys": "",
                 "chunk_tokens": args.chunk_tokens, "chunk_overlap_tokens": args.chunk_overlap_tokens}
    if args.llm:
        overrides["llm_provider"] = args.llm
    return Settings(**overrides)


@asynccontextmanager
async def serve(settings: Settings):
    """The real app in process, plus a client for it."""
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://bench", timeout=1800) as client:
            yield app, client


def save_cell(args, name: str, combo: str, settings: Settings, app, items: list[BenchItem],
              n_docs: int, retrieval: dict | None, rows: list[dict] | None, started: float) -> None:
    mode, reranker = ("hybrid", "none") if combo == CLOSED_BOOK else parse_combo(combo)
    elapsed = round(time.perf_counter() - started)
    cell = {"dataset": name, "combo": combo, "mode": mode, "reranker": reranker,
            "n_docs": n_docs, "n_questions": len(items), "top_k": args.top_k, "seed": args.seed,
            "llm": app.state.llm.model if rows is not None else None,
            "embedder": settings.embedding_model, "rerank_model": settings.rerank_model,
            "retrieval": retrieval,
            "answers": None if rows is None else {"summary": summarize(rows), "items": rows},
            "elapsed_s": elapsed, "finished_at": datetime.now(UTC).isoformat(timespec="seconds")}
    write_json(cell_path(args.out, name, combo), cell)
    print(f"  {combo}: done in {elapsed / 60:.1f} min", flush=True)
    write_report(args.out)


async def run_dataset(name: str, combos: list[str], args) -> None:
    todo = [c for c in combos if not cell_path(args.out, name, c).exists()]
    if args.retrieval_only and CLOSED_BOOK in todo:
        todo.remove(CLOSED_BOOK)  # nothing to retrieve from
    if not todo:
        print(f"{name}: all requested cells done", flush=True)
        return
    print(f"{name}: loading", flush=True)
    bench = LOADERS[name]()
    items = bench.items
    if args.limit and len(items) > args.limit:
        items = random.Random(args.seed).sample(items, args.limit)

    open_book = [c for c in todo if c != CLOSED_BOOK]
    if open_book:
        settings = bench_settings(name, args.schema_suffix, args)
        await init_schema(settings)  # creates the schema if missing, applies pending migrations
        async with serve(settings) as (app, client):
            doc_ids = await existing_doc_ids(app, bench)
            if not doc_ids:
                print(f"  {settings.db_schema} is empty, ingesting", flush=True)
                doc_ids = await ingest(app, settings, bench.docs)
            for combo in open_book:
                mode, reranker = parse_combo(combo)
                started = time.perf_counter()
                print(f"  {combo}: retrieval ({len(items)} questions)", flush=True)
                retrieval = await eval_variant(client, bench, doc_ids, items, mode, reranker,
                                               args.top_k, args.concurrency)
                rows = None
                if not args.retrieval_only:
                    print(f"  {combo}: answers", flush=True)
                    rows = await answer_items(client, items, bench.unit_kind, mode, reranker,
                                              args.concurrency)
                save_cell(args, name, combo, settings, app, items, len(doc_ids), retrieval, rows, started)

    if CLOSED_BOOK in todo:
        settings = bench_settings(name, "_closed", args)
        await init_schema(settings, reset=True)  # always an empty library
        async with serve(settings) as (app, client):
            started = time.perf_counter()
            print(f"  {CLOSED_BOOK}: answers", flush=True)
            rows = await answer_items(client, items, bench.unit_kind, "hybrid", "none", args.concurrency)
            save_cell(args, name, CLOSED_BOOK, settings, app, items, 0, None, rows, started)


# --- report -----------------------------------------------------------------------------------

def load_cells(out: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("*__*.json"))]


def _paired(a: dict, b: dict, section: str, metric: str, keep) -> tuple[float, float, float] | None:
    """Paired bootstrap CI of a - b over the questions both cells scored."""
    if not a.get(section) or not b.get(section):
        return None
    xs = {r["id"]: r for r in a[section]["items"] if keep(r)}
    ys = {r["id"]: r for r in b[section]["items"] if keep(r)}
    ids = sorted(xs.keys() & ys.keys())
    if len(ids) < 2:
        return None
    return paired_ci([float(xs[i][metric]) for i in ids], [float(ys[i][metric]) for i in ids])


def _f(v, digits: int = 3) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _seconds(ms) -> str:
    return "-" if ms is None else f"{ms / 1000:.1f}"


def matrix_markdown(cells: list[dict]) -> str:
    """One table per dataset, rows in ROWS order, plus paired CIs against BASELINE."""
    if not cells:
        return "No finished cells yet.\n"
    first = next((c for c in cells if c["retrieval"]), cells[0])
    lines = ["### Retrieval x reranker matrix", "",
             f"llm `{first['llm']}`, embedder `{first['embedder']}`, reranker `{first['rerank_model']}`, "
             f"search top_k {first['top_k']}, seed {first['seed']}", "",
             "Retrieval columns score POST /search; answer columns score POST /chat with the same "
             "setting. Latencies are p50 / p95 under the run's concurrency, so compare them within "
             "this table only.", ""]
    for dataset in sorted({c["dataset"] for c in cells}, key=lambda d: (d not in ANSWER_DATASETS, d)):
        by_combo = {c["combo"]: c for c in cells if c["dataset"] == dataset}
        some = max(by_combo.values(), key=lambda c: c["n_docs"])
        k = some["top_k"]
        lines += [f"**{dataset}**: {some['n_questions']} questions, {some['n_docs']} docs", "",
                  f"| combo | recall@1 | recall@5 | complete@5 | MRR@{k} | search ms | "
                  "answer acc | cited gold | refusal acc | false refusal | answer s | errors | min |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for combo in ROWS:
            cell = by_combo.get(combo)
            if cell is None:
                if combo != CLOSED_BOOK:  # optional row: only shown once it has run
                    lines.append(f"| {combo} | (pending) |" + " |" * 11)
                continue
            r = (cell["retrieval"] or {}).get("summary") or {}
            a = (cell["answers"] or {}).get("summary") or {}
            search = f"{r['latency_ms_p50']} / {r['latency_ms_p95']}" if r else "-"
            lines.append(
                f"| {combo} | {_f(r.get('recall@1'))} | {_f(r.get('recall@5'))} | {_f(r.get('complete@5'))} "
                f"| {_f(r.get('mrr@10'))} | {search} "
                f"| {_f(a.get('accuracy'))} | {_f(a.get('cited_gold_rate'))} "
                f"| {_f(a.get('refusal_accuracy'))} | {_f(a.get('false_refusal_rate'))} "
                f"| {_seconds(a.get('latency_ms_p50'))} / {_seconds(a.get('latency_ms_p95'))} "
                f"| {a.get('errors', '-')} | {cell['elapsed_s'] / 60:.0f} |")
        yes = next((c["answers"]["summary"]["always_yes_accuracy"] for c in by_combo.values()
                    if c["answers"]), None)
        if yes is not None:
            lines += ["", f"Answering \"yes\" to everything would score {yes:.3f} on this sample."]
        base = by_combo.get(BASELINE)
        diffs = []
        for combo in ROWS:
            cell = by_combo.get(combo)
            if base is None or cell is None or combo == BASELINE:
                continue
            parts = []
            acc = _paired(cell, base, "answers", "correct", lambda x: x["answerable"])
            if acc:
                parts.append(f"answer acc {acc[0]:+.3f} [{acc[1]:+.3f}, {acc[2]:+.3f}]")
            rec = _paired(cell, base, "retrieval", "recall@5", lambda x: "recall@5" in x)
            if rec:
                parts.append(f"recall@5 {rec[0]:+.3f} [{rec[1]:+.3f}, {rec[2]:+.3f}]")
            if parts:
                diffs.append(f"- {combo} - {BASELINE}: " + ", ".join(parts))
        if diffs:
            lines += ["", "Paired differences against the baseline, 95% bootstrap CI:", *diffs]
        lines.append("")
    return "\n".join(lines)


def write_report(out: Path) -> str:
    cells = load_cells(out)
    markdown = matrix_markdown(cells)
    (out / "summary.md").write_text(markdown, encoding="utf-8")
    summary = {f"{c['dataset']}/{c['combo']}": {
        "retrieval": (c["retrieval"] or {}).get("summary"),
        "answers": (c["answers"] or {}).get("summary"), "elapsed_s": c["elapsed_s"]} for c in cells}
    write_json(out / "summary.json", summary)
    return markdown


async def run(args) -> None:
    if args.rerank_url:
        CohereReranker.URL = args.rerank_url
    combos = [c if c == CLOSED_BOOK else combo_name(*parse_combo(c)) for c in args.combos.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.datasets.split(","):
        await run_dataset(name, combos, args)
    print(write_report(args.out))
    print(f"wrote {args.out / 'summary.md'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default=",".join(ANSWER_DATASETS))
    p.add_argument("--combos", default=",".join(ALL_COMBOS),
                   help=f"comma list from {','.join(ROWS)} (default: all but {CLOSED_BOOK})")
    p.add_argument("--limit", type=int, default=300, help="sample at most N questions per dataset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k", type=int, default=5, choices=[5, 10],
                   help="search depth; 5 is what the agent's tool uses")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--llm", choices=["anthropic", "openai", "fake"], help="override LLM_PROVIDER")
    p.add_argument("--rerank-url", help="override the Cohere endpoint, e.g. a local server")
    p.add_argument("--retrieval-only", action="store_true", help="skip /chat answers (no LLM)")
    p.add_argument("--schema-suffix", default="", help="match a `benchmarks run --schema-suffix` ingest")
    p.add_argument("--chunk-tokens", type=int, default=500, help="used only when a schema needs ingesting")
    p.add_argument("--chunk-overlap-tokens", type=int, default=60)
    p.add_argument("--out", type=Path, default=None,
                   help="cell directory (default: evals/results/matrix-<date>); reuse it to resume")
    p.add_argument("--report-only", action="store_true", help="rebuild summary.md from finished cells")
    args = p.parse_args()
    args.out = args.out or RESULTS_DIR / f"matrix-{datetime.now(UTC):%Y-%m-%d}"
    if args.report_only:
        print(write_report(args.out))
    else:
        run_async(run(args))


if __name__ == "__main__":
    main()
