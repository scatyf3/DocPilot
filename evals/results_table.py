"""One table for the public benchmarks: retrieval and answer metrics per retrieval setting.

    python -m evals.results_table evals/results/2026-11-04/matrix-qwen            # -> table.tsv
    python -m evals.results_table evals/results/2026-11-04/matrix-qwen --format md

Reads the cells written by `python -m evals.bench_matrix` (<dataset>__<combo>.json). Columns:

  retrieval  POST /search at the agent's top_k (5)
    recall@1, recall@5  share of a question's gold units (documents or evidence quotes) in the
                        top 1 / 5. With two gold units, recall@1 can be at most 0.5.
    complete@5          1 if every gold unit is in the top 5, else 0
    MRR@5               1 / rank of the first chunk holding any gold unit (0 if none in the top 5)
    search ms           p50, p95
  answers    POST /chat, the agent the web app runs, with the same retrieval setting
    answer score        HotpotQA, MultiHop-RAG: the normalized reference appears in the answer
                        as whole tokens (yes/no: the first yes or no must match). Answerable
                        questions only.
    cited gold          of answers that were not refusals, the share citing a chunk that holds
                        a gold unit
    no search           answered without calling search_docs
    ungrounded          a non-refusal answer with no valid citation (an invented id counts as none)
    refusal acc         refused, on questions the corpus cannot answer
    false refusal       refused, on answerable questions
    LLM s               turn time minus tool time, p50, p95. Parallel tool calls are summed,
                        so it undercounts by under a second at p50.
    total s             the whole agent turn, p50, p95, under the run's concurrency

Baseline rows: answering "yes" to everything, and (where some questions are unanswerable)
refusing everything. Every cell answers the same seeded sample, so cells compare question by
question; summary.md next to the cells has paired bootstrap CIs for the differences.
"""

import argparse
import json
from pathlib import Path
from statistics import quantiles

DATASETS = {"hotpot": "HotpotQA (distractor)", "multihop": "MultiHop-RAG"}
MODES = ("vector", "fts", "hybrid")
COLUMNS = ["dataset", "questions", "unanswerable", "rerank", "mode",
           "recall@1", "recall@5", "complete@5", "MRR@5", "search ms p50", "search ms p95",
           "answer score", "cited gold", "no search", "ungrounded", "refusal acc", "false refusal",
           "LLM s p50", "LLM s p95", "total s p50", "total s p95"]


def _pct(values: list, q: int, scale: float = 1.0) -> float | None:
    values = [v for v in values if v is not None]
    return quantiles(values, n=100)[q - 1] / scale if len(values) > 1 else None


def cell_row(cell: dict, rerank: str) -> dict:
    r, a, items = cell["retrieval"]["summary"], cell["answers"]["summary"], cell["answers"]["items"]
    llm = [i.get("llm_ms") for i in items]
    total = [i["latency_ms"] for i in items]
    return {"rerank": rerank, "mode": cell["mode"],
            "recall@1": r["recall@1"], "recall@5": r["recall@5"], "complete@5": r["complete@5"],
            "MRR@5": r["mrr@10"],  # only top_k (5) results are retrieved, so this is MRR@5
            "search ms p50": r["latency_ms_p50"], "search ms p95": r["latency_ms_p95"],
            "answer score": a["accuracy"], "cited gold": a["cited_gold_rate"],
            "no search": a["no_search_rate"], "ungrounded": a["ungrounded_rate"],
            "refusal acc": a["refusal_accuracy"], "false refusal": a["false_refusal_rate"],
            "LLM s p50": _pct(llm, 50, 1000), "LLM s p95": _pct(llm, 95, 1000),
            "total s p50": _pct(total, 50, 1000), "total s p95": _pct(total, 95, 1000)}


def rows(matrix: Path) -> list[dict]:
    cells: dict[str, dict[str, dict]] = {}
    for path in sorted(matrix.glob("*__*.json")):
        cell = json.loads(path.read_text(encoding="utf-8"))
        if cell.get("answers") and cell.get("retrieval"):
            cells.setdefault(cell["dataset"], {})[cell["combo"]] = cell
    out = []
    for name in sorted(cells, key=lambda d: (d not in DATASETS, d)):
        by_combo = cells[name]
        a = next(iter(by_combo.values()))["answers"]["summary"]
        rerank_model = next(iter(by_combo.values())).get("rerank_model") or "reranker"
        common = {"dataset": DATASETS.get(name, name), "questions": a["n"], "unanswerable": a["n_null"]}
        for reranker, label in (("none", "none"), ("cohere", rerank_model)):
            for mode in MODES:
                combo = mode if reranker == "none" else f"{mode}+{reranker}"
                if combo in by_combo:
                    out.append(common | cell_row(by_combo[combo], label))
        out.append(common | {"rerank": "baseline", "mode": "always yes",
                             "answer score": a["always_yes_accuracy"]})
        if a["n_null"]:
            out.append(common | {"rerank": "baseline", "mode": "always refuse", "answer score": 0.0,
                                 "refusal acc": 1.0, "false refusal": 1.0})
    return out


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
    return str(value)


def to_tsv(table: list[dict]) -> str:
    lines = ["\t".join(COLUMNS)]
    lines += ["\t".join(_cell(r.get(c)) for c in COLUMNS) for r in table]
    return "\n".join(lines) + "\n"


def to_markdown(table: list[dict]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(_cell(r.get(c)) or "-" for c in COLUMNS) + " |" for r in table]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m evals.results_table", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("matrix", type=Path, help="a bench_matrix output directory")
    p.add_argument("--format", choices=["tsv", "md"], default="tsv")
    p.add_argument("--out", type=Path, help="default: <matrix>/table.<format>")
    args = p.parse_args()
    table = rows(args.matrix)
    text = to_tsv(table) if args.format == "tsv" else to_markdown(table)
    out = args.out or args.matrix / f"table.{args.format}"
    out.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
