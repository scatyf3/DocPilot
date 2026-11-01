"""Markdown diff of a results file against the committed baseline (used for the PR comment).

    python -m evals.compare evals/results/<run>.json [--baseline evals/baseline.json] [--out diff.md]
"""

import argparse
import json
from pathlib import Path

from evals.dataset import EVALS_DIR

# Metrics where a lower number is better; everything else is higher-is-better.
LOWER_IS_BETTER = {"false_refusal_rate", "latency_ms_p50", "latency_ms_p95", "cost_usd_total", "errors",
                   "judge_errors", "unjudged", "no_search_rate", "incomplete_rate", "rejected_citations",
                   "ungrounded_rate"}


def flatten(result: dict) -> dict[str, float | None]:
    flat: dict[str, float | None] = {}
    for variant, r in (result.get("retrieval") or {}).items():
        for metric in ("recall@5", "mrr", "recall@5_single", "recall@5_multi"):
            flat[f"retrieval/{variant}/{metric}"] = r["summary"].get(metric)
    answers = result.get("answers") or {}
    for metric, value in (answers.get("summary") or {}).items():
        if metric != "n":
            flat[f"answers/{metric}"] = value
    return flat


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def _delta(name: str, base, cur) -> str:
    if base is None or cur is None:
        return ""
    diff = cur - base
    if abs(diff) < 1e-9:
        return "0"
    better = (diff < 0) if name.split("/")[-1] in LOWER_IS_BETTER else (diff > 0)
    return f"{diff:+.3f} {'better' if better else '**worse**'}"


def compare(current: dict, baseline: dict) -> str:
    cur, base = flatten(current), flatten(baseline)
    lines = ["### DocPilot eval vs baseline", ""]
    for key in ("embedder", "llm", "judge"):
        c, b = current["config"].get(key), baseline.get("config", {}).get(key)
        if c != b:
            lines.append(f"> Note: `{key}` differs (baseline `{b}`, this run `{c}`): "
                         "deltas are not like-for-like.")
    lines += ["", "| metric | baseline | this PR | delta |", "|---|---|---|---|"]
    for name in sorted(set(cur) | set(base)):
        b, c = base.get(name), cur.get(name)
        if b is None and c is None:
            continue
        lines.append(f"| `{name}` | {_fmt(b)} | {_fmt(c)} | {_delta(name, b, c)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m evals.compare")
    p.add_argument("current", type=Path)
    p.add_argument("--baseline", type=Path, default=EVALS_DIR / "baseline.json")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    report = compare(json.loads(args.current.read_text(encoding="utf-8")),
                     json.loads(args.baseline.read_text(encoding="utf-8")))
    if args.out:
        args.out.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
