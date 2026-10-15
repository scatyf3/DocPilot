"""Sample answers from a results file into a Markdown sheet for manual review.

The LLM judge is itself a model; hand-checking ~20 of its verdicts tells you whether to
trust it. Fill in the checkboxes, then compare your calls with the judge's.

    python -m evals.spotcheck evals/results/<run>.json [-n 20] [--seed 0]
"""

import argparse
import json
import random
from pathlib import Path


def render(rows: list[dict], source: str) -> str:
    out = [f"# Spot check: {len(rows)} answers from `{source}`", "",
           "For each item tick what *you* believe, then compare with the judge.", ""]
    for r in rows:
        verdict = r.get("verdict") or {}
        out += [
            f"## {r['id']} ({r['type']})", "",
            f"**Question:** {r['question']}", "",
            f"**Reference:** {r['reference']}", "",
            "**Answer:**", "", *(f"> {line}" for line in (r["answer"] or "(empty)").splitlines()), "",
        ]
        for c in r["citations"]:
            snippet = " ".join(c["text"].split())[:400]
            out += [f"- `c:{c['id']}` *{c['title']}*: {snippet}"]
        out += [
            "",
            f"Judge: correct={verdict.get('correct')}, citations_faithful="
            f"{verdict.get('citations_faithful')}, refused={verdict.get('refused')}",
            f"> {verdict.get('explanation', '')}", "",
            "- [ ] I agree the answer is correct",
            "- [ ] I agree the citations support the claims",
            "- [ ] I agree with the judge overall", "", "---", "",
        ]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m evals.spotcheck")
    p.add_argument("results", type=Path)
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    result = json.loads(args.results.read_text(encoding="utf-8"))
    rows = (result.get("answers") or {}).get("items") or []
    if not rows:
        raise SystemExit("results file has no answers; run without --retrieval-only")
    sample = sorted(random.Random(args.seed).sample(rows, min(args.n, len(rows))), key=lambda r: r["id"])
    out = args.out or args.results.with_name(args.results.stem + "-spotcheck.md")
    out.write_text(render(sample, args.results.name), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
