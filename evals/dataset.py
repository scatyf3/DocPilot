"""Load and validate the labeled QA set.

Relevance is labeled by evidence *quote*, not chunk id: a retrieved chunk counts as relevant
if it contains the quote (whitespace-normalised). Labels therefore survive re-chunking.

    python -m evals.dataset        # validate; exits non-zero on any problem
"""

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from evals.metrics import normalize

EVALS_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVALS_DIR / "corpus"
DATASET_PATH = EVALS_DIR / "dataset.jsonl"
TYPES = {"single", "multi", "unanswerable"}


@dataclass
class Evidence:
    doc: str
    quote: str


@dataclass
class Item:
    id: str
    question: str
    answer: str
    evidence: list[Evidence]
    type: str

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"


def load_dataset(path: Path = DATASET_PATH) -> list[Item]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            raw["evidence"] = [Evidence(**e) for e in raw["evidence"]]
            items.append(Item(**raw))
    return items


def validate(items: list[Item], corpus_dir: Path = CORPUS_DIR) -> list[str]:
    """Return a list of problems (empty means valid)."""
    docs = {p.name: normalize(p.read_text(encoding="utf-8")) for p in corpus_dir.glob("*.md")}
    errors = []
    ids = Counter(i.id for i in items)
    errors += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    for item in items:
        where = f"{item.id}:"
        if item.type not in TYPES:
            errors.append(f"{where} bad type {item.type!r}")
        if not item.question.strip() or not item.answer.strip():
            errors.append(f"{where} empty question or answer")
        if item.type == "unanswerable" and item.evidence:
            errors.append(f"{where} unanswerable item has evidence")
        if item.type != "unanswerable" and not item.evidence:
            errors.append(f"{where} answerable item has no evidence")
        if item.type == "multi" and len({e.doc for e in item.evidence}) < 2:
            errors.append(f"{where} multi item must cite at least two documents")
        for ev in item.evidence:
            if ev.doc not in docs:
                errors.append(f"{where} unknown doc {ev.doc!r}")
            elif normalize(ev.quote) not in docs[ev.doc]:
                errors.append(f"{where} quote not found in {ev.doc}: {ev.quote!r}")
    return errors


def main() -> None:
    items = load_dataset()
    errors = validate(items)
    counts = Counter(i.type for i in items)
    print(f"{len(items)} items: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for e in errors:
        print("ERROR", e)
    if errors:
        sys.exit(1)
    print("dataset OK: every evidence quote appears verbatim in its document")


if __name__ == "__main__":
    main()
