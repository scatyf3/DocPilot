"""Retrieval metrics over quote-labeled relevance.

recall@k : fraction of a question's evidence quotes found in the top-k chunks. For a
           two-document question, finding one of the two sources scores 0.5.
MRR      : mean over questions of 1 / rank of the first chunk that contains any evidence
           quote (0 if none was retrieved).
"""

import re
from statistics import mean


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def contains(chunk_text: str, quote: str) -> bool:
    return normalize(quote) in normalize(chunk_text)


def recall_at_k(ranked_texts: list[str], quotes: list[str], k: int = 5) -> float:
    if not quotes:
        raise ValueError("recall is undefined for a question without evidence")
    top = ranked_texts[:k]
    return sum(any(contains(t, q) for t in top) for q in quotes) / len(quotes)


def reciprocal_rank(ranked_texts: list[str], quotes: list[str]) -> float:
    for rank, text in enumerate(ranked_texts, start=1):
        if any(contains(text, q) for q in quotes):
            return 1.0 / rank
    return 0.0


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 4) if values else None
