"""Reciprocal rank fusion.

RRF merges ranked lists using only ranks, never raw scores: BM25-ish ts_rank and cosine
similarity live on different scales, so adding them directly would be meaningless.
score(d) = sum over lists of 1 / (k + rank_of_d_in_list), with 1-based ranks.
k=60 is the constant from the original paper (Cormack et al., 2009); it damps the head.
"""

from collections.abc import Hashable, Sequence


def rrf(rankings: Sequence[Sequence[Hashable]], k: int = 60) -> list[tuple[Hashable, float]]:
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    # Stable on ties: Python's sort keeps first-seen order for equal scores.
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
