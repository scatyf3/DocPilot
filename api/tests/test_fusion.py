import pytest

from docpilot.retrieval.fusion import rrf
from docpilot.retrieval.search import fts_query


def test_rrf_rewards_items_ranked_well_in_both_lists():
    fused = rrf([["a", "b", "c"], ["b", "d", "a"]], k=60)
    ids = [item for item, _ in fused]
    assert ids[:2] == ["b", "a"]  # in both lists beats a single first place
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_score_formula():
    [(item, score)] = rrf([["x"]], k=60)
    assert item == "x" and score == pytest.approx(1 / 61)


def test_rrf_empty():
    assert rrf([[], []]) == []


def test_fts_query_ors_words_and_drops_punctuation():
    assert fts_query("What's the SLO (p99)?") == "What | s | the | SLO | p99"
    assert fts_query("?!") == ""
