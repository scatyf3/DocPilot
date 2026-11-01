import pytest

from docpilot.retrieval.fusion import rrf
from docpilot.retrieval.search import bm25_idf, select_terms, tsquery_literal


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


def test_select_terms_prefers_rare_terms_within_budget():
    freqs = {"the": 900, "policy": 90, "keystone": 3, "absent": 0}
    assert select_terms(freqs, max_postings=1000) == ["keystone", "policy", "the"]
    assert select_terms(freqs, max_postings=100) == ["keystone", "policy"]
    assert select_terms({"common": 5000}, max_postings=100) == ["common"]  # never empty if any match
    assert select_terms({"absent": 0}) == []


def test_bm25_idf_favours_rare_terms_and_survives_stale_stats():
    assert bm25_idf(1000, 3) > bm25_idf(1000, 500) > 0
    assert bm25_idf(10, 50) > 0  # chunks uploaded since the stats were cached


def test_tsquery_literal_quotes_for_the_tsquery_parser():
    assert tsquery_literal("plain") == "'plain'"
    assert tsquery_literal("o'neil") == "'o''neil'"
    assert tsquery_literal("a\\b") == "'a\\\\b'"
