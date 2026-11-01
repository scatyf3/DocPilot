"""Eval-suite logic: metrics, dataset validation, stream parsing, comparison."""

import json
from pathlib import Path

import pytest

from evals.benchmarks import auc, score_ranks, unit_ranks
from evals.compare import compare
from evals.dataset import Evidence, Item, load_dataset, validate
from evals.metrics import contains, recall_at_k, reciprocal_rank
from evals.run import parse_ui_stream, summarize_answers


def test_contains_normalises_whitespace_and_case():
    assert contains("The SLO is\n  99.9%   uptime.", "slo is 99.9% uptime")
    assert not contains("The SLO is 99.5%", "99.9%")


def test_recall_at_k_counts_each_quote():
    ranked = ["alpha beta", "gamma", "delta", "x", "y", "epsilon"]
    assert recall_at_k(ranked, ["beta"], k=5) == 1.0
    assert recall_at_k(ranked, ["beta", "epsilon"], k=5) == 0.5  # epsilon is rank 6
    assert recall_at_k(ranked, ["beta", "epsilon"], k=6) == 1.0
    with pytest.raises(ValueError):
        recall_at_k(ranked, [], k=5)


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "target here"], ["target"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b"], ["target"]) == 0.0
    assert reciprocal_rank(["q2 text", "q1 text"], ["q1", "q2"]) == 1.0  # first of any quote


def test_shipped_dataset_is_valid_and_balanced():
    items = load_dataset()
    assert validate(items) == []
    assert len(items) >= 100
    types = [i.type for i in items]
    assert types.count("multi") >= len(items) * 0.3
    assert types.count("unanswerable") >= 10


def test_validator_catches_bad_quotes(tmp_path: Path):
    (tmp_path / "a.md").write_text("Pager rotation is weekly.\n", encoding="utf-8")
    items = [
        Item("q1", "?", "weekly", [Evidence("a.md", "rotation   is weekly")], "single"),  # ok
        Item("q2", "?", "x", [Evidence("a.md", "is monthly")], "single"),
        Item("q3", "?", "x", [Evidence("a.md", "weekly"), Evidence("a.md", "Pager")], "multi"),
        Item("q4", "?", "x", [Evidence("a.md", "weekly")], "unanswerable"),
        Item("q4", "?", "x", [Evidence("missing.md", "x")], "single"),
    ]
    errors = validate(items, tmp_path)
    assert any("q2" in e and "quote not found" in e for e in errors)
    assert any("q3" in e and "two documents" in e for e in errors)
    assert any("q4" in e and "has evidence" in e for e in errors)
    assert any("unknown doc" in e for e in errors)
    assert any("duplicate id q4" in e for e in errors)
    assert not any(e.startswith("q1") for e in errors)


def test_parse_ui_stream_rebuilds_answer():
    chunks = [
        {"type": "start", "messageId": "m"},
        {"type": "text-start", "id": "text-0"},
        {"type": "text-delta", "id": "text-0", "delta": "Let me check."},
        {"type": "text-delta", "id": "text-1", "delta": "It is 25 days "},
        {"type": "text-delta", "id": "text-1", "delta": "[c:4]."},
        {"type": "data-citations", "data": [{"id": 4, "title": "PTO", "text": "25 days"}]},
        {"type": "data-trace", "data": {"latency_ms": 12}},
        {"type": "finish", "finishReason": "length"},
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    out = parse_ui_stream(body)
    assert out["answer"] == "Let me check.\n\nIt is 25 days [c:4]."
    assert out["citations"][0]["id"] == 4 and out["trace"]["latency_ms"] == 12
    assert out["finish_reason"] == "length"


def test_summarize_answers():
    def row(type_, refused, cited, correct):
        return {"type": type_, "refused": refused, "cited_evidence": cited, "error": None,
                "verdict": {"correct": correct, "citations_faithful": cited, "refused": refused},
                "latency_ms": 100, "cost_usd": 0.01}

    s = summarize_answers([row("single", False, True, True), row("multi", False, False, False),
                           row("single", True, False, False), row("unanswerable", True, False, True)])
    assert s["correctness"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["citation_faithfulness"] == 0.5
    assert s["refusal_accuracy"] == 1.0
    assert s["false_refusal_rate"] == pytest.approx(0.3333, abs=1e-4)
    assert s["cost_usd_total"] == 0.04
    assert s["judge_errors"] == 0 and s["unjudged"] == 0


def test_summarize_answers_counts_failures_instead_of_dropping_them():
    def row(verdict, error=None):
        return {"type": "single", "refused": False, "cited_evidence": False, "error": error,
                "verdict": verdict, "latency_ms": 1, "cost_usd": 0}

    good = [row({"correct": True, "citations_faithful": True}) for _ in range(5)]
    s = summarize_answers(good + [row(None, error="timeout") for _ in range(5)])
    assert s["correctness"] == 0.5 and s["errors"] == 5 and s["unjudged"] == 5
    s = summarize_answers(good + [row({"error": "boom"}) for _ in range(5)])
    assert s["correctness"] == 0.5 and s["judge_errors"] == 5 and s["unjudged"] == 5
    s = summarize_answers([row(None) for _ in range(3)])  # judge off: unknown, not zero
    assert s["correctness"] is None and s["unjudged"] is None


def test_a_run_where_every_turn_crashes_cannot_look_clean():
    rows = [{"type": "single", "refused": False, "cited_evidence": False, "error": "boom", "verdict": None,
             "latency_ms": None, "cost_usd": 0} for _ in range(4)]
    s = summarize_answers(rows, judge_enabled=True)
    assert s["correctness"] == 0.0 and s["unjudged"] == 4 and s["errors"] == 4 and s["incomplete_rate"] == 1.0


def test_summarize_answers_reports_agent_behaviour():
    def row(type_, seen, tools, finish, rejected):
        return {"type": type_, "refused": False, "cited_evidence": True, "error": None, "verdict": None,
                "latency_ms": 1, "cost_usd": 0, "evidence_seen_recall": seen, "finish_reason": finish,
                "tool_calls": [{"name": t} for t in tools], "rejected_citation_ids": rejected}

    s = summarize_answers([row("single", 1.0, ["search_docs"], "stop", []),
                           row("multi", 0.5, ["search_docs", "read_doc"], "other", [7]),
                           row("single", 0.0, [], "stop", [8, 9]),
                           row("unanswerable", None, ["search_docs"], "stop", [])])
    assert s["evidence_seen_recall"] == 0.5
    assert s["no_search_rate"] == 0.25 and s["incomplete_rate"] == 0.25
    assert s["rejected_citations"] == 3


def test_compare_marks_regressions():
    def result(recall, embedder="fake"):
        return {"config": {"embedder": embedder},
                "retrieval": {"hybrid": {"summary": {"recall@5": recall, "mrr": 0.5}}}}

    report = compare(result(0.70), result(0.80))
    assert "-0.100 **worse**" in report and "`retrieval/hybrid/mrr`" in report
    assert "differs" in compare(result(0.8, "openai"), result(0.8))


def test_unit_ranks_by_document_and_by_quote():
    ranked = [{"document_id": 7, "text": "alpha"}, {"document_id": 9, "text": "The  Beta fact"}]
    assert unit_ranks(ranked, ["a", "b", "c"], "doc", {"a": 9, "b": 7}) == [2, 1, None]
    assert unit_ranks(ranked, ["beta fact", "gamma"], "quote", {}) == [2, None]


def test_score_ranks():
    s = score_ranks([2, 6])
    assert (s["recall@1"], s["recall@5"], s["recall@10"]) == (0.0, 0.5, 1.0)
    assert s["complete@5"] == 0.0 and s["mrr@10"] == 0.5
    perfect = score_ranks([1, 2])
    assert perfect["complete@5"] == 1.0 and perfect["ndcg@10"] == pytest.approx(1.0)
    assert score_ranks([None])["mrr@10"] == 0.0 and score_ranks([None])["ndcg@10"] == 0.0
    assert score_ranks([11])["mrr@10"] == 0.0  # beyond the cutoff


def test_auc():
    assert auc([2.0, 3.0], [1.0]) == 1.0
    assert auc([1.0], [1.0]) == 0.5
    assert auc([], [1.0]) is None
