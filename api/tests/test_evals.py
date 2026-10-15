"""Eval-suite logic: metrics, dataset validation, stream parsing, comparison."""

import json
from pathlib import Path

import pytest

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
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    out = parse_ui_stream(body)
    assert out["answer"] == "Let me check.\n\nIt is 25 days [c:4]."
    assert out["citations"][0]["id"] == 4 and out["trace"]["latency_ms"] == 12


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


def test_compare_marks_regressions():
    def result(recall, embedder="fake"):
        return {"config": {"embedder": embedder},
                "retrieval": {"hybrid": {"summary": {"recall@5": recall, "mrr": 0.5}}}}

    report = compare(result(0.70), result(0.80))
    assert "-0.100 **worse**" in report and "`retrieval/hybrid/mrr`" in report
    assert "differs" in compare(result(0.8, "openai"), result(0.8))
