"""Eval-suite logic: metrics, dataset validation, stream parsing, comparison."""

import json
from pathlib import Path

import pytest

from evals.bench_matrix import (
    ALL_COMBOS,
    answer_matches,
    cited_gold,
    matrix_markdown,
    normalize_answer,
    parse_combo,
    summarize,
)
from evals.benchmarks import (
    BenchItem,
    auc,
    best_scores,
    qasper_items,
    qasper_markdown,
    score_ranks,
    unit_ranks,
)
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


def test_normalize_answer_drops_citations_punctuation_and_articles():
    assert normalize_answer("The answer is Sam Bankman-Fried [c:12].") == "answer is sam bankman fried"


def test_answer_matches():
    assert answer_matches("It was Sam Bankman-Fried [c:3].", "Sam Bankman-Fried")
    assert not answer_matches("Samuel Bankman", "Sam Bankman-Fried")
    assert not answer_matches("Googleplex", "Google")  # whole tokens only
    assert answer_matches("Yes. Both reports say so [c:1], no doubt.", "Yes")
    assert not answer_matches("No, the second article disagrees. Yes, the first agrees.", "Yes")
    assert not answer_matches("anything", "")


def test_answer_matches_strips_unicode_punctuation():
    assert normalize_answer("Yes—broadly consistent") == "yes broadly consistent"
    assert answer_matches("**Yes—broadly consistent.**", "Yes")
    assert answer_matches("“Don’t Stop” was the single.", "Don’t Stop")
    assert answer_matches("He played for Bayern Munich…", "Bayern Munich")


def test_cited_gold_by_document_and_by_quote():
    doc_item = BenchItem("h1", "?", "bridge", ["Paris", "Lyon"], "x")
    assert cited_gold([{"title": "Lyon", "text": ""}], doc_item, "doc")
    assert not cited_gold([{"title": "Nice", "text": "Paris"}], doc_item, "doc")
    quote_item = BenchItem("m1", "?", "inference_query", ["the fine was  $4 billion"], "x")
    assert cited_gold([{"title": "u", "text": "Reports say The fine was $4 billion."}], quote_item, "quote")


def test_bench_answer_summary():
    def row(type_, answerable, refused, correct, reference="x", cited=True, tools=("search_docs",)):
        return {"type": type_, "answerable": answerable, "refused": refused, "correct": correct,
                "reference": reference, "cited_gold": cited if answerable else None, "error": None,
                "finish_reason": "stop", "tool_calls": list(tools), "ungrounded": False,
                "latency_ms": 10, "cost_usd": 0.0}

    s = summarize([row("comparison_query", True, False, True, "Yes"),
                   row("inference_query", True, False, False, "Google", cited=False),
                   row("inference_query", True, True, False, "Google", cited=False, tools=()),
                   row("null_query", False, True, True), row("null_query", False, False, False)])
    assert s["accuracy"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["accuracy_by_type"] == {"comparison_query": 1.0, "inference_query": 0.0}
    assert s["always_yes_accuracy"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["refusal_accuracy"] == 0.5 and s["false_refusal_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["cited_gold_rate"] == 0.5  # refused answers are not "attempted"
    assert s["no_search_rate"] == 0.2 and s["n_null"] == 2


def test_matrix_combos():
    assert ALL_COMBOS == ["vector", "fts", "hybrid", "vector+cohere", "fts+cohere", "hybrid+cohere"]
    assert parse_combo("hybrid") == ("hybrid", "none")
    assert parse_combo("fts+cohere") == ("fts", "cohere")
    with pytest.raises(ValueError):
        parse_combo("bm25")


def test_matrix_markdown_tables_pending_cells_and_paired_diffs():
    def cell(combo, correct, recall):
        items = [{"id": f"q{i}", "answerable": True, "correct": c} for i, c in enumerate(correct)]
        retrieval = [{"id": f"q{i}", "recall@5": r} for i, r in enumerate(recall)]
        return {"dataset": "hotpot", "combo": combo,
                "n_questions": 4, "n_docs": 9, "top_k": 5, "seed": 0, "llm": "qwen3",
                "embedder": "bge-m3", "rerank_model": "bge-reranker", "elapsed_s": 120,
                "retrieval": {"summary": {"recall@1": 0.25, "recall@5": sum(recall) / 4,
                                          "complete@5": 0.5, "mrr@10": 0.4,
                                          "latency_ms_p50": 30, "latency_ms_p95": 60},
                              "items": retrieval},
                "answers": {"summary": {"accuracy": sum(correct) / 4, "cited_gold_rate": 0.5,
                                        "refusal_accuracy": None, "false_refusal_rate": 0.0,
                                        "latency_ms_p50": 4200, "latency_ms_p95": 9000,
                                        "errors": 0, "always_yes_accuracy": 0.25},
                            "items": items}}

    md = matrix_markdown([cell("hybrid", [1, 1, 0, 0], [1, 1, 0, 0]),
                          cell("vector", [1, 1, 1, 1], [1, 1, 1, 0])])
    assert "**hotpot**: 4 questions, 9 docs" in md
    assert "| vector | 0.250 | 0.750 | 0.500 | 0.400 | 30 / 60 | 1.000 |" in md
    assert "| 4.2 / 9.0 | 0 | 2 |" in md
    assert "| fts+cohere | (pending) |" in md
    assert md.index("| vector |") < md.index("| hybrid |")  # rows keep ALL_COMBOS order
    assert "- vector - hybrid: answer acc +0.500" in md and "recall@5 +0.250" in md
    assert "scored 0.250" not in md and "would score 0.250" in md
    assert "closed-book" not in md  # optional row stays hidden until it has run

    closed = cell("closed-book", [1, 0, 0, 0], [0, 0, 0, 0]) | {"retrieval": None, "n_docs": 0}
    md = matrix_markdown([cell("hybrid", [1, 1, 0, 0], [1, 1, 0, 0]), closed])
    assert "**hotpot**: 4 questions, 9 docs" in md  # doc count from the open-book cell
    assert "| closed-book | - | - | - | - | - | 0.250 |" in md
    assert "- closed-book - hybrid: answer acc -0.250 [" in md
    assert matrix_markdown([]) == "No finished cells yet.\n"


QASPER_PAPER = {
    "title": "A Paper", "abstract": "We study things.",
    "full_text": [{"section_name": "Introduction", "paragraphs": ["Intro text.", " "]},
                  {"section_name": "Model ::: Encoder", "paragraphs": ["The encoder is a BiLSTM."]},
                  {"section_name": None, "paragraphs": ["Orphan paragraph."]}],
    "figures_and_tables": [{"file": "1-Table1-1.png", "caption": "Table 1: Results on SQuAD."}],
    "qas": [
        {"question_id": "q1", "question": "What encoder?", "answers": [
            {"answer": {"unanswerable": False, "extractive_spans": ["BiLSTM"], "yes_no": None,
                        "free_form_answer": "", "evidence": ["The encoder is a BiLSTM."]}},
            {"answer": {"unanswerable": False, "extractive_spans": [], "yes_no": None,
                        "free_form_answer": "a BiLSTM",
                        "evidence": ["FLOAT SELECTED: Table 1: Results on SQuAD.", "Model ::: Encoder"]}}]},
        {"question_id": "q2", "question": "Any code?", "answers": [
            {"answer": {"unanswerable": True, "extractive_spans": [], "yes_no": None,
                        "free_form_answer": "", "evidence": []}}]},
        {"question_id": "q3", "question": "Is it good?", "answers": [
            {"answer": {"unanswerable": False, "extractive_spans": [], "yes_no": True,
                        "free_form_answer": "", "evidence": ["Model ::: Encoder"]}}]},
    ],
}


def test_qasper_markdown_keeps_section_nesting_and_captions():
    md = qasper_markdown(QASPER_PAPER)
    assert md.startswith("# A Paper\n\n## Abstract\n\nWe study things.\n\n## Introduction\n\nIntro text.")
    assert "### Encoder\n\nThe encoder is a BiLSTM." in md
    assert "Orphan paragraph." in md
    assert md.endswith("## Figures and tables\n\nTable 1: Results on SQuAD.")


def test_qasper_items_gold_sets():
    q1, q2 = qasper_items("p1", QASPER_PAPER)  # q3's only evidence is a section name: dropped
    assert (q1.type, q1.scope) == ("mixed", "p1")
    assert q1.unit_sets == [["The encoder is a BiLSTM."], ["Table 1: Results on SQuAD."]]
    assert q1.units == ["Table 1: Results on SQuAD.", "The encoder is a BiLSTM."]
    assert (q2.type, q2.units, q2.unit_sets) == ("unanswerable", [], None)


def test_best_scores_takes_the_best_annotator_per_metric():
    item = BenchItem("q", "?", "x", ["a", "b", "c"], unit_sets=[["a", "b"], ["c"]])
    ranked = [{"text": "c"}, {"text": "zz"}, {"text": "a"}]
    s = best_scores(ranked, item, "quote", {})
    assert s["recall@1"] == 1.0  # annotator 2's only paragraph is first
    assert s["recall@5"] == 1.0 and s["mrr@10"] == 1.0
    assert best_scores(ranked, BenchItem("q", "?", "x", ["a", "b"]), "quote", {})["recall@5"] == 0.5


def test_qasper_library_variant_names_the_paper_and_drops_the_scope(monkeypatch, tmp_path):
    import evals.benchmarks as bm

    (tmp_path / "qasper").mkdir()
    (tmp_path / "qasper/qasper-dev-v0.3.json").write_text(json.dumps({"p1": QASPER_PAPER}))
    monkeypatch.setattr(bm, "DATA_DIR", tmp_path)
    scoped, library = bm.load_qasper(), bm.load_qasper(library=True)
    assert scoped.items[0].scope == "p1" and scoped.items[0].question == "What encoder?"
    assert library.items[0].scope is None
    assert library.items[0].question == 'In the paper "A Paper": What encoder?'
    assert library.items[0].unit_sets == scoped.items[0].unit_sets


def test_qasper_prediction_from_an_agent_turn():
    from docpilot.agent.prompts import REFUSAL
    from evals.qasper_official import chunk_units, paper_units, predict

    units = paper_units(QASPER_PAPER)
    first = ("# A Paper\n\n## Abstract\n\nWe study things.\n\n## Introduction\n\nIntro text.\n\n"
             "### Encoder\n\nThe encoder")
    assert chunk_units(first, units) == ["Intro text.", "The encoder is a BiLSTM."]  # a fragment counts
    figures = "## Figures and tables\n\nTable 1: Results on SQuAD."
    assert chunk_units(figures, units) == ["FLOAT SELECTED: Table 1: Results on SQuAD."]

    turn = {"question_id": "q1", "error": None,
            "answer": "A BiLSTM encoder [c:1]. It wins on SQuAD [c:2]. Also [c:7].",
            "citations": [{"id": 1, "text": first}, {"id": 2, "text": figures}]}
    chunk, cited = predict(turn, units)["chunk"], predict(turn, units)["cited"]
    assert chunk["predicted_answer"] == "A BiLSTM encoder . It wins on SQuAD . Also ."
    assert chunk["predicted_evidence"] == ["Intro text.", "The encoder is a BiLSTM.",
                                           "FLOAT SELECTED: Table 1: Results on SQuAD."]
    # Per citation, only the paragraph closest to its claim; the unresolved [c:7] adds nothing.
    assert cited["predicted_evidence"] == ["The encoder is a BiLSTM.",
                                           "FLOAT SELECTED: Table 1: Results on SQuAD."]

    refused = predict(turn | {"answer": REFUSAL, "citations": []}, units)["chunk"]
    assert (refused["predicted_answer"], refused["predicted_evidence"]) == ("Unanswerable", [])
    failed = predict(turn | {"error": "boom", "answer": ""}, units)["cited"]
    assert (failed["predicted_answer"], failed["predicted_evidence"]) == ("", [])
