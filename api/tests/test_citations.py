from docpilot.agent.citations import cited_chunk_ids, split_citations
from docpilot.agent.prompts import REFUSAL, is_refusal


def test_cited_ids_unique_in_order():
    text = "A [c:12]. B [c:7] and again [c:12]. Not a cite: [c:x] [12]"
    assert cited_chunk_ids(text) == [12, 7]


def test_no_citations():
    assert cited_chunk_ids("plain answer") == []


def test_refusal_detection():
    assert is_refusal(REFUSAL)
    assert is_refusal("Sorry — I couldn’t find this in your documents.")
    assert not is_refusal("The SLO is 99.9% [c:3].")


def test_refusal_detection_handles_context_and_partial_answers():
    assert is_refusal("Let me check.\n\nI couldn't find this in your documents.")
    # A refusal may go on to say what was found instead, with citations.
    assert is_refusal("I couldn't find this in your documents. They only cover key rotation [c:28].")
    # The prompt asks for partial answers: a cited claim first means it did answer something.
    assert not is_refusal("PTO is 25 days [c:1]. For carry-over, I couldn't find this in your documents.")


def test_split_citations_keeps_only_ids_the_model_was_shown():
    answer = "A [c:1]. B [c:2]. C [c:3]. A again [c:1]."
    assert split_citations(answer, [1]) == ([1], [2, 3])
    assert split_citations(answer, [1, 2]) == ([1, 2], [3])
    assert split_citations(answer, []) == ([], [1, 2, 3])


def test_ids_too_long_for_a_bigint_are_not_citations():
    assert cited_chunk_ids("x [c:99999999999999999999] y [c:7]") == [7]
