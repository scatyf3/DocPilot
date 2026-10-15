from docpilot.agent.citations import cited_chunk_ids
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
