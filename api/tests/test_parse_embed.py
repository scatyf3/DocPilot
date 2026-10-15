import numpy as np
import pytest

from docpilot.embeddings import FakeEmbedder
from docpilot.ingest.parse import parse_document


def test_html_keeps_headings_as_markdown_and_drops_scripts():
    html = b"""<html><head><title>Guide</title><script>evil()</script></head>
    <body><h1>Guide</h1><p>Hello <b>world</b>.</p><ul><li><p>one</p></li></ul></body></html>"""
    doc = parse_document("guide.html", html)
    assert doc.title == "Guide"
    assert doc.text == "# Guide\n\nHello world .\n\n- one"


def test_markdown_title_from_first_heading():
    assert parse_document("notes.md", b"intro\n\n# Real Title\n\nbody").title == "Real Title"
    assert parse_document("notes.md", b"no heading").title == "notes"


def test_unsupported_type():
    with pytest.raises(ValueError, match="unsupported"):
        parse_document("x.docx", b"")


async def test_fake_embedder_is_deterministic_normalised_and_lexical():
    emb = FakeEmbedder(1024)
    a, b, c = await emb.embed(["on-call pager rotation", "pager rotation for on-call", "PTO policy"])
    assert a.shape == (1024,) and np.isclose(np.linalg.norm(a), 1.0)
    assert np.allclose(a, (await emb.embed(["on-call pager rotation"]))[0])
    assert float(a @ b) > float(a @ c)
