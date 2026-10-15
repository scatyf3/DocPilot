from docpilot.ingest.chunk import approx_tokens, chunk_text, split_sections

DOC = """# Title

Intro paragraph.

## Setup

Install things.

```bash
# not a heading
echo hi
```

## Usage

Use things.
"""


def test_split_sections_on_headings_but_not_in_code_fences():
    sections = split_sections(DOC)
    assert [s.splitlines()[0] for s in sections] == ["# Title", "## Setup", "## Usage"]
    assert "# not a heading" in sections[1]


def test_small_sections_are_packed_into_one_chunk():
    chunks = chunk_text(DOC, max_tokens=500)
    assert len(chunks) == 1
    assert "Intro paragraph." in chunks[0] and "Use things." in chunks[0]


def test_sections_split_when_over_budget():
    chunks = chunk_text(DOC, max_tokens=12, overlap_tokens=2)
    assert len(chunks) >= 3
    assert chunks[0].startswith("# Title")


def test_long_section_windows_overlap_and_keep_heading():
    words = [f"w{i}" for i in range(1000)]
    text = "## Big\n\n" + " ".join(words)
    chunks = chunk_text(text, max_tokens=100, overlap_tokens=20)  # 75-word windows, 15 overlap
    assert len(chunks) > 1
    assert all(c.startswith("## Big") for c in chunks)
    # Any 16-word span (overlap + 1) is fully inside some chunk.
    for start in range(0, 1000 - 16, 37):
        span = " ".join(words[start : start + 16])
        assert any(span in c for c in chunks), span
    assert all(approx_tokens(c) <= 110 for c in chunks)


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n\n ") == []
