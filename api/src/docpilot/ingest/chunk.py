"""Heading-aware chunker.

1. Split the Markdown at headings, so a chunk rarely mixes unrelated sections.
2. Sections longer than `max_tokens` are cut into overlapping word windows. With an overlap
   of O words, any span of <= O+1 words lands whole in at least one window, so short facts
   are never split across a chunk boundary.
3. Consecutive small sections are packed together up to `max_tokens`, so we don't index
   dozens of tiny heading-only chunks.

Token counts are approximated from words (~0.75 words per token); that is plenty accurate
for sizing chunks and avoids shipping a tokenizer.
"""

import math
import re

WORDS_PER_TOKEN = 0.75
_HEADING = re.compile(r"^#{1,6}\s+\S")
_FENCE = re.compile(r"^\s*(```|~~~)")


def approx_tokens(text: str) -> int:
    return math.ceil(len(text.split()) / WORDS_PER_TOKEN)


def split_sections(text: str) -> list[str]:
    """Split before every Markdown heading line, ignoring '#' lines inside code fences."""
    sections: list[list[str]] = [[]]
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        if not in_fence and _HEADING.match(line) and any(s.strip() for s in sections[-1]):
            sections.append([])
        sections[-1].append(line)
    return [s for s in ("\n".join(lines).strip() for lines in sections) if s]


def _windows(section: str, window_words: int, overlap_words: int) -> list[str]:
    """Overlapping word windows that slice the original text (formatting is preserved)."""
    spans = [m.span() for m in re.finditer(r"\S+", section)]
    first_line = section.splitlines()[0]
    heading = first_line if _HEADING.match(first_line) else None
    step = max(1, window_words - overlap_words)

    out: list[str] = []
    for start in range(0, len(spans), step):
        end = min(start + window_words, len(spans))
        piece = section[spans[start][0] : spans[end - 1][1]]
        # Later windows repeat the section heading so each chunk keeps its context.
        if heading and start > 0:
            piece = f"{heading}\n\n{piece}"
        out.append(piece)
        if end == len(spans):
            break
    return out


def chunk_text(text: str, max_tokens: int = 500, overlap_tokens: int = 60) -> list[str]:
    window_words = max(1, int(max_tokens * WORDS_PER_TOKEN))
    overlap_words = min(int(overlap_tokens * WORDS_PER_TOKEN), window_words - 1)

    pieces: list[str] = []
    for section in split_sections(text):
        if approx_tokens(section) <= max_tokens:
            pieces.append(section)
        else:
            pieces.extend(_windows(section, window_words, overlap_words))

    chunks: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if current and approx_tokens("\n\n".join([*current, piece])) > max_tokens:
            chunks.append("\n\n".join(current))
            current = []
        current.append(piece)
    if current:
        chunks.append("\n\n".join(current))
    return chunks
