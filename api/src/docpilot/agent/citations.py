"""Inline citations: the model writes [c:<chunk_id>] and we resolve those ids to chunks.

Ids are resolved against the database, so a hallucinated id simply produces no citation
(and shows up as uncited text in the UI) instead of pointing at the wrong source.
"""

import re

CITATION = re.compile(r"\[c:(\d+)\]")


def cited_chunk_ids(text: str) -> list[int]:
    """Unique ids in order of first appearance."""
    seen: dict[int, None] = {}
    for match in CITATION.finditer(text):
        seen.setdefault(int(match.group(1)), None)
    return list(seen)
