"""Inline citations: the model writes [c:<chunk_id>] and we resolve those ids to chunks.

Only ids the model was actually shown count: chunks returned by a tool in this conversation,
in this turn or an earlier one (a follow-up can reuse them without searching). Any other id
produces no citation and shows up as uncited text in the UI, even if a chunk with that id
exists, because the model cannot have read it. Past answers aren't evidence of what it saw:
they're stored as written, including any invented ids.
"""

import re
from collections.abc import Iterable

CITATION = re.compile(r"\[c:(\d{1,18})\]")  # 18 digits fit a bigint; longer can't be a real id


def cited_chunk_ids(text: str) -> list[int]:
    """Unique ids in order of first appearance."""
    seen: dict[int, None] = {}
    for match in CITATION.finditer(text):
        seen.setdefault(int(match.group(1)), None)
    return list(seen)


def split_citations(answer: str, shown_ids: Iterable[int]) -> tuple[list[int], list[int]]:
    """Return (ids the model was shown, ids it was not), in order of first appearance."""
    shown = set(shown_ids)
    cited = cited_chunk_ids(answer)
    return [i for i in cited if i in shown], [i for i in cited if i not in shown]
