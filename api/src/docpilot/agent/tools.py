"""The agent's three tools and their execution.

Each tool's input is a pydantic model: its JSON Schema is what the model sees, and the same
model validates whatever the model sends back, so schema and validation can't drift apart.
"""

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from docpilot.llm.base import ToolCall, ToolResult, ToolSpec
from docpilot.retrieval.search import RetrievedChunk

READ_DOC_MAX_CHUNKS = 40


class SearchDocsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, description="What to look for, in natural language.")
    top_k: int = Field(5, ge=1, le=10, description="How many chunks to return (default 5).")


class ReadDocArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: int = Field(description="The doc_id shown on a search result.")


class CreatePageArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=200, description="Page title.")
    markdown: str = Field(min_length=1, description="Full page body in Markdown.")


TOOLS: dict[str, tuple[type[BaseModel], str]] = {
    "search_docs": (SearchDocsArgs, (
        "Search the user's document library and return the most relevant chunks, each with "
        "a chunk id to cite as [c:<id>]. Call this before answering any question about the "
        "documents; call it again with other wording or for each part of a multi-part question."
    )),
    "read_doc": (ReadDocArgs, (
        "Return the full text of one document, chunk by chunk. Call this when search "
        "snippets are not enough, e.g. to summarise a whole document."
    )),
    "create_page": (CreatePageArgs, (
        "Save a new Markdown page to the user's library. Call this only when the user asks "
        "you to write, draft or save a page. The page becomes searchable."
    )),
}

TOOL_SPECS = [
    ToolSpec(name=name, description=desc, parameters=model.model_json_schema())
    for name, (model, desc) in TOOLS.items()
]


def format_chunk(chunk_id: int, document_id: int, title: str, text: str) -> str:
    title = title.replace('"', "'")
    return f'<chunk id="{chunk_id}" doc_id="{document_id}" title="{title}">\n{text}\n</chunk>'


@dataclass
class ToolOutcome:
    result: ToolResult            # sent back to the model
    output: Any                   # shown in the UI's tool part
    summary: str                  # one line for the trace
    chunk_ids: list[int] = field(default_factory=list)
    duration_ms: int = 0


SearchFn = Callable[[str, int], Awaitable[list[RetrievedChunk]]]
ReadFn = Callable[[int], Awaitable[dict | None]]
CreateFn = Callable[[str, str], Awaitable[int]]


class Toolbox:
    """Runs tool calls. Dependencies are plain async functions so tests can stub them."""

    specs = TOOL_SPECS

    def __init__(self, search: SearchFn, read_document: ReadFn, create_page: CreateFn):
        self._search = search
        self._read = read_document
        self._create = create_page

    async def run(self, call: ToolCall) -> ToolOutcome:
        started = time.perf_counter()
        try:
            outcome = await self._dispatch(call)
        except ValidationError as e:
            outcome = self._error(call, f"invalid arguments: {e.errors(include_url=False)}")
        except Exception as e:  # a failing tool should not kill the turn; tell the model
            outcome = self._error(call, f"{type(e).__name__}: {e}")
        outcome.duration_ms = round((time.perf_counter() - started) * 1000)
        return outcome

    async def _dispatch(self, call: ToolCall) -> ToolOutcome:
        if call.name not in TOOLS:
            return self._error(call, f"unknown tool {call.name!r}")
        if call.args is None:
            return self._error(call, f"arguments were not valid JSON: {call.raw_args[:200]}")
        args = TOOLS[call.name][0].model_validate(call.args)

        if isinstance(args, SearchDocsArgs):
            chunks = await self._search(args.query, args.top_k)
            body = "\n\n".join(format_chunk(c.id, c.document_id, c.title, c.text) for c in chunks)
            return ToolOutcome(
                result=ToolResult(call.id, body or "No matching chunks."),
                output=[{"id": c.id, "title": c.title, "score": round(c.score, 4),
                         "snippet": c.text[:200]} for c in chunks],
                summary=f"{len(chunks)} chunks for {args.query!r}",
                chunk_ids=[c.id for c in chunks],
            )

        if isinstance(args, ReadDocArgs):
            doc = await self._read(args.document_id)
            if doc is None:
                return self._error(call, f"no document with id {args.document_id}")
            chunks = doc["chunks"][:READ_DOC_MAX_CHUNKS]
            body = "\n\n".join(format_chunk(c["id"], doc["id"], doc["title"], c["text"])
                               for c in chunks)
            return ToolOutcome(
                result=ToolResult(call.id, body),
                output={"document_id": doc["id"], "title": doc["title"], "chunks": len(chunks)},
                summary=f"read {doc['title']!r} ({len(chunks)} chunks)",
                chunk_ids=[c["id"] for c in chunks],
            )

        assert isinstance(args, CreatePageArgs)
        doc_id = await self._create(args.title, args.markdown)
        return ToolOutcome(
            result=ToolResult(call.id, f"Saved page {args.title!r} as document {doc_id}."),
            output={"document_id": doc_id, "title": args.title},
            summary=f"created page {args.title!r} (doc {doc_id})",
        )

    @staticmethod
    def _error(call: ToolCall, message: str) -> ToolOutcome:
        return ToolOutcome(result=ToolResult(call.id, f"Error: {message}", is_error=True),
                           output={"error": message}, summary=f"error: {message}")
