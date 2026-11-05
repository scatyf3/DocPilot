"""POST /chat (streaming agent answer) plus conversation and trace lookups.

/chat speaks the AI SDK UI message stream, so the web app's `useChat` consumes it directly,
and the eval suite replays labeled questions through this same endpoint.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from docpilot import store
from docpilot import ui_stream as ui
from docpilot.agent.citations import split_citations
from docpilot.agent.loop import Done, run_agent
from docpilot.agent.prompts import is_ungrounded
from docpilot.agent.tools import Toolbox
from docpilot.chat_stream import ChatStreamEncoder
from docpilot.config import RerankerName, RetrievalMode
from docpilot.ingest.pipeline import ingest_text
from docpilot.observability import request_id
from docpilot.security import CurrentPrincipal, rate_limit

log = logging.getLogger(__name__)
router = APIRouter()


class UIPart(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    text: str | None = None


class UIMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    parts: list[UIPart] = []


class RetrievalOptions(BaseModel):
    mode: RetrievalMode | None = None
    reranker: RerankerName | None = None
    # Chat with these documents only: search_docs and read_doc see nothing else.
    document_ids: list[int] | None = Field(None, min_length=1, max_length=100)


class ChatRequest(BaseModel):
    """What DefaultChatTransport POSTs: {id, messages, trigger, messageId, ...body}."""

    model_config = ConfigDict(extra="allow")
    id: str = Field(min_length=1, max_length=200)
    messages: list[UIMessage]
    retrieval: RetrievalOptions = RetrievalOptions()

    def question(self) -> str:
        for m in reversed(self.messages):
            if m.role == "user":
                return "".join(p.text or "" for p in m.parts if p.type == "text").strip()
        return ""


@router.post("/chat", dependencies=[Depends(rate_limit("chat"))])
async def chat(request: Request, body: ChatRequest,
               principal: CurrentPrincipal) -> StreamingResponse:
    question = body.question()
    if not question:
        raise HTTPException(422, "no user message to answer")
    s = request.app.state
    if len(question) > s.settings.max_question_chars:
        raise HTTPException(413, f"question too long (max {s.settings.max_question_chars} characters)")
    tenant = principal.tenant_id
    # Before anything is read: a colleague's conversation id must not let us continue it.
    if not await store.claim_conversation(s.pool, tenant, principal.user_id, body.id):
        raise HTTPException(404, "conversation not found")
    # The database, not the client, is the source of truth for earlier turns.
    history = [{"role": m["role"], "content": m["content"]}
               for m in await store.conversation_history(s.pool, tenant, body.id)]
    shown_earlier = await store.retrieved_chunk_ids(s.pool, tenant, body.id)
    history = recent_history(history, s.settings.max_history_messages)

    scope = body.retrieval.document_ids

    # The tools close over the caller's tenant. The model chooses queries and ids, never the
    # tenant, so a prompt-injected document can't point it at another tenant's library.
    async def search(query: str, top_k: int):
        return await s.retriever.search(
            query, tenant_id=tenant, top_k=top_k, mode=body.retrieval.mode,
            reranker=body.retrieval.reranker, document_ids=scope,
        )

    async def read_document(doc_id: int):
        if scope is not None and doc_id not in scope:
            return None  # outside the chat's documents: same answer as a missing one
        return await store.get_document(s.pool, tenant, doc_id)

    async def create_page(title: str, markdown: str) -> int:
        doc_id, _ = await ingest_text(s.pool, s.embedder, s.settings, tenant_id=tenant, title=title,
                                      text=page_text(title, markdown), source="agent", kind="generated",
                                      created_by=principal.user_id, cache=s.cache)
        return doc_id

    toolbox = Toolbox(search, read_document, create_page)
    events = run_agent(s.llm, toolbox, history, question, max_steps=s.settings.max_agent_steps)
    return StreamingResponse(
        _encode(events, request, tenant, body.id, question, shown_earlier),
        media_type="text/event-stream",
        headers=ui.HEADERS,
    )


def page_text(title: str, markdown: str) -> str:
    """Only chunk text is indexed, so a page whose body omits its title couldn't be found by it."""
    return markdown if markdown.lstrip().startswith("#") else f"# {title}\n\n{markdown}"


def recent_history(history: list[dict], limit: int) -> list[dict]:
    """Keep the prompt bounded on long conversations, always starting on a user turn."""
    recent = history[-limit:] if limit else []
    while recent and recent[0]["role"] != "user":
        recent = recent[1:]
    return recent


async def _encode(
    events, request: Request, tenant_id: int, conversation_id: str, question: str,
    shown_earlier: list[int],
) -> AsyncIterator[str]:
    """Stream agent events as UI chunks; when the agent is done, resolve citations and persist."""
    pool = request.app.state.pool
    deadline = time.monotonic() + request.app.state.settings.chat_timeout_seconds
    message_id = uuid.uuid4()
    encoder = ChatStreamEncoder(str(message_id))
    for chunk in encoder.start():
        yield chunk
    try:
        while True:
            try:
                event = await asyncio.wait_for(anext(events), deadline - time.monotonic())
            except StopAsyncIteration:
                break
            if isinstance(event, Done):
                valid, rejected = split_citations(
                    event.answer, [*event.trace["retrieved_chunk_ids"], *shown_earlier])
                if rejected:
                    log.warning("conversation %s: model cited unseen chunks %s", conversation_id, rejected)
                event.trace["rejected_citation_ids"] = rejected
                event.trace["ungrounded"] = is_ungrounded(
                    event.answer, event.finish_reason, valid, [t["name"] for t in event.trace["tool_calls"]])
                log.info("chat turn finished", extra={
                    "conversation_id": conversation_id, "model": event.trace["model"],
                    "finish_reason": event.finish_reason, "latency_ms": event.trace["latency_ms"],
                    "input_tokens": event.trace["input_tokens"],
                    "output_tokens": event.trace["output_tokens"],
                    "cost_usd": event.trace["cost_usd"], "tool_calls": len(event.trace["tool_calls"]),
                    "rejected_citations": len(rejected), "ungrounded": event.trace["ungrounded"],
                })
                citations = await store.get_chunks(pool, tenant_id, valid)
                await store.save_turn(pool, tenant_id=tenant_id, conversation_id=conversation_id,
                                      user_text=question,
                                      assistant_id=message_id, assistant_text=event.answer,
                                      trace=event.trace)
                chunks = encoder.finish(citations, event.trace, event.finish_reason)
            else:
                chunks = encoder.encode(event)
            for chunk in chunks:
                yield chunk
    except TimeoutError:
        log.warning("chat turn timed out", extra={"conversation_id": conversation_id})
        for chunk in encoder.fail("This took too long and was stopped. Please try again."):
            yield chunk
    except Exception:  # surface failures in the chat UI instead of a dropped stream
        # The exception may quote internal URLs or upstream responses: log it, don't send it.
        log.exception("chat turn failed", extra={"conversation_id": conversation_id})
        for chunk in encoder.fail(f"Something went wrong. Reference: {request_id.get()}"):
            yield chunk


@router.get("/conversations/{conversation_id}")
async def get_conversation(request: Request, conversation_id: str,
                           principal: CurrentPrincipal) -> dict:
    pool = request.app.state.pool
    exists, owner = await store.conversation_owner(pool, principal.tenant_id, conversation_id)
    messages = (await store.conversation_history(pool, principal.tenant_id, conversation_id)
                if exists and owner == principal.user_id else [])
    if not messages:
        raise HTTPException(404, "conversation not found")
    return {"id": conversation_id, "messages": messages}


@router.get("/traces/{message_id}")
async def get_trace(request: Request, message_id: uuid.UUID,
                    principal: CurrentPrincipal) -> dict:
    trace = await store.get_trace(request.app.state.pool, principal.tenant_id, message_id,
                                  principal.user_id)
    if trace is None:
        raise HTTPException(404, "trace not found")
    return trace
