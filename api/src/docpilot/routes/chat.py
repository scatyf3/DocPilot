"""POST /chat (streaming agent answer) plus conversation and trace lookups.

/chat speaks the AI SDK UI message stream, so the web app's `useChat` consumes it directly,
and the eval suite replays labeled questions through this same endpoint.
"""

import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from docpilot import store
from docpilot import ui_stream as ui
from docpilot.agent.citations import cited_chunk_ids
from docpilot.agent.loop import Done, run_agent
from docpilot.agent.tools import Toolbox
from docpilot.chat_stream import ChatStreamEncoder
from docpilot.config import RerankerName, RetrievalMode
from docpilot.ingest.pipeline import ingest_text

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


class ChatRequest(BaseModel):
    """What DefaultChatTransport POSTs: {id, messages, trigger, messageId, ...body}."""

    model_config = ConfigDict(extra="allow")
    id: str
    messages: list[UIMessage]
    retrieval: RetrievalOptions = RetrievalOptions()

    def question(self) -> str:
        for m in reversed(self.messages):
            if m.role == "user":
                return "".join(p.text or "" for p in m.parts if p.type == "text").strip()
        return ""


@router.post("/chat")
async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    question = body.question()
    if not question:
        raise HTTPException(422, "no user message to answer")
    s = request.app.state
    # The database, not the client, is the source of truth for earlier turns.
    history = [{"role": m["role"], "content": m["content"]}
               for m in await store.conversation_history(s.pool, body.id)]

    async def search(query: str, top_k: int):
        return await s.retriever.search(
            query, top_k=top_k, mode=body.retrieval.mode, reranker=body.retrieval.reranker
        )

    async def read_document(doc_id: int):
        return await store.get_document(s.pool, doc_id)

    async def create_page(title: str, markdown: str) -> int:
        doc_id, _ = await ingest_text(s.pool, s.embedder, s.settings, title=title,
                                      text=markdown, source="agent", kind="generated")
        return doc_id

    toolbox = Toolbox(search, read_document, create_page)
    events = run_agent(s.llm, toolbox, history, question, max_steps=s.settings.max_agent_steps)
    return StreamingResponse(
        _encode(events, request, body.id, question),
        media_type="text/event-stream",
        headers=ui.HEADERS,
    )


async def _encode(events, request: Request, conversation_id: str, question: str) -> AsyncIterator[str]:
    """Stream agent events as UI chunks; when the agent is done, resolve citations and persist."""
    pool = request.app.state.pool
    message_id = uuid.uuid4()
    encoder = ChatStreamEncoder(str(message_id))
    for chunk in encoder.start():
        yield chunk
    try:
        async for event in events:
            if isinstance(event, Done):
                citations = await store.get_chunks(pool, cited_chunk_ids(event.answer))
                await store.save_turn(pool, conversation_id=conversation_id, user_text=question,
                                      assistant_id=message_id, assistant_text=event.answer,
                                      trace=event.trace)
                chunks = encoder.finish(citations, event.trace, event.finish_reason)
            else:
                chunks = encoder.encode(event)
            for chunk in chunks:
                yield chunk
    except Exception as e:  # surface failures in the chat UI instead of a dropped stream
        log.exception("chat turn failed")
        for chunk in encoder.fail(f"{type(e).__name__}: {e}"):
            yield chunk


@router.get("/conversations/{conversation_id}")
async def get_conversation(request: Request, conversation_id: str) -> dict:
    messages = await store.conversation_history(request.app.state.pool, conversation_id)
    if not messages:
        raise HTTPException(404, "conversation not found")
    return {"id": conversation_id, "messages": messages}


@router.get("/traces/{message_id}")
async def get_trace(request: Request, message_id: uuid.UUID) -> dict:
    trace = await store.get_trace(request.app.state.pool, message_id)
    if trace is None:
        raise HTTPException(404, "trace not found")
    return trace
