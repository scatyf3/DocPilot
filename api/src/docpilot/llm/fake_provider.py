"""Deterministic scripted "model" for tests, CI and offline demos (LLM_PROVIDER=fake).

Turn 1: call search_docs with the user's question.
Turn 2: answer with the sentence from the retrieved chunks that best overlaps the question,
        cited as [c:<id>], or refuse if nothing was retrieved.
It exercises the real agent loop, tools, citations, stream encoding and traces with no API key.
"""

import re
from collections.abc import AsyncIterator

from docpilot.agent.prompts import REFUSAL
from docpilot.llm.base import TextDelta, ToolCall, ToolResult, ToolSpec, TurnResult

_CHUNK = re.compile(r'<chunk id="(\d+)"[^>]*>\n(.*?)\n</chunk>', re.DOTALL)
_WORD = re.compile(r"[a-z0-9]+")


class FakeProvider:
    model = "fake"

    def start(self, history: list[dict], user_text: str) -> list:
        return [*history, {"role": "user", "content": user_text}]

    async def stream_turn(
        self, system: str, messages: list, tools: list[ToolSpec], *, allow_tools: bool = True
    ) -> AsyncIterator[TextDelta | TurnResult]:
        question = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        last = messages[-1]

        if last["role"] == "user" and allow_tools:
            call = ToolCall(id=f"call_{len(messages)}", name="search_docs", args={"query": question})
            messages.append({"role": "assistant", "tool_calls": [call]})
            yield TurnResult(text="", tool_calls=[call], stop_reason="tool_use",
                             input_tokens=_count(question), output_tokens=5)
            return

        chunks = [m for r in last.get("results", []) for m in _CHUNK.findall(r.content)]
        answer = _best_sentence(question, chunks) or REFUSAL
        for piece in re.findall(r"\S+\s*", answer):
            yield TextDelta(piece)
        messages.append({"role": "assistant", "content": answer})
        yield TurnResult(text=answer, input_tokens=sum(_count(t) for _, t in chunks),
                         output_tokens=_count(answer))

    def add_tool_results(self, messages: list, results: list[ToolResult]) -> None:
        messages.append({"role": "tool", "results": results})


def _count(text: str) -> int:
    return len(text.split())


def _best_sentence(question: str, chunks: list[tuple[str, str]]) -> str | None:
    q_words = set(_WORD.findall(question.lower()))
    best, best_overlap = None, 0
    for chunk_id, text in chunks:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            sentence = sentence.strip(" -*#>")
            overlap = len(q_words & set(_WORD.findall(sentence.lower())))
            if overlap > best_overlap:
                best, best_overlap = f"{sentence} [c:{chunk_id}]", overlap
    return best
