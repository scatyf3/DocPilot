"""Any OpenAI-compatible /v1/chat/completions endpoint with function calling.

Point OPENAI_BASE_URL at the LLM Gateway (or vLLM, OpenRouter, ...) and set CHAT_MODEL.
Streamed tool calls arrive as fragments keyed by `index`; we stitch them back together.
"""

import json
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from docpilot.config import Settings
from docpilot.llm.base import StopReason, TextDelta, ToolCall, ToolResult, ToolSpec, TurnResult

_STOP: dict[str, StopReason] = {
    "stop": "end", "tool_calls": "tool_use", "function_call": "tool_use",
    "length": "max_tokens", "content_filter": "refusal",
}


class OpenAIProvider:
    def __init__(self, settings: Settings):
        self.model = settings.chat_model
        self._client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

    def start(self, history: list[dict], user_text: str) -> list:
        return [*({"role": m["role"], "content": m["content"]} for m in history),
                {"role": "user", "content": user_text}]

    async def stream_turn(
        self, system: str, messages: list, tools: list[ToolSpec]
    ) -> AsyncIterator[TextDelta | TurnResult]:
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            tools=[{"type": "function",
                    "function": {"name": t.name, "description": t.description,
                                 "parameters": t.parameters}}
                   for t in tools],
            stream=True,
            stream_options={"include_usage": True},
        )
        text: list[str] = []
        partial_calls: dict[int, dict[str, str]] = {}
        finish_reason, usage = "stop", None

        async for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.delta.content:
                text.append(choice.delta.content)
                yield TextDelta(choice.delta.content)
            for frag in choice.delta.tool_calls or []:
                slot = partial_calls.setdefault(frag.index, {"id": "", "name": "", "arguments": ""})
                slot["id"] = frag.id or slot["id"]
                if frag.function:
                    slot["name"] += frag.function.name or ""
                    slot["arguments"] += frag.function.arguments or ""
            if choice.finish_reason:
                finish_reason = choice.finish_reason

        calls = [ToolCall(id=c["id"], name=c["name"], args=_parse_args(c["arguments"]),
                          raw_args=c["arguments"])
                 for _, c in sorted(partial_calls.items())]
        assistant: dict = {"role": "assistant", "content": "".join(text) or None}
        if calls:
            assistant["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": c.raw_args}}
                for c in calls
            ]
        messages.append(assistant)
        stop = _STOP.get(finish_reason, "end")
        if calls and stop == "end":
            stop = "tool_use"  # some servers report "stop" even when they emitted tool calls
        yield TurnResult(
            text="".join(text),
            tool_calls=calls,
            stop_reason=stop,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    def add_tool_results(self, messages: list, results: list[ToolResult]) -> None:
        for r in results:
            messages.append({"role": "tool", "tool_call_id": r.call_id, "content": r.content})


def _parse_args(raw: str) -> dict | None:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    return args if isinstance(args, dict) else None
