"""Claude via the official Anthropic SDK (streaming Messages API, manual tool loop).

API rules this code relies on (current Claude models):
- Adaptive thinking is always on; depth is set with output_config.effort, never budget_tokens.
- The full `response.content` (thinking + tool_use blocks) goes back into history unchanged.
- All tool_result blocks for one turn are sent back in a single user message.
- tool_choice stays "auto"; the system prompt tells Claude when to search.
"""

from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from docpilot.config import Settings
from docpilot.llm.base import StopReason, TextDelta, ToolCall, ToolResult, ToolSpec, TurnResult

_STOP: dict[str, StopReason] = {
    "end_turn": "end", "stop_sequence": "end", "tool_use": "tool_use",
    "max_tokens": "max_tokens", "refusal": "refusal",
}


class AnthropicProvider:
    def __init__(self, settings: Settings):
        self.model = settings.anthropic_model
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._effort = settings.anthropic_effort
        self._max_tokens = settings.max_output_tokens

    def start(self, history: list[dict], user_text: str) -> list:
        return [*({"role": m["role"], "content": m["content"]} for m in history),
                {"role": "user", "content": user_text}]

    async def stream_turn(
        self, system: str, messages: list, tools: list[ToolSpec]
    ) -> AsyncIterator[TextDelta | TurnResult]:
        # Tool inputs here are small, so we keep the API's default buffered tool-input
        # streaming: the server validates each input against its schema before sending it.
        async with self._client.messages.stream(
            model=self.model,
            max_tokens=self._max_tokens,
            system=system,
            messages=messages,
            tools=[{"name": t.name, "description": t.description, "input_schema": t.parameters}
                   for t in tools],
            thinking={"type": "adaptive"},
            output_config={"effort": self._effort},
        ) as stream:
            async for event in stream:
                if event.type == "text":
                    yield TextDelta(event.text)
            final = await stream.get_final_message()

        messages.append({"role": "assistant", "content": final.content})
        calls = [
            ToolCall(id=b.id, name=b.name, args=b.input if isinstance(b.input, dict) else None,
                     raw_args=str(b.input))
            for b in final.content if b.type == "tool_use"
        ]
        yield TurnResult(
            text="".join(b.text for b in final.content if b.type == "text"),
            tool_calls=calls,
            stop_reason=_STOP.get(final.stop_reason or "end_turn", "end"),
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
        )

    def add_tool_results(self, messages: list, results: list[ToolResult]) -> None:
        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content,
                 "is_error": r.is_error}
                for r in results
            ],
        })
