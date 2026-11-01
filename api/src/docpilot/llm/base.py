"""Provider-neutral types for one streamed model turn.

The agent loop (agent/loop.py) is written once against this interface. Each provider keeps
the conversation in its own native message format (Anthropic content blocks, OpenAI
chat messages, ...) and is the only code that touches that format.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

StopReason = Literal["end", "tool_use", "max_tokens", "refusal"]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema of the tool's input object


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any] | None  # None when the model produced unparseable JSON
    raw_args: str = ""


@dataclass
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass
class TextDelta:
    text: str


@dataclass
class TurnResult:
    """Emitted once, after the last TextDelta of a turn."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: StopReason = "end"
    input_tokens: int = 0
    output_tokens: int = 0


class LLMProvider(Protocol):
    model: str

    def start(self, history: list[dict], user_text: str) -> list:
        """Build native messages from prior {role, content} turns plus the new question."""
        ...

    def stream_turn(
        self, system: str, messages: list, tools: list[ToolSpec], *, allow_tools: bool = True
    ) -> AsyncIterator[TextDelta | TurnResult]:
        """Stream one model turn; appends the assistant turn to `messages` before finishing.

        With allow_tools=False the model must answer in text. `tools` is still sent, because
        the history may hold tool calls and providers require the tools to stay defined.
        """
        ...

    def add_tool_results(self, messages: list, results: list[ToolResult]) -> None:
        """Append all results for the turn (one message for Anthropic, N for OpenAI)."""
        ...
