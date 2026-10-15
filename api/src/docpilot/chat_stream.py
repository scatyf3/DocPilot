"""Turns agent events into UI message stream chunks.

Stateful because the protocol is: every text part is bracketed by text-start/text-end with
one id, and every agent step by start-step/finish-step. Kept free of I/O so it is unit-tested.
"""

from typing import Any

from docpilot import ui_stream as ui
from docpilot.agent.loop import AgentEvent, StepStart, ToolFinished, ToolStarted
from docpilot.llm.base import TextDelta


class ChatStreamEncoder:
    def __init__(self, message_id: str):
        self.message_id = message_id
        self._text_id: str | None = None
        self._step = 0
        self._in_step = False

    def start(self) -> list[str]:
        return [ui.start(self.message_id)]

    def encode(self, event: AgentEvent) -> list[str]:
        if isinstance(event, StepStart):
            out = self._close_step()
            self._step, self._in_step = event.step, True
            return [*out, ui.start_step()]
        if isinstance(event, TextDelta):
            out = []
            if self._text_id is None:
                self._text_id = f"text-{self._step}"
                out.append(ui.text_start(self._text_id))
            return [*out, ui.text_delta(self._text_id, event.text)]
        if isinstance(event, ToolStarted):
            return [*self._close_text(), ui.tool_input(event.call.id, event.call.name, event.call.args)]
        if isinstance(event, ToolFinished):
            if event.outcome.result.is_error:
                return [ui.tool_error(event.call.id, event.outcome.summary)]
            return [ui.tool_output(event.call.id, event.outcome.output)]
        return []

    def finish(self, citations: list[dict], trace: dict[str, Any], finish_reason: str) -> list[str]:
        return [
            *self._close_step(),
            ui.data("citations", citations),
            ui.data("trace", {"message_id": self.message_id, **trace}),
            ui.finish(finish_reason),
            ui.DONE,
        ]

    def fail(self, message: str) -> list[str]:
        return [*self._close_step(), ui.error(message), ui.DONE]

    def _close_text(self) -> list[str]:
        if self._text_id is None:
            return []
        chunk, self._text_id = ui.text_end(self._text_id), None
        return [chunk]

    def _close_step(self) -> list[str]:
        if not self._in_step:
            return []
        self._in_step = False
        return [*self._close_text(), ui.finish_step()]
