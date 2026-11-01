"""The agent loop: model turn -> run requested tools -> feed results back -> repeat.

The model decides when (and how often) to search; we only bound the number of steps.
The loop is provider-neutral and yields events that the /chat route turns into the UI
stream, and it ends with a `Done` event carrying the full answer and the trace.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from docpilot.agent.prompts import INCOMPLETE, SYSTEM_PROMPT
from docpilot.agent.tools import Toolbox, ToolOutcome
from docpilot.llm.base import LLMProvider, TextDelta, ToolCall, TurnResult
from docpilot.pricing import cost_usd


@dataclass
class StepStart:
    step: int


@dataclass
class ToolStarted:
    call: ToolCall


@dataclass
class ToolFinished:
    call: ToolCall
    outcome: ToolOutcome


@dataclass
class Done:
    answer: str
    finish_reason: str  # "stop" | "length" | "content-filter" | "other" (AI SDK vocabulary)
    trace: dict[str, Any]


AgentEvent = StepStart | TextDelta | ToolStarted | ToolFinished | Done


async def run_agent(
    provider: LLMProvider, toolbox: Toolbox, history: list[dict], question: str,
    max_steps: int = 6,
) -> AsyncIterator[AgentEvent]:
    started = time.perf_counter()
    messages = provider.start(history, question)
    step_texts: list[str] = []
    tool_log: list[dict] = []
    retrieved: dict[int, None] = {}  # ordered set of chunk ids the model was shown
    input_tokens = output_tokens = 0
    finish_reason = "stop"

    for step in range(max_steps):
        yield StepStart(step)
        turn: TurnResult | None = None
        # The last step can't call tools, so the model answers from what it has already found.
        last_step = step == max_steps - 1
        async for event in provider.stream_turn(SYSTEM_PROMPT, messages, toolbox.specs,
                                                allow_tools=not last_step):
            if isinstance(event, TextDelta):
                yield event
            else:
                turn = event
        assert turn is not None, "provider ended a turn without a TurnResult"
        input_tokens += turn.input_tokens
        output_tokens += turn.output_tokens
        if turn.text:
            step_texts.append(turn.text)

        # Check why the model stopped *before* running anything it asked for.
        if turn.stop_reason == "refusal":
            note = "The model declined to answer this request."
            yield TextDelta(note)
            step_texts.append(note)
            finish_reason = "content-filter"
            break
        if turn.stop_reason == "max_tokens":
            finish_reason = "length"  # a tool call cut off mid-input must not be executed
            break
        if not turn.tool_calls:
            break
        if last_step:
            finish_reason = "other"  # still asked for tools after the step budget: stop here
            break

        for call in turn.tool_calls:
            yield ToolStarted(call)
        outcomes = await asyncio.gather(*(toolbox.run(c) for c in turn.tool_calls))
        for call, outcome in zip(turn.tool_calls, outcomes, strict=True):
            retrieved.update(dict.fromkeys(outcome.chunk_ids))
            tool_log.append({"name": call.name, "args": call.args, "summary": outcome.summary,
                             "is_error": outcome.result.is_error,
                             "duration_ms": outcome.duration_ms})
            yield ToolFinished(call, outcome)
        provider.add_tool_results(messages, [o.result for o in outcomes])

    if not turn.text and finish_reason != "content-filter":  # never end a turn with nothing to show
        yield TextDelta(INCOMPLETE)
        step_texts.append(INCOMPLETE)
        if finish_reason == "stop":
            finish_reason = "other"

    yield Done(
        answer="\n\n".join(step_texts),
        finish_reason=finish_reason,
        trace={
            "model": provider.model,
            "retrieved_chunk_ids": list(retrieved),
            "tool_calls": tool_log,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "cost_usd": cost_usd(provider.model, input_tokens, output_tokens),
        },
    )
