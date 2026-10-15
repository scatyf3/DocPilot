import json

from docpilot import ui_stream as ui
from docpilot.agent.loop import StepStart, ToolFinished, ToolStarted
from docpilot.agent.tools import ToolOutcome
from docpilot.chat_stream import ChatStreamEncoder
from docpilot.llm.base import TextDelta, ToolCall, ToolResult


def parse(chunks: list[str]) -> list:
    """Decode SSE lines back into JSON chunks (or the [DONE] sentinel)."""
    out = []
    for c in chunks:
        assert c.startswith("data: ") and c.endswith("\n\n")
        body = c[len("data: "):-2]
        out.append(body if body == "[DONE]" else json.loads(body))
    return out


def test_sse_framing():
    assert ui.text_delta("t1", "hi") == 'data: {"type":"text-delta","id":"t1","delta":"hi"}\n\n'
    assert ui.DONE == "data: [DONE]\n\n"
    assert ui.HEADERS["x-vercel-ai-ui-message-stream"] == "v1"


def test_encoder_brackets_text_parts_and_steps():
    enc = ChatStreamEncoder("m1")
    call = ToolCall(id="call_1", name="search_docs", args={"query": "pto"})
    outcome = ToolOutcome(result=ToolResult("call_1", "..."), output=[{"id": 1}], summary="1 chunk")
    chunks = enc.start()
    for event in [StepStart(0), TextDelta("Let me look."), ToolStarted(call),
                  ToolFinished(call, outcome), StepStart(1), TextDelta("25 days "),
                  TextDelta("[c:1]")]:
        chunks += enc.encode(event)
    chunks += enc.finish([{"id": 1}], {"input_tokens": 3}, "stop")

    parsed = parse(chunks)
    types = [p if p == "[DONE]" else p["type"] for p in parsed]
    assert types == [
        "start", "start-step", "text-start", "text-delta", "text-end",
        "tool-input-available", "tool-output-available",
        "finish-step", "start-step", "text-start", "text-delta", "text-delta", "text-end",
        "finish-step", "data-citations", "data-trace", "finish", "[DONE]",
    ]
    assert parsed[0] == {"type": "start", "messageId": "m1"}
    assert parsed[5]["dynamic"] is True and parsed[5]["input"] == {"query": "pto"}
    deltas = [p for p in parsed if p != "[DONE]" and p["type"] == "text-delta"]
    assert {d["id"] for d in deltas} == {"text-0", "text-1"}
    assert parsed[-3]["data"]["message_id"] == "m1"


def test_tool_error_and_failure():
    enc = ChatStreamEncoder("m2")
    call = ToolCall(id="c", name="read_doc", args={"document_id": 9})
    bad = ToolOutcome(result=ToolResult("c", "Error", is_error=True), output={}, summary="no doc")
    chunks = enc.encode(StepStart(0)) + enc.encode(ToolFinished(call, bad)) + enc.fail("boom")
    parsed = parse(chunks)
    assert parsed[1] == {"type": "tool-output-error", "toolCallId": "c", "errorText": "no doc",
                         "dynamic": True}
    assert parsed[-2] == {"type": "error", "errorText": "boom"} and parsed[-1] == "[DONE]"
