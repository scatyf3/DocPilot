"""Encoder for the Vercel AI SDK "UI message stream" protocol (v1).

The web client's `useChat` + `DefaultChatTransport` parses exactly this: Server-Sent Events
where each `data:` line is one JSON chunk, terminated by `data: [DONE]`. Chunk shapes were
checked against `uiMessageChunkSchema` in the installed `ai` package.
"""

import json
from typing import Any

HEADERS = {
    "x-vercel-ai-ui-message-stream": "v1",
    "cache-control": "no-cache",
    "x-accel-buffering": "no",  # stop proxies (nginx, Fly) from buffering the stream
}
DONE = "data: [DONE]\n\n"


def sse(chunk: dict[str, Any]) -> str:
    return f"data: {json.dumps(chunk, separators=(',', ':'), default=str)}\n\n"


def start(message_id: str) -> str:
    return sse({"type": "start", "messageId": message_id})


def start_step() -> str:
    return sse({"type": "start-step"})


def finish_step() -> str:
    return sse({"type": "finish-step"})


def text_start(part_id: str) -> str:
    return sse({"type": "text-start", "id": part_id})


def text_delta(part_id: str, delta: str) -> str:
    return sse({"type": "text-delta", "id": part_id, "delta": delta})


def text_end(part_id: str) -> str:
    return sse({"type": "text-end", "id": part_id})


# Our tools run on the server and aren't declared in the client, so they are "dynamic" tools.
def tool_input(call_id: str, name: str, args: Any) -> str:
    return sse(
        {"type": "tool-input-available", "toolCallId": call_id, "toolName": name,
         "input": args, "dynamic": True}
    )


def tool_output(call_id: str, output: Any) -> str:
    return sse({"type": "tool-output-available", "toolCallId": call_id, "output": output,
                "dynamic": True})


def tool_error(call_id: str, error: str) -> str:
    return sse({"type": "tool-output-error", "toolCallId": call_id, "errorText": error,
                "dynamic": True})


def data(name: str, payload: Any) -> str:
    """Custom `data-<name>` part; the client stores it on the message's `parts`."""
    return sse({"type": f"data-{name}", "data": payload})


def error(text: str) -> str:
    return sse({"type": "error", "errorText": text})


def finish(reason: str = "stop") -> str:
    return sse({"type": "finish", "finishReason": reason})
