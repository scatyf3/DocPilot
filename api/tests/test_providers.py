"""Provider adapters against stubbed SDK clients: request shape in, normalised turn out."""

from types import SimpleNamespace as NS

from docpilot.agent.tools import TOOL_SPECS
from docpilot.config import Settings
from docpilot.llm.anthropic_provider import AnthropicProvider
from docpilot.llm.base import TextDelta, ToolResult, TurnResult
from docpilot.llm.openai_provider import OpenAIProvider


class FakeAnthropicStream:
    def __init__(self, events, final):
        self.events, self.final = events, final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def gen():
            for e in self.events:
                yield e
        return gen()

    async def get_final_message(self):
        return self.final


async def drain(gen):
    return [e async for e in gen]


async def test_anthropic_request_shape_and_turn_result():
    provider = AnthropicProvider(Settings(anthropic_api_key="test", anthropic_effort="medium"))
    content = [NS(type="thinking", thinking="", signature="sig"),
               NS(type="text", text="Let me search."),
               NS(type="tool_use", id="toolu_1", name="search_docs", input={"query": "pto"})]
    final = NS(content=content, stop_reason="tool_use", usage=NS(input_tokens=50, output_tokens=7))
    captured = {}

    def stream(**kwargs):
        captured.update(kwargs)
        return FakeAnthropicStream([NS(type="text", text="Let me search.")], final)

    provider._client = NS(messages=NS(stream=stream))
    messages = provider.start([{"role": "assistant", "content": "earlier"}], "How much PTO?")
    events = await drain(provider.stream_turn("sys", messages, TOOL_SPECS))

    assert captured["thinking"] == {"type": "adaptive"}
    assert captured["output_config"] == {"effort": "medium"}
    assert "tool_choice" not in captured and "budget_tokens" not in str(captured)
    assert {t["name"] for t in captured["tools"]} == {"search_docs", "read_doc", "create_page"}
    assert isinstance(events[0], TextDelta)
    turn = events[-1]
    assert isinstance(turn, TurnResult) and turn.stop_reason == "tool_use"
    assert turn.tool_calls[0].args == {"query": "pto"} and turn.input_tokens == 50
    # Full content, thinking block included, goes back into history unchanged.
    assert messages[-1] == {"role": "assistant", "content": content}

    provider.add_tool_results(messages, [ToolResult("toolu_1", "a"), ToolResult("toolu_2", "b", True)])
    assert messages[-1]["role"] == "user" and len(messages[-1]["content"]) == 2


async def test_openai_stitches_streamed_tool_call_fragments():
    provider = OpenAIProvider(Settings(openai_api_key="test", chat_model="gateway-model"))

    def chunk(content=None, tool_calls=None, finish=None, usage=None):
        choices = [] if usage else [NS(delta=NS(content=content, tool_calls=tool_calls),
                                       finish_reason=finish)]
        return NS(choices=choices, usage=usage)

    def frag(index, id=None, name=None, args=None):
        return NS(index=index, id=id, function=NS(name=name, arguments=args))

    chunks = [
        chunk(content="Checking"),
        chunk(tool_calls=[frag(0, id="call_a", name="search_docs", args='{"que')]),
        chunk(tool_calls=[frag(0, args='ry": "pto"}')]),
        chunk(tool_calls=[frag(1, id="call_b", name="read_doc", args="{bad json")]),
        chunk(finish="tool_calls"),
        chunk(usage=NS(prompt_tokens=30, completion_tokens=9)),
    ]

    async def create(**kwargs):
        assert kwargs["stream"] is True and kwargs["messages"][0]["role"] == "system"

        async def gen():
            for c in chunks:
                yield c
        return gen()

    provider._client = NS(chat=NS(completions=NS(create=create)))
    messages = provider.start([], "q")
    turn = (await drain(provider.stream_turn("sys", messages, TOOL_SPECS)))[-1]
    assert turn.stop_reason == "tool_use" and turn.text == "Checking"
    assert [(c.id, c.name, c.args) for c in turn.tool_calls] == [
        ("call_a", "search_docs", {"query": "pto"}), ("call_b", "read_doc", None)]
    assert (turn.input_tokens, turn.output_tokens) == (30, 9)
    provider.add_tool_results(messages, [ToolResult("call_a", "x"), ToolResult("call_b", "y")])
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "tool"]


async def test_providers_can_forbid_tools_on_the_last_step():
    settings = Settings(llm_provider="anthropic", anthropic_api_key="x", openai_api_key="x")
    captured = {}

    def stream(**kwargs):
        captured.update(kwargs)
        final = NS(content=[NS(type="text", text="ok")], stop_reason="end_turn",
                   usage=NS(input_tokens=1, output_tokens=1))
        return FakeAnthropicStream([NS(type="text", text="ok")], final)

    anthropic = AnthropicProvider(settings)
    anthropic._client = NS(messages=NS(stream=stream))
    await drain(anthropic.stream_turn("sys", anthropic.start([], "q"), TOOL_SPECS, allow_tools=False))
    assert captured["tool_choice"] == {"type": "none"} and captured["tools"]

    openai_kwargs = {}

    async def create(**kwargs):
        openai_kwargs.update(kwargs)

        async def chunks():
            yield NS(usage=None, choices=[NS(delta=NS(content="ok", tool_calls=None), finish_reason="stop")])

        return chunks()

    openai = OpenAIProvider(settings)
    openai._client = NS(chat=NS(completions=NS(create=create)))
    await drain(openai.stream_turn("sys", openai.start([], "q"), TOOL_SPECS, allow_tools=False))
    assert openai_kwargs["tool_choice"] == "none" and openai_kwargs["tools"]
