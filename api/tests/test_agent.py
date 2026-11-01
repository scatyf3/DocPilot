"""Agent loop with the fake provider and a stubbed retrieval layer (no DB, no keys)."""

from docpilot.agent.loop import Done, StepStart, ToolFinished, ToolStarted, run_agent
from docpilot.agent.prompts import INCOMPLETE, REFUSAL
from docpilot.agent.tools import Toolbox
from docpilot.llm.base import TextDelta, ToolCall, TurnResult
from docpilot.llm.fake_provider import FakeProvider
from docpilot.retrieval.search import RetrievedChunk

CHUNKS = [
    RetrievedChunk(id=11, document_id=1, ord=0, title="Benefits", score=0.9,
                   text="## PTO\n\nEmployees get 25 days of PTO per year. Unused days expire."),
    RetrievedChunk(id=12, document_id=2, ord=3, title="Runbook", score=0.5,
                   text="Pages go to the primary on-call first."),
]


def make_toolbox(chunks=CHUNKS, created=None):
    async def search(query, top_k):
        return chunks[:top_k]

    async def read_document(doc_id):
        return None

    async def create_page(title, markdown):
        created.append((title, markdown))
        return 99

    return Toolbox(search, read_document, create_page)


async def collect(provider, toolbox, question="How many PTO days do employees get?", **kw):
    return [e async for e in run_agent(provider, toolbox, [], question, **kw)]


async def test_fake_agent_searches_then_answers_with_citation():
    events = await collect(FakeProvider(), make_toolbox())
    assert [type(e).__name__ for e in events[:3]] == ["StepStart", "ToolStarted", "ToolFinished"]
    done = events[-1]
    assert isinstance(done, Done)
    assert "25 days" in done.answer and "[c:11]" in done.answer
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == done.answer
    trace = done.trace
    assert trace["retrieved_chunk_ids"] == [11, 12]
    assert trace["tool_calls"][0]["name"] == "search_docs"
    assert trace["model"] == "fake" and trace["cost_usd"] == 0.0
    assert trace["input_tokens"] > 0 and trace["latency_ms"] >= 0


async def test_fake_agent_refuses_when_nothing_retrieved():
    done = (await collect(FakeProvider(), make_toolbox(chunks=[])))[-1]
    assert done.answer == REFUSAL


class ScriptedProvider:
    """Replays fixed turns so we can test the loop's control flow."""

    model = "claude-opus-5-5"

    def __init__(self, turns):
        self.turns = list(turns)
        self.tool_result_batches = []
        self.allow_tools = []

    def start(self, history, user_text):
        return [{"role": "user", "content": user_text}]

    async def stream_turn(self, system, messages, tools, *, allow_tools=True):
        self.allow_tools.append(allow_tools)
        turn = self.turns.pop(0)
        if turn.text:
            yield TextDelta(turn.text)
        yield turn

    def add_tool_results(self, messages, results):
        self.tool_result_batches.append(results)


async def test_tool_results_go_back_in_one_batch_and_bad_args_become_errors():
    calls = [ToolCall("a", "search_docs", {"query": "pto"}),
             ToolCall("b", "search_docs", {"top_k": 3}),           # missing query
             ToolCall("c", "create_page", None, raw_args="{oops")]  # unparseable JSON
    provider = ScriptedProvider([
        TurnResult("", calls, "tool_use", 100, 20),
        TurnResult("Done [c:11]", [], "end", 300, 40),
    ])
    created = []
    events = await collect(provider, make_toolbox(created=created))
    [batch] = provider.tool_result_batches
    assert [r.call_id for r in batch] == ["a", "b", "c"]
    assert [r.is_error for r in batch] == [False, True, True]
    assert created == []
    done = events[-1]
    assert done.answer == "Done [c:11]"
    assert done.trace["input_tokens"] == 400 and done.trace["output_tokens"] == 60
    assert done.trace["cost_usd"] == round((400 * 4 + 60 * 20) / 1e6, 6)


async def test_tools_never_run_after_max_tokens_or_refusal():
    for stop, reason in [("max_tokens", "length"), ("refusal", "content-filter")]:
        call = ToolCall("a", "create_page", {"title": "t", "markdown": "m"})
        provider = ScriptedProvider([TurnResult("", [call], stop, 10, 10)])
        created = []
        events = await collect(provider, make_toolbox(created=created))
        assert created == [] and provider.tool_result_batches == []
        assert not any(isinstance(e, ToolStarted) for e in events)
        assert events[-1].finish_reason == reason


async def test_step_budget_is_enforced():
    call = ToolCall("a", "search_docs", {"query": "x"})
    provider = ScriptedProvider([TurnResult("", [call], "tool_use", 1, 1) for _ in range(3)])
    events = await collect(provider, make_toolbox(), max_steps=3)
    assert sum(isinstance(e, StepStart) for e in events) == 3
    assert sum(isinstance(e, ToolFinished) for e in events) == 2
    assert events[-1].finish_reason == "other"
    assert provider.allow_tools == [True, True, False]
    assert events[-1].answer == INCOMPLETE  # a stubborn model still leaves something to show


async def test_last_step_answers_from_what_it_found():
    call = ToolCall("a", "search_docs", {"query": "x"})
    provider = ScriptedProvider([TurnResult("", [call], "tool_use", 1, 1),
                                 TurnResult("25 days [c:11]", [], "end", 1, 1)])
    done = (await collect(provider, make_toolbox(), max_steps=2))[-1]
    assert provider.allow_tools == [True, False]
    assert done.answer == "25 days [c:11]" and done.finish_reason == "stop"


async def test_empty_answers_are_replaced_with_a_note():
    for stop, reason in [("max_tokens", "length"), ("end", "other")]:
        provider = ScriptedProvider([TurnResult("", [], stop, 1, 1)])
        events = await collect(provider, make_toolbox())
        assert events[-1].answer == INCOMPLETE and events[-1].finish_reason == reason
        assert "".join(e.text for e in events if isinstance(e, TextDelta)) == INCOMPLETE


async def test_create_page_tool():
    created = []
    outcome = await make_toolbox(created=created).run(
        ToolCall("p", "create_page", {"title": "Summary", "markdown": "# S"})
    )
    assert created == [("Summary", "# S")]
    assert outcome.output == {"document_id": 99, "title": "Summary"}
    assert not outcome.result.is_error


async def test_tool_schemas_forbid_extra_fields():
    from docpilot.agent.tools import TOOL_SPECS

    by_name = {s.name: s.parameters for s in TOOL_SPECS}
    assert set(by_name) == {"search_docs", "read_doc", "create_page"}
    assert by_name["search_docs"]["required"] == ["query"]
    assert all(p["additionalProperties"] is False for p in by_name.values())


async def test_tool_failures_tell_the_model_without_leaking_details():
    async def search(query, top_k):
        raise RuntimeError("connection to postgres://user:hunter2@db-host refused")

    async def unused(*args):
        return None

    outcome = await Toolbox(search, unused, unused).run(ToolCall("a", "search_docs", {"query": "x"}))
    assert outcome.result.is_error and "RuntimeError" in outcome.result.content
    assert "hunter2" not in outcome.result.content and "hunter2" not in outcome.summary
