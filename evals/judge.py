"""LLM-as-judge: grades one answer for correctness, citation faithfulness and refusal.

Claude returns a JSON verdict constrained by `output_config.format` (structured outputs), so
we never parse free text. The judge sees the reference answer and the text of every chunk
the answer cited, which is what makes "is the citation faithful?" checkable.
"""

import json
import os

from anthropic import AsyncAnthropic

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-5-5")
JUDGE_EFFORT = os.environ.get("JUDGE_EFFORT", "low")

UNANSWERABLE = "UNANSWERABLE: the documents do not contain the answer."

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "citations_faithful": {"type": "boolean"},
        "refused": {"type": "boolean"},
        "explanation": {"type": "string"},
    },
    "required": ["correct", "citations_faithful", "refused", "explanation"],
    "additionalProperties": False,
}

INSTRUCTIONS = """You grade answers produced by a document question-answering assistant.

Return:
- correct: true if the candidate answer states the key facts of the reference answer and \
makes no claim that contradicts it. Extra correct detail is fine. If the reference says the \
question is UNANSWERABLE, correct is true only if the candidate declines to answer.
- citations_faithful: true if every claim that carries a citation [c:<id>] is supported by \
the text of that cited source, and the answer's main factual claims are cited. True for a \
pure refusal with no claims.
- refused: true if the candidate says it could not find the answer in the documents.
- explanation: one or two sentences."""


class Judge:
    def __init__(self, model: str = JUDGE_MODEL):
        self.model = model
        self._client = AsyncAnthropic()
        self.input_tokens = self.output_tokens = 0

    async def grade(self, question: str, reference: str | None, answer: str,
                    cited: list[dict]) -> dict:
        sources = "\n\n".join(f'<source id="{c["id"]}" title="{c["title"]}">\n{c["text"]}\n</source>'
                              for c in cited) or "(no sources cited)"
        prompt = (
            f"<question>\n{question}\n</question>\n\n"
            f"<reference_answer>\n{reference or UNANSWERABLE}\n</reference_answer>\n\n"
            f"<candidate_answer>\n{answer}\n</candidate_answer>\n\n"
            f"<cited_sources>\n{sources}\n</cited_sources>"
        )
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=INSTRUCTIONS,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": JUDGE_EFFORT,
                           "format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
        )
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        if response.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"judge stopped with {response.stop_reason}")
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)
