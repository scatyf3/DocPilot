"""Optional second-stage reranker.

A cross-encoder reads (query, chunk) together, so it judges relevance far better than the
first-stage retrievers, but it is too slow to run over the whole corpus. We only rerank the
fused top candidates. `none` keeps fusion order; `cohere` calls Cohere's hosted Rerank API.
"""

from typing import Protocol

import httpx

from docpilot.config import Settings


class Reranker(Protocol):
    async def rerank(self, query: str, texts: list[str]) -> list[tuple[int, float]]:
        """Return (index into texts, relevance score), best first."""
        ...


class NoReranker:
    async def rerank(self, query: str, texts: list[str]) -> list[tuple[int, float]]:
        return [(i, 0.0) for i in range(len(texts))]


class CohereReranker:
    URL = "https://api.cohere.com/v2/rerank"

    def __init__(self, api_key: str, model: str):
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._model = model

    async def rerank(self, query: str, texts: list[str]) -> list[tuple[int, float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self.URL,
                headers=self._headers,
                json={"model": self._model, "query": query, "documents": texts},
            )
            resp.raise_for_status()
        return [(r["index"], r["relevance_score"]) for r in resp.json()["results"]]


def make_reranker(name: str, settings: Settings) -> Reranker:
    if name == "cohere":
        if not settings.rerank_api_key:
            raise ValueError("RERANKER=cohere requires RERANK_API_KEY")
        return CohereReranker(settings.rerank_api_key, settings.rerank_model)
    return NoReranker()
