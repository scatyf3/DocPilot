"""Embedding providers.

- `openai`: any OpenAI-compatible /v1/embeddings endpoint (OpenAI, or the LLM Gateway).
- `fake`: deterministic feature-hashed bag of words, so tests and offline dev need no key.
  It behaves like a crude lexical retriever, which is still useful for an ablation baseline.
"""

import hashlib
import re
from typing import Protocol

import numpy as np
from openai import AsyncOpenAI

from docpilot.config import Settings


class Embedder(Protocol):
    name: str

    async def embed(self, texts: list[str]) -> list[np.ndarray]: ...


class OpenAIEmbedder:
    def __init__(self, settings: Settings):
        self.name = f"openai:{settings.embedding_model}"
        self._client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        self._model = settings.embedding_model
        self._dim = settings.embedding_dim
        self._batch = settings.embedding_batch_size

    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self._batch):
            resp = await self._client.embeddings.create(
                model=self._model, input=texts[i : i + self._batch], dimensions=self._dim
            )
            out.extend(np.asarray(d.embedding, dtype=np.float32) for d in resp.data)
        return out


_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = set(
    "a an and are as at be by for from has have how in is it its of on or that the this to "
    "was what when where which who why will with does do did our we you your".split()
)


class FakeEmbedder:
    """Hash each word into one of `dim` buckets (with a sign), then L2-normalise."""

    def __init__(self, dim: int = 1024):
        self.name = "fake:hashing-bow"
        self._dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for word in _TOKEN.findall(text.lower()):
            if word in _STOPWORDS:
                continue
            h = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "little")
            vec[h % self._dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = np.linalg.norm(vec)
        if norm == 0:
            vec[0] = 1.0  # pgvector cosine distance is undefined for the zero vector
            return vec
        return vec / norm

    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        return [self._vector(t) for t in texts]


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "fake":
        return FakeEmbedder(settings.embedding_dim)
    return OpenAIEmbedder(settings)
