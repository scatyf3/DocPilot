"""Redis caches for retrieval: query embeddings and whole search results.

Both are namespaced by tenant. Sharing them would be cheaper but leaks: a cache hit is faster
than a miss, so tenant B could time a query to learn whether tenant A had asked it, or embedded
that text, recently.

Search results are invalidated by generation, not by deleting keys. Each tenant has a counter
that every write to its library increments, and the counter is part of every result key, so a
write orphans the old results at once (they expire on their own). This holds up under races
too: a search that read generation 5 and finishes after a write may store a stale result, but
under generation 5, which nothing reads any more.

Every method swallows Redis errors and reports a miss: the cache makes things faster, never
makes them fail.
"""

import hashlib
import json
import logging
import time

import numpy as np
import redis.asyncio as aioredis
from redis.exceptions import RedisError

log = logging.getLogger(__name__)
PREFIX = "docpilot:"


def _digest(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


class RetrievalCache:
    BACKOFF_SECONDS = 5.0

    def __init__(self, client: aioredis.Redis, *, namespace: str, search_ttl: int, embedding_ttl: int):
        self._redis = client
        # The DB schema is part of the namespace: eval schemas share Redis with the app.
        self._ns = f"{PREFIX}{namespace}:"
        self._search_ttl = search_ttl
        self._embedding_ttl = embedding_ttl
        self._down_until = 0.0
        self.hits = self.misses = 0

    def _up(self) -> bool:
        return time.monotonic() >= self._down_until

    def _failed(self, e: Exception) -> None:
        self._down_until = time.monotonic() + self.BACKOFF_SECONDS
        log.warning("retrieval cache: redis unavailable: %s", type(e).__name__)

    # --- query embeddings ------------------------------------------------------------------

    def _embedding_key(self, tenant_id: int, model: str, text: str) -> str:
        return f"{self._ns}emb:{tenant_id}:{_digest(model, text)}"

    async def get_embedding(self, tenant_id: int, model: str, text: str) -> np.ndarray | None:
        if not self._embedding_ttl or not self._up():
            return None
        try:
            raw = await self._redis.get(self._embedding_key(tenant_id, model, text))
        except (RedisError, OSError) as e:
            self._failed(e)
            return None
        return None if raw is None else np.frombuffer(raw, dtype=np.float32).copy()

    async def set_embedding(self, tenant_id: int, model: str, text: str, vector: np.ndarray) -> None:
        if not self._embedding_ttl or not self._up():
            return
        try:
            await self._redis.set(self._embedding_key(tenant_id, model, text),
                                  np.asarray(vector, dtype=np.float32).tobytes(), ex=self._embedding_ttl)
        except (RedisError, OSError) as e:
            self._failed(e)

    # --- search results --------------------------------------------------------------------

    def _generation_key(self, tenant_id: int) -> str:
        return f"{self._ns}gen:{tenant_id}"

    async def _generation(self, tenant_id: int) -> int:
        return int(await self._redis.get(self._generation_key(tenant_id)) or 0)

    async def get_search(self, tenant_id: int, params: dict) -> tuple[list[dict] | None, str | None]:
        """(cached results or None, the key to store fresh results under, or None to skip)."""
        if not self._search_ttl or not self._up():
            return None, None
        try:
            key = f"{self._ns}search:{tenant_id}:{await self._generation(tenant_id)}:{_digest(params)}"
            raw = await self._redis.get(key)
        except (RedisError, OSError) as e:
            self._failed(e)
            return None, None
        if raw is None:
            self.misses += 1
            return None, key
        self.hits += 1
        return json.loads(raw), key

    async def set_search(self, key: str, results: list[dict]) -> None:
        try:
            await self._redis.set(key, json.dumps(results), ex=self._search_ttl)
        except (RedisError, OSError) as e:
            self._failed(e)

    async def invalidate(self, tenant_id: int) -> None:
        """Call after a write to the tenant's library has committed."""
        try:
            await self._redis.incr(self._generation_key(tenant_id))
        except (RedisError, OSError) as e:
            # Stale results can now be served until they expire: bounded by the TTL.
            log.warning("retrieval cache: could not invalidate tenant %s: %s", tenant_id, type(e).__name__)
