"""Shared-key API authentication and per-client rate limits, as FastAPI dependencies.

A key protects the API as a whole; it says nothing about which user is calling. Every caller
sees the same library, so don't expose this to mutually untrusting users without adding that.
The rate limiter lives in process memory, so with N instances the effective limit is N times
the configured one. It stops runaway clients and cost spikes, not a determined attacker.
"""

import hmac
import math
import time
from collections.abc import Callable

from fastapi import HTTPException, Request

UNAUTHORIZED = {"WWW-Authenticate": "Bearer"}


def _presented_key(request: Request) -> str | None:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and token:
        return token.strip()
    return request.headers.get("x-api-key")


async def require_api_key(request: Request) -> None:
    keys = request.app.state.settings.api_key_list
    if not keys:
        return  # development without keys; create_app refuses this in production
    presented = _presented_key(request)
    # Compare against every key, in constant time, so timing reveals neither key nor position.
    if presented is None or not any([hmac.compare_digest(presented.encode(), k.encode()) for k in keys]):
        raise HTTPException(401, "missing or invalid API key", headers=UNAUTHORIZED)


class RateLimiter:
    """Token buckets: a client may burst up to one minute's allowance, then refills steadily."""

    MAX_CLIENTS = 10_000

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last update)

    def acquire(self, key: str, per_minute: int) -> float:
        """Take one token. Returns 0 on success, otherwise the seconds to wait."""
        now = self._clock()
        rate = per_minute / 60
        tokens, updated = self._buckets.get(key, (float(per_minute), now))
        tokens = min(per_minute, tokens + (now - updated) * rate)
        if tokens < 1:
            self._buckets[key] = (tokens, now)
            return (1 - tokens) / rate
        if key not in self._buckets and len(self._buckets) >= self.MAX_CLIENTS:
            # Forget the oldest client: bounded memory matters more than a perfect count.
            del self._buckets[next(iter(self._buckets))]
        self._buckets[key] = (tokens - 1, now)
        return 0.0


def _client_id(request: Request) -> str:
    header = request.app.state.settings.client_ip_header
    forwarded = request.headers.get(header) if header else None
    return forwarded or (request.client.host if request.client else "unknown")


def rate_limit(bucket: str):
    """Dependency limiting one route family, using the rate_limit_<bucket>_per_minute setting."""

    async def check(request: Request) -> None:
        settings = request.app.state.settings
        if not settings.rate_limit_enabled:
            return
        per_minute = getattr(settings, f"rate_limit_{bucket}_per_minute")
        wait = request.app.state.rate_limiter.acquire(f"{bucket}:{_client_id(request)}", per_minute)
        if wait:
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(math.ceil(wait))})

    return check
