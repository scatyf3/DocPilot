"""Who is calling (a Principal), and how often they may: FastAPI dependencies.

Callers are one of:
- a user, with a JWT access token from /auth/login: acts on their own tenant;
- a service, with a shared API key: acts on the default tenant (1), the single-tenant setup
  this app started as, which the web app's server-side proxy and the evals still use;
- anonymous, only in development with no API keys configured: also the default tenant.

Rate limits are token buckets. With REDIS_URL they live in Redis, updated by one Lua script,
so every instance draws from the same bucket: N instances no longer allow N times the limit.
Without Redis, or while it is down, each instance falls back to its own in-memory buckets.
"""

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, Request
from redis.exceptions import RedisError

from docpilot.auth import AuthError, constant_time_in, decode_access_token

log = logging.getLogger(__name__)
UNAUTHORIZED = {"WWW-Authenticate": "Bearer"}
DEFAULT_TENANT_ID = 1


@dataclass(frozen=True)
class Principal:
    kind: Literal["user", "api_key", "anonymous"]
    tenant_id: int
    user_id: int | None = None


def _bearer(request: Request) -> str | None:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer":
        return None
    return token.strip() or None


def _looks_like_jwt(token: str) -> bool:
    return token.count(".") == 2


async def get_principal(request: Request) -> Principal:
    """Authenticate the request. FastAPI caches it, so routes can depend on it again for free."""
    settings = request.app.state.settings
    bearer = _bearer(request)
    if bearer and _looks_like_jwt(bearer):
        if not settings.jwt_secret_list:
            raise HTTPException(401, "user tokens are not enabled", headers=UNAUTHORIZED)
        try:
            claims = decode_access_token(settings, bearer)
        except AuthError as e:
            raise HTTPException(401, str(e), headers=UNAUTHORIZED) from e
        principal = Principal("user", claims.tenant_id, claims.user_id)
    else:
        presented = bearer or request.headers.get("x-api-key")
        keys = settings.api_key_list
        if presented is not None and keys and constant_time_in(presented, keys):
            principal = Principal("api_key", DEFAULT_TENANT_ID)
        elif not keys and settings.environment == "development":
            principal = Principal("anonymous", DEFAULT_TENANT_ID)
        else:
            raise HTTPException(401, "missing or invalid credentials", headers=UNAUTHORIZED)
    request.state.principal = principal
    return principal


CurrentPrincipal = Annotated[Principal, Depends(get_principal)]


# --- Rate limits -------------------------------------------------------------------------------


class RateLimiter:
    """In-process token buckets: a client may burst up to one minute's allowance, then refills."""

    MAX_CLIENTS = 10_000

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last update)

    def take(self, key: str, per_minute: int) -> float:
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

    async def acquire(self, key: str, per_minute: int) -> float:
        return self.take(key, per_minute)


# The same bucket as RateLimiter.take, run atomically inside Redis. Time comes from the Redis
# server, not the caller, so instances with skewed clocks still agree. The key expires once a
# full bucket would have refilled, so idle clients cost no memory.
_TOKEN_BUCKET = """
local capacity = tonumber(ARGV[1])
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local rate = capacity / 60000
local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1]) or capacity
local ts = tonumber(state[2]) or now
tokens = math.min(capacity, tokens + math.max(0, now - ts) * rate)
local wait = 0
if tokens < 1 then
  wait = math.ceil((1 - tokens) / rate)
else
  tokens = tokens - 1
end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', now)
redis.call('PEXPIRE', KEYS[1], 60000)
return wait
"""


class RedisRateLimiter:
    """Buckets shared by every instance. Falls back to local buckets while Redis is down.

    Failing open (local limits) rather than closed (429 for everyone) is deliberate: the limits
    exist to stop runaway clients and cost spikes, and a Redis outage shouldn't take the API
    down with it. After a failure Redis is left alone for a few seconds, so an outage costs
    one timeout every few seconds rather than one per request.
    """

    BACKOFF_SECONDS = 5.0

    def __init__(self, client: aioredis.Redis, fallback: RateLimiter | None = None,
                 prefix: str = "docpilot:rl:"):
        self._redis = client
        self._script = client.register_script(_TOKEN_BUCKET)
        self._fallback = fallback or RateLimiter()
        self._prefix = prefix
        self._down_until = 0.0

    async def acquire(self, key: str, per_minute: int) -> float:
        if time.monotonic() >= self._down_until:
            try:
                wait_ms = await self._script(keys=[self._prefix + key], args=[per_minute])
                return int(wait_ms) / 1000
            except (RedisError, OSError) as e:
                self._down_until = time.monotonic() + self.BACKOFF_SECONDS
                log.warning("rate limiter: redis unavailable, using local buckets: %s", type(e).__name__)
        return self._fallback.take(key, per_minute)


def client_ip(request: Request) -> str:
    header = request.app.state.settings.client_ip_header
    forwarded = request.headers.get(header) if header else None
    return forwarded or (request.client.host if request.client else "unknown")


def _client_id(request: Request) -> str:
    principal: Principal | None = getattr(request.state, "principal", None)
    if principal is not None and principal.user_id is not None:
        return f"user:{principal.user_id}"  # a user's limit follows them across networks
    return f"ip:{client_ip(request)}"


async def enforce(request: Request, bucket: str, key: str | None = None) -> None:
    """Take a token from `bucket` for this client (or for `key`), or raise 429."""
    settings = request.app.state.settings
    if not settings.rate_limit_enabled:
        return
    per_minute = getattr(settings, f"rate_limit_{bucket}_per_minute")
    wait = await request.app.state.rate_limiter.acquire(f"{bucket}:{key or _client_id(request)}", per_minute)
    if wait:
        raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(math.ceil(wait))})


def rate_limit(bucket: str):
    """Dependency limiting one route family, using the rate_limit_<bucket>_per_minute setting.

    Routers authenticate first (router dependencies run before route ones), so the limit can
    follow the user rather than their IP.
    """

    async def check(request: Request) -> None:
        await enforce(request, bucket)

    return check
