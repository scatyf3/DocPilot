"""User accounts: argon2id passwords, short-lived JWT access tokens, rotating refresh tokens.

Access tokens are stateless, so a stolen one works until it expires (ACCESS_TOKEN_TTL_SECONDS,
15 minutes by default) and logout cannot recall it. Refresh tokens are the long-lived
credential, so they are stored (hashed), single-use, and grouped in families: replaying a used
one revokes every token descended from the same login. Whichever of thief and owner refreshes
second gets logged out, and the theft is logged.
"""

import asyncio
import hashlib
import hmac
import logging
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from psycopg_pool import AsyncConnectionPool

from docpilot.config import Settings

log = logging.getLogger(__name__)

ISSUER = "docpilot"
AUDIENCE = "docpilot-api"
ALGORITHM = "HS256"


class AuthError(Exception):
    """The credentials were wrong. The message is safe to show the caller."""


# --- Passwords ---------------------------------------------------------------------------------


class Passwords:
    """argon2id hashing, off the event loop and bounded in concurrency.

    One hash takes ~50 ms of CPU and argon2_memory_kib of RAM (64 MiB by default). On the
    event loop it would stall every other request; unbounded, 50 concurrent logins would
    allocate 3 GiB. The semaphore turns a burst into a short queue instead.
    """

    def __init__(self, settings: Settings):
        self._hasher = PasswordHasher(time_cost=settings.argon2_time_cost,
                                      memory_cost=settings.argon2_memory_kib,
                                      parallelism=settings.argon2_parallelism)
        self._slots = asyncio.Semaphore(settings.password_hash_concurrency)
        self._dummy: str | None = None

    async def hash(self, password: str) -> str:
        async with self._slots:
            return await asyncio.to_thread(self._hasher.hash, password)

    async def verify(self, stored_hash: str | None, password: str) -> tuple[bool, str | None]:
        """(matches, new hash if the stored one used weaker parameters than today's).

        An unknown user is checked against a dummy hash, so the response time doesn't tell
        an attacker which emails have accounts.
        """
        if stored_hash is None:
            if self._dummy is None:
                self._dummy = await self.hash(secrets.token_urlsafe(16))
            stored_hash = self._dummy
            known = False
        else:
            known = True
        async with self._slots:
            try:
                await asyncio.to_thread(self._hasher.verify, stored_hash, password)
            except (VerificationError, InvalidHashError):
                return False, None
        if not known:
            return False, None
        if self._hasher.check_needs_rehash(stored_hash):
            return True, await self.hash(password)
        return True, None


# --- Access tokens -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AccessClaims:
    user_id: int
    tenant_id: int


def _kid(secret: str) -> str:
    # Names the key in the token header without revealing anything about it.
    return hashlib.sha256(secret.encode()).hexdigest()[:12]


def issue_access_token(settings: Settings, user_id: int, tenant_id: int) -> str:
    secret = settings.jwt_secret_list[0]
    now = int(time.time())
    claims = {"iss": ISSUER, "aud": AUDIENCE, "sub": str(user_id), "tid": tenant_id,
              "iat": now, "exp": now + settings.access_token_ttl_seconds, "jti": uuid.uuid4().hex}
    return jwt.encode(claims, secret, algorithm=ALGORITHM, headers={"kid": _kid(secret)})


def decode_access_token(settings: Settings, token: str) -> AccessClaims:
    """Verify signature, expiry, issuer and audience. Raises AuthError on anything off."""
    secrets_by_kid = {_kid(s): s for s in settings.jwt_secret_list}
    try:
        kid = jwt.get_unverified_header(token).get("kid")
        secret = secrets_by_kid.get(kid)
        if secret is None:
            raise AuthError("invalid token")
        # A fixed algorithm list: the token's own "alg" header never picks how it is checked.
        claims = jwt.decode(token, secret, algorithms=[ALGORITHM], audience=AUDIENCE, issuer=ISSUER,
                            leeway=10, options={"require": ["exp", "iat", "sub", "tid"]})
        return AccessClaims(user_id=int(claims["sub"]), tenant_id=int(claims["tid"]))
    except jwt.ExpiredSignatureError as e:
        raise AuthError("token expired") from e
    except (jwt.InvalidTokenError, ValueError, TypeError) as e:
        raise AuthError("invalid token") from e


# --- Refresh tokens ----------------------------------------------------------------------------


def _hash_refresh(token: str) -> bytes:
    # The token is 256 random bits, so a fast hash is enough: there is nothing to brute-force.
    return hashlib.sha256(token.encode()).digest()


@dataclass
class TokenPair:
    access_token: str
    refresh_token: str
    expires_in: int
    refresh_expires_in: int

    def to_dict(self) -> dict:
        return {"access_token": self.access_token, "token_type": "bearer",
                "expires_in": self.expires_in, "refresh_token": self.refresh_token,
                "refresh_expires_in": self.refresh_expires_in}


async def _insert_refresh(conn, settings: Settings, user_id: int, family_id: uuid.UUID) -> str:
    token = secrets.token_urlsafe(32)
    await conn.execute(
        """INSERT INTO refresh_tokens (user_id, family_id, token_hash, expires_at)
           VALUES (%s, %s, %s, %s)""",
        (user_id, family_id, _hash_refresh(token),
         datetime.now(UTC) + timedelta(days=settings.refresh_token_ttl_days)),
    )
    return token


def _pair(settings: Settings, user_id: int, tenant_id: int, refresh: str) -> TokenPair:
    return TokenPair(issue_access_token(settings, user_id, tenant_id), refresh,
                     settings.access_token_ttl_seconds, settings.refresh_token_ttl_days * 86_400)


async def start_session(pool: AsyncConnectionPool, settings: Settings,
                        user_id: int, tenant_id: int) -> TokenPair:
    """A new token family, after a successful login or signup."""
    async with pool.connection() as conn:
        refresh = await _insert_refresh(conn, settings, user_id, uuid.uuid4())
    return _pair(settings, user_id, tenant_id, refresh)


async def rotate(pool: AsyncConnectionPool, settings: Settings, token: str) -> TokenPair:
    """Spend a refresh token on a new pair. A second use revokes the whole family.

    The row lock serialises two refreshes racing with the same token: the second waits, then
    sees the token spent. Two tabs refreshing at the same instant therefore log the user out;
    that is the price of noticing a replayed token at all.
    """
    async with pool.connection() as conn, conn.transaction():
        row = await (await conn.execute(
            """SELECT r.id, r.user_id, r.family_id, r.expires_at, r.used_at, r.revoked_at, u.tenant_id
               FROM refresh_tokens r JOIN users u ON u.id = r.user_id
               WHERE r.token_hash = %s FOR UPDATE OF r""",
            (_hash_refresh(token),),
        )).fetchone()
        if row is None:
            raise AuthError("invalid refresh token")
        spent = row["used_at"] is not None or row["revoked_at"] is not None
        if spent:
            await conn.execute(
                "UPDATE refresh_tokens SET revoked_at = now() WHERE family_id = %s AND revoked_at IS NULL",
                (row["family_id"],),
            )
        elif row["expires_at"] <= datetime.now(UTC):
            raise AuthError("refresh token expired")
        else:
            await conn.execute("UPDATE refresh_tokens SET used_at = now() WHERE id = %s", (row["id"],))
            refresh = await _insert_refresh(conn, settings, row["user_id"], row["family_id"])
    if spent:  # raised after the commit, so the revocation sticks
        if row["used_at"] is not None:  # a revoked one is just a stale logout; a used one was copied
            log.warning("refresh token reused; revoked its family",
                        extra={"user_id": row["user_id"], "family_id": str(row["family_id"])})
        raise AuthError("invalid refresh token")
    return _pair(settings, row["user_id"], row["tenant_id"], refresh)


async def revoke_family(pool: AsyncConnectionPool, token: str) -> None:
    """Logout: end this login on every device it was refreshed on. Unknown tokens are ignored."""
    async with pool.connection() as conn:
        await conn.execute(
            """UPDATE refresh_tokens SET revoked_at = now()
               WHERE revoked_at IS NULL AND family_id =
                     (SELECT family_id FROM refresh_tokens WHERE token_hash = %s)""",
            (_hash_refresh(token),),
        )


async def delete_expired_refresh_tokens(pool: AsyncConnectionPool, older_than_days: int = 7) -> int:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "DELETE FROM refresh_tokens WHERE expires_at < now() - make_interval(days => %s)",
            (older_than_days,),
        )
        return cur.rowcount


# --- Users -------------------------------------------------------------------------------------


async def create_user(pool: AsyncConnectionPool, passwords: Passwords, *, email: str,
                      password: str, workspace: str) -> tuple[int, int]:
    """A new user in a new tenant they own. Returns (user_id, tenant_id)."""
    password_hash = await passwords.hash(password)
    async with pool.connection() as conn, conn.transaction():
        taken = await (await conn.execute(
            "SELECT 1 FROM users WHERE lower(email) = lower(%s)", (email,))).fetchone()
        if taken:
            raise AuthError("an account with this email already exists")
        tenant = await (await conn.execute(
            "INSERT INTO tenants (name) VALUES (%s) RETURNING id", (workspace,))).fetchone()
        user = await (await conn.execute(
            "INSERT INTO users (tenant_id, email, password_hash) VALUES (%s, %s, %s) RETURNING id",
            (tenant["id"], email, password_hash))).fetchone()
    return user["id"], tenant["id"]


async def authenticate(pool: AsyncConnectionPool, passwords: Passwords, email: str,
                       password: str) -> tuple[int, int]:
    """(user_id, tenant_id) for a correct email and password, else AuthError."""
    async with pool.connection() as conn:
        user = await (await conn.execute(
            "SELECT id, tenant_id, password_hash FROM users WHERE lower(email) = lower(%s)", (email,),
        )).fetchone()
    ok, new_hash = await passwords.verify(user["password_hash"] if user else None, password)
    if not ok:
        raise AuthError("wrong email or password")  # the same for both, by design
    if new_hash:
        async with pool.connection() as conn:
            await conn.execute("UPDATE users SET password_hash = %s WHERE id = %s", (new_hash, user["id"]))
    return user["id"], user["tenant_id"]


async def get_user(pool: AsyncConnectionPool, user_id: int) -> dict | None:
    async with pool.connection() as conn:
        return await (await conn.execute(
            """SELECT u.id, u.email, u.tenant_id, t.name AS workspace, u.created_at
               FROM users u JOIN tenants t ON t.id = u.tenant_id WHERE u.id = %s""",
            (user_id,),
        )).fetchone()


def constant_time_in(presented: str, keys: list[str]) -> bool:
    """Compare against every key in constant time, so timing reveals neither key nor position."""
    return any([hmac.compare_digest(presented.encode(), k.encode()) for k in keys])
