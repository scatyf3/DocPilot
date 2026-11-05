"""Tokens and passwords (no database), then the account lifecycle against Postgres."""

import asyncio
import base64
import json
import os
import time

import httpx
import jwt
import pytest

from docpilot import auth
from docpilot.app import check_settings
from docpilot.config import Settings

SECRET = "s" * 40
OLD_SECRET = "o" * 40


def cheap(**overrides) -> Settings:
    return Settings(**{"jwt_secrets": SECRET, "argon2_time_cost": 1, "argon2_memory_kib": 1024,
                       "argon2_parallelism": 1, **overrides})


# --- access tokens -------------------------------------------------------------------------


def test_access_token_round_trip():
    s = cheap()
    claims = auth.decode_access_token(s, auth.issue_access_token(s, user_id=7, tenant_id=3))
    assert (claims.user_id, claims.tenant_id) == (7, 3)


def test_secrets_rotate_like_api_keys():
    old = auth.issue_access_token(cheap(jwt_secrets=OLD_SECRET), 1, 1)
    # Deploy step 1: new secret first (signs), old second (still verifies).
    assert auth.decode_access_token(cheap(jwt_secrets=f"{SECRET},{OLD_SECRET}"), old).user_id == 1
    new = auth.issue_access_token(cheap(jwt_secrets=f"{SECRET},{OLD_SECRET}"), 2, 1)
    assert auth.decode_access_token(cheap(jwt_secrets=SECRET), new).user_id == 2
    # Step 2: the old secret is dropped, and tokens it signed stop working.
    with pytest.raises(auth.AuthError):
        auth.decode_access_token(cheap(jwt_secrets=SECRET), old)


def _forge(header: dict, claims: dict, signature: bytes = b"") -> str:
    def b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()
    return ".".join([b64(json.dumps(header).encode()), b64(json.dumps(claims).encode()), b64(signature)])


@pytest.mark.parametrize("name", ["alg-none", "wrong-secret", "tampered-tenant", "expired", "wrong-audience",
                                  "no-tenant", "garbage"])
def test_bad_access_tokens_are_rejected(name):
    s = cheap()
    now = int(time.time())
    good = {"iss": auth.ISSUER, "aud": auth.AUDIENCE, "sub": "1", "tid": 2, "iat": now, "exp": now + 60}
    kid = {"kid": auth._kid(SECRET)}
    tokens = {
        "alg-none": _forge({"alg": "none", "typ": "JWT", **kid}, good),
        "wrong-secret": jwt.encode(good, "x" * 40, algorithm="HS256", headers=kid),
        "expired": jwt.encode({**good, "iat": now - 120, "exp": now - 60}, SECRET, algorithm="HS256",
                              headers=kid),
        "wrong-audience": jwt.encode({**good, "aud": "other"}, SECRET, algorithm="HS256", headers=kid),
        "no-tenant": jwt.encode({k: v for k, v in good.items() if k != "tid"}, SECRET, algorithm="HS256",
                                headers=kid),
        "garbage": "a.b.c",
    }
    real = auth.issue_access_token(s, 1, 2)
    head, payload, sig = real.split(".")
    claims = json.loads(base64.urlsafe_b64decode(payload + "=="))
    tampered = base64.urlsafe_b64encode(json.dumps({**claims, "tid": 1}).encode()).rstrip(b"=").decode()
    tokens["tampered-tenant"] = f"{head}.{tampered}.{sig}"
    with pytest.raises(auth.AuthError):
        auth.decode_access_token(s, tokens[name])


# --- passwords -----------------------------------------------------------------------------


async def test_passwords_hash_verify_and_upgrade():
    weak, strong = auth.Passwords(cheap()), auth.Passwords(cheap(argon2_time_cost=2))
    stored = await weak.hash("hunter2 hunter2")
    assert stored.startswith("$argon2id$")
    assert await weak.verify(stored, "hunter2 hunter2") == (True, None)
    assert await weak.verify(stored, "wrong") == (False, None)
    assert await weak.verify(None, "anything") == (False, None)  # unknown user: still hashes
    ok, upgraded = await strong.verify(stored, "hunter2 hunter2")
    assert ok and upgraded and "t=2" in upgraded


async def test_hashing_concurrency_is_bounded(monkeypatch):
    passwords = auth.Passwords(cheap(password_hash_concurrency=2))
    running = peak = 0

    def slow_hash(password):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        time.sleep(0.05)
        running -= 1
        return "h"

    monkeypatch.setattr(passwords, "_hasher", type("SlowHasher", (), {"hash": staticmethod(slow_hash)})())
    await asyncio.gather(*[passwords.hash("p") for _ in range(8)])
    assert peak == 2


# --- configuration -------------------------------------------------------------------------


def test_production_needs_credentials_and_long_secrets():
    with pytest.raises(RuntimeError, match="API_KEYS or JWT_SECRETS"):
        check_settings(Settings(environment="production"))
    with pytest.raises(RuntimeError, match="32 characters"):
        check_settings(Settings(environment="production", jwt_secrets="short"))
    check_settings(Settings(environment="production", jwt_secrets=SECRET))
    check_settings(Settings(environment="production", api_keys="k"))


# --- account lifecycle (Postgres) ------------------------------------------------------------

needs_db = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="needs DATABASE_URL")


@pytest.fixture
async def client():
    from docpilot.app import create_app
    from docpilot.db import init_schema

    s = cheap(db_schema="docpilot_test_auth", embedding_provider="fake", llm_provider="fake",
              rate_limit_enabled=False)
    await init_schema(s, reset=True)
    app = create_app(s)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            yield c


async def register(client, email="ada@example.com", password="correct horse battery"):
    return await client.post("/auth/register", json={"email": email, "password": password})


async def refresh(client, tokens: dict):
    return await client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})


@needs_db
async def test_register_login_and_me(client):
    resp = await register(client)
    assert resp.status_code == 201
    assert resp.json()["token_type"] == "bearer" and resp.json()["expires_in"] == 900
    assert (await register(client, "ADA@example.com")).status_code == 409  # emails ignore case
    assert (await register(client, "x@example.com", "short")).status_code == 422

    login = await client.post("/auth/login",
                              json={"email": "Ada@Example.com", "password": "correct horse battery"})
    assert login.status_code == 200
    me = await client.get("/auth/me", headers={"Authorization": f"Bearer {login.json()['access_token']}"})
    assert me.json()["email"] == "ada@example.com" and "password_hash" not in me.json()

    wrong = await client.post("/auth/login", json={"email": "ada@example.com", "password": "nope"})
    unknown = await client.post("/auth/login", json={"email": "who@example.com", "password": "nope"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()  # no hint which emails have accounts


@needs_db
async def test_refresh_rotates_and_reuse_revokes_the_family(client):
    first = (await register(client)).json()
    second = await client.post("/auth/refresh", json={"refresh_token": first["refresh_token"]})
    assert second.status_code == 200
    second = second.json()
    assert second["refresh_token"] != first["refresh_token"]
    third = (await client.post("/auth/refresh", json={"refresh_token": second["refresh_token"]})).json()

    # Someone replays the first token (stolen earlier): refused, and the whole login is ended,
    # including the newest token, which the thief may also hold by now.
    replay = await client.post("/auth/refresh", json={"refresh_token": first["refresh_token"]})
    assert replay.status_code == 401
    assert (await refresh(client, third)).status_code == 401

    # Other logins of the same user are separate families and survive.
    other = (await client.post("/auth/login", json={"email": "ada@example.com",
                                                    "password": "correct horse battery"})).json()
    assert (await refresh(client, other)).status_code == 200


@needs_db
async def test_concurrent_refreshes_with_one_token_cannot_both_win(client):
    tokens = (await register(client)).json()
    results = await asyncio.gather(*[
        client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]}) for _ in range(5)])
    winners = [r for r in results if r.status_code == 200]
    assert len(winners) == 1 and sum(r.status_code == 401 for r in results) == 4
    # The losers looked like a replay, so the winner's new token was revoked too.
    again = await client.post("/auth/refresh", json={"refresh_token": winners[0].json()["refresh_token"]})
    assert again.status_code == 401


@needs_db
async def test_logout_ends_the_login(client):
    tokens = (await register(client)).json()
    refreshed = (await client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})).json()
    logout = await client.post("/auth/logout", json={"refresh_token": refreshed["refresh_token"]})
    assert logout.status_code == 204
    assert (await refresh(client, refreshed)).status_code == 401
    assert (await client.post("/auth/logout", json={"refresh_token": "unknown"})).status_code == 204


@needs_db
async def test_expired_refresh_tokens_are_refused_and_cleaned_up(client):
    tokens = (await register(client)).json()
    pool = client._transport.app.state.pool
    async with pool.connection() as conn:
        await conn.execute("UPDATE refresh_tokens SET expires_at = now() - interval '30 days'")
    resp = await client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert resp.status_code == 401 and resp.json()["detail"] == "refresh token expired"
    assert await auth.delete_expired_refresh_tokens(pool) == 1


@needs_db
async def test_jwt_and_api_key_principals(client):
    app = client._transport.app
    app.state.settings = app.state.settings.model_copy(update={"api_keys": "svc"})
    access = (await register(client)).json()["access_token"]
    assert (await client.get("/documents")).status_code == 401  # keys configured: no anonymous
    assert (await client.get("/documents", headers={"Authorization": f"Bearer {access}"})).status_code == 200
    assert (await client.get("/documents", headers={"Authorization": "Bearer svc"})).status_code == 200
    me = (await client.get("/auth/me", headers={"X-API-Key": "svc"})).json()
    assert me == {"kind": "api_key", "tenant_id": 1}
    bad = await client.get("/documents", headers={"Authorization": f"Bearer {access[:-2]}xx"})
    assert bad.status_code == 401 and bad.headers["www-authenticate"] == "Bearer"


@needs_db
async def test_signup_can_be_disabled(client):
    app = client._transport.app
    app.state.settings = app.state.settings.model_copy(update={"allow_signup": False})
    assert (await register(client)).status_code == 403
