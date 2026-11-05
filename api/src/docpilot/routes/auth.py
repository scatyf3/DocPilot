"""Accounts: sign up, log in, refresh, log out, who am I.

Sign-up creates a user and a tenant (workspace) they own. These routes take no API key: they
are how a user gets credentials in the first place. Login is rate limited per IP and per email,
so neither one address hammering many accounts nor many addresses hammering one gets far.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from docpilot import auth
from docpilot.security import CurrentPrincipal, client_ip, enforce

router = APIRouter(prefix="/auth", tags=["auth"])

EMAIL = r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{1,63}$"


class Credentials(BaseModel):
    email: str = Field(pattern=EMAIL, max_length=320)
    # argon2 accepts any length; the cap keeps a megabyte "password" from costing CPU.
    password: str = Field(min_length=1, max_length=256)


class SignUp(Credentials):
    password: str = Field(min_length=10, max_length=256)
    workspace: str | None = Field(None, min_length=1, max_length=200)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=1, max_length=200)


def _require_users(request: Request) -> None:
    if not request.app.state.settings.jwt_secret_list:
        raise HTTPException(404, "user accounts are not enabled (set JWT_SECRETS)")


async def _limit(request: Request, email: str | None = None) -> None:
    await enforce(request, "auth", f"ip:{client_ip(request)}")
    if email:
        await enforce(request, "auth", f"email:{email.lower()}")


@router.post("/register", status_code=201)
async def register(request: Request, body: SignUp) -> dict:
    _require_users(request)
    s = request.app.state
    if not s.settings.allow_signup:
        raise HTTPException(403, "sign-up is disabled")
    await _limit(request)
    try:
        user_id, tenant_id = await auth.create_user(
            s.pool, s.passwords, email=body.email, password=body.password,
            workspace=body.workspace or f"{body.email.split('@')[0]}'s workspace")
    except auth.AuthError as e:
        raise HTTPException(409, str(e)) from e
    return (await auth.start_session(s.pool, s.settings, user_id, tenant_id)).to_dict()


@router.post("/login")
async def login(request: Request, body: Credentials) -> dict:
    _require_users(request)
    s = request.app.state
    await _limit(request, body.email)
    try:
        user_id, tenant_id = await auth.authenticate(s.pool, s.passwords, body.email, body.password)
    except auth.AuthError as e:
        raise HTTPException(401, str(e)) from e
    return (await auth.start_session(s.pool, s.settings, user_id, tenant_id)).to_dict()


@router.post("/refresh")
async def refresh(request: Request, body: RefreshRequest) -> dict:
    _require_users(request)
    s = request.app.state
    await _limit(request)
    try:
        return (await auth.rotate(s.pool, s.settings, body.refresh_token)).to_dict()
    except auth.AuthError as e:
        raise HTTPException(401, str(e)) from e


@router.post("/logout", status_code=204)
async def logout(request: Request, body: RefreshRequest) -> None:
    _require_users(request)
    await auth.revoke_family(request.app.state.pool, body.refresh_token)


@router.get("/me")
async def me(request: Request, principal: CurrentPrincipal) -> dict:
    if principal.user_id is None:
        return {"kind": principal.kind, "tenant_id": principal.tenant_id}
    user = await auth.get_user(request.app.state.pool, principal.user_id)
    if user is None:  # deleted since the token was issued
        raise HTTPException(401, "user no longer exists")
    return {"kind": "user", **user}
