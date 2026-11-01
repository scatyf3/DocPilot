"""Logging setup and a request id on every request.

Logs carry metadata only (paths, ids, token counts), never questions or document text, which
may be sensitive and don't belong in a log aggregator.
"""

import json
import logging
import re
import time
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime

from docpilot.config import Settings

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

log = logging.getLogger("docpilot.access")
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_STANDARD = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {"time": datetime.fromtimestamp(record.created, UTC).isoformat(),
                 "level": record.levelname, "logger": record.name, "message": record.getMessage(),
                 "request_id": request_id.get()}
        entry |= {k: v for k, v in record.__dict__.items() if k not in _STANDARD}
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(settings: Settings) -> None:
    handler = logging.StreamHandler()
    if settings.log_json:
        handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=settings.log_level.upper(), handlers=[handler], force=True,
                        format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per upstream call is noise


class RequestContextMiddleware:
    """Assign a request id, echo it in the response, and log one line per request.

    Plain ASGI rather than BaseHTTPMiddleware, which would buffer around the SSE chat stream.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        incoming = headers.get(b"x-request-id", b"").decode("latin-1")
        rid = incoming if _SAFE_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        token = request_id.set(rid)
        started = time.perf_counter()
        status = 500  # stays 500 if the app raises before starting a response

        async def send_with_id(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = [*message.get("headers", []), (b"x-request-id", rid.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            ms = round((time.perf_counter() - started) * 1000)
            level = logging.DEBUG if scope["path"] in ("/healthz", "/readyz") else logging.INFO
            log.log(level, "%s %s %s %dms", scope["method"], scope["path"], status, ms,
                    extra={"method": scope["method"], "path": scope["path"], "status": status,
                           "duration_ms": ms})
            request_id.reset(token)
