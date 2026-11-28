"""
Audit Log Middleware — automatically logs all write operations.
Append-only log: actor, action, entity, IP, timestamp.
"""
import logging

from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("Studvisor.audit")

WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

# Routes to skip audit (high-frequency read-like POSTs)
SKIP_AUDIT = {"/chat", "/chat/stream", "/v2/ai/student/chat", "/v2/ai/student/chat/stream"}


class AuditLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)

        if request.method in WRITE_METHODS and request.url.path not in SKIP_AUDIT:
            # Extract actor from JWT if present
            auth = request.headers.get("Authorization", "")
            payload = None
            if auth.startswith("Bearer "):
                from backend.core.security import decode_token
                payload = decode_token(auth.split(" ", 1)[1])
            payload = payload or {}
            role = payload.get("role", "anonymous")
            username = payload.get("sub", "anonymous")
            actor_id = payload.get("entity_id", 0)
            ip = request.client.host if request.client else None

            # The anonymous wall must not be traceable to a student through the audit trail.
            if request.url.path.startswith("/anon/"):
                role, username, actor_id, ip = "anonymous", "anonymous", 0, None

            logger.info(
                "[AUDIT] %s %s -> %s | actor=%s:%s(id=%s) | ip=%s",
                request.method, request.url.path, response.status_code, role, username, actor_id, ip,
            )
            await run_in_threadpool(
                _persist, role, username, actor_id, request.method, request.url.path, response.status_code, ip
            )

        return response


def _persist(role, username, actor_id, method, path, status, ip):
    from backend.app.database import SessionLocal
    from backend.app.models import AuditLog

    db = SessionLocal()
    try:
        db.add(AuditLog(
            actor_role=role,
            actor_username=username,
            actor_id=actor_id,
            action=f"{method} {path}",
            resource_type=path.strip("/").split("/")[0] or "root",
            new_value=f"Status: {status}",
            ip_address=ip,
        ))
        db.commit()
    except Exception:
        db.rollback()
        logger.error("Failed to persist audit log", exc_info=True)
    finally:
        db.close()
