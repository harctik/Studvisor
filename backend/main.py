"""
Studvisor v2.0 — Main Application Factory
"""
import logging
import os

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger("Studvisor.app")

# ── Rate Limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

app = FastAPI(
    title="Studvisor — Unified Campus Intelligence Platform",
    description="AI-powered Student ERP with multi-role auth, predictive analytics, gamification, and 80+ endpoints",
    version="2.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Middleware ────────────────────────────────────────────────────────────────
from backend.app.middleware import RequestLoggingMiddleware, SecurityHeadersMiddleware
from backend.middleware.audit_log import AuditLogMiddleware
from backend.middleware.role_guard import RoleGuardMiddleware

app.add_middleware(AuditLogMiddleware)
app.add_middleware(RoleGuardMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── Database ─────────────────────────────────────────────────────────────────
from backend.app.database import Base, engine

Base.metadata.create_all(bind=engine)

# ── Route Registration ───────────────────────────────────────────────────────
from backend.api.routes import (
    achievement_router,
    admin_analytics_router,
    admin_router,
    ai_engine_router,
    analytics_router,
    announcement_router,
    anon_chat_router,
    assignment_router,
    attendance_router,
    auth_router,
    bunk_alerts_router,
    calendar_router,
    campus_router,
    chat_router,
    chat_stream_router,
    complaint_router,
    courses_router,
    documents_router,
    event_router,
    exam_router,
    export_router,
    faculty_portal_router,
    faculty_router,
    fees_router,
    gpa_router,
    internships_router,
    leaderboard_router,
    leave_router,
    library_router,
    lost_found_router,
    marks_router,
    merit_router,
    notes_router,
    notification_router,
    peer_matching_router,
    placement_router,
    poll_router,
    reports_router,
    search_router,
    student_router,
    syllabus_router,
    timetable_router,
    websocket_router,
)

# Register all routers
ALL_ROUTERS = [
    auth_router, student_router, attendance_router, marks_router,
    leave_router, gpa_router, chat_router, fees_router,
    admin_router, faculty_portal_router, reports_router, library_router,
    calendar_router, documents_router, bunk_alerts_router, anon_chat_router,
    complaint_router, notification_router, merit_router, timetable_router,
    analytics_router, assignment_router, exam_router, syllabus_router,
    notes_router, poll_router, event_router, announcement_router,
    faculty_router, lost_found_router, achievement_router, leaderboard_router,
    placement_router, courses_router, internships_router, campus_router,
    websocket_router, chat_stream_router, export_router, admin_analytics_router,
    search_router, peer_matching_router,
    ai_engine_router,
]

for router in ALL_ROUTERS:
    app.include_router(router)

# ── Exception Handlers ───────────────────────────────────────────────────────
@app.exception_handler(FastAPIHTTPException)
async def http_exception_handler(request: Request, exc: FastAPIHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"success": False, "message": exc.detail, "data": None, "error": exc.detail})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"success": False, "message": "Internal server error", "data": None, "error": "Internal server error"})

# ── Root & Health ────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"success": True, "message": "Studvisor v2.0 — Unified Campus Intelligence Platform", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"success": True, "status": "ok", "version": "2.0.0"}

