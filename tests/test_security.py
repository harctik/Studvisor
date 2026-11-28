"""Tests for access control on payments, the websocket, formerly public routes and error output."""
import uuid

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.app.database import SessionLocal
from backend.app.models import Admin, AnonPost, Attendance, AuditLog, Note, Payment, Student, StudentFee
from backend.core.security import create_access_token, create_refresh_token
from backend.main import app

client = TestClient(app)


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


def _claims(role, entity_id):
    return {"sub": f"{role}-{entity_id}", "role": role, "entity_id": entity_id}


def _add(row):
    with SessionLocal() as db:
        db.add(row)
        db.commit()
        return row.id


def _student():
    name = f"sec_{uuid.uuid4().hex[:10]}"
    student_id = _add(Student(username=name, email=f"{name}@example.test", hashed_password="x", full_name="Security Test"))
    return student_id, _bearer(create_access_token(_claims("student", student_id)))


def _admin():
    name = f"adm_{uuid.uuid4().hex[:10]}"
    admin_id = _add(Admin(username=name, email=f"{name}@example.test", hashed_password="x", full_name="Test Admin"))
    return _bearer(create_access_token(_claims("admin", admin_id)))


def _fee_state(fee_id):
    with SessionLocal() as db:
        fee = db.get(StudentFee, fee_id)
        return fee.amount_paid, fee.status, db.query(Payment).filter(Payment.student_fee_id == fee_id).count()


# ─── Fees ────────────────────────────────────────────────────────────────────

def test_student_claim_does_not_change_the_fee():
    student_id, student = _student()
    fee_id = _add(StudentFee(student_id=student_id, amount_due=1000, amount_paid=0))

    r = client.post("/fees/pay", json={"student_fee_id": fee_id, "amount": 1000}, headers=student)

    assert r.status_code == 202
    assert r.json()["status"] == "Pending"
    assert _fee_state(fee_id) == (0, "Pending", 0)


def test_claims_cannot_exceed_the_open_balance():
    student_id, student = _student()
    fee_id = _add(StudentFee(student_id=student_id, amount_due=1000, amount_paid=0))
    pay = lambda amount: client.post("/fees/pay", json={"student_fee_id": fee_id, "amount": amount}, headers=student)  # noqa: E731

    assert pay(-5).status_code == 422
    assert pay(1001).status_code == 400
    assert pay(600).status_code == 202
    assert pay(600).status_code == 400


def test_student_cannot_claim_against_another_students_fee():
    owner_id, _ = _student()
    _, intruder = _student()
    fee_id = _add(StudentFee(student_id=owner_id, amount_due=1000, amount_paid=0))

    r = client.post("/fees/pay", json={"student_fee_id": fee_id, "amount": 10}, headers=intruder)

    assert r.status_code == 404


def test_only_an_admin_can_approve_and_only_once():
    student_id, student = _student()
    fee_id = _add(StudentFee(student_id=student_id, amount_due=1000, amount_paid=0))
    claim_id = client.post("/fees/pay", json={"student_fee_id": fee_id, "amount": 1000}, headers=student).json()["claim_id"]
    approve = f"/fees/admin/claims/{claim_id}/approve"

    assert client.post(approve, headers=student).status_code == 403
    assert client.get("/fees/admin/claims", headers=student).status_code == 403
    faculty = _bearer(create_access_token(_claims("faculty", 1)))
    assert client.post(approve, headers=faculty).status_code == 403
    assert _fee_state(fee_id) == (0, "Pending", 0)

    admin = _admin()
    assert claim_id in [c["id"] for c in client.get("/fees/admin/claims", headers=admin).json()["claims"]]
    assert client.post(approve, headers=admin).status_code == 200
    assert client.post(approve, headers=admin).status_code == 409
    assert _fee_state(fee_id) == (1000, "Paid", 1)
    assert client.get("/fees/claims", headers=student).json()["claims"][0]["status"] == "Approved"


def test_rejected_claim_leaves_the_fee_unpaid():
    student_id, student = _student()
    fee_id = _add(StudentFee(student_id=student_id, amount_due=500, amount_paid=0))
    claim_id = client.post("/fees/pay", json={"student_fee_id": fee_id, "amount": 500}, headers=student).json()["claim_id"]
    admin = _admin()

    assert client.post(f"/fees/admin/claims/{claim_id}/reject", headers=admin).status_code == 200
    assert client.post(f"/fees/admin/claims/{claim_id}/approve", headers=admin).status_code == 409
    assert client.post("/fees/admin/claims/999999/approve", headers=admin).status_code == 404
    assert _fee_state(fee_id) == (0, "Pending", 0)


# ─── Websocket ───────────────────────────────────────────────────────────────

def test_websocket_accepts_the_matching_student():
    student_id, _ = _student()
    with client.websocket_connect(f"/ws/{student_id}") as ws:
        ws.send_json({"type": "auth", "token": create_access_token(_claims("student", student_id))})
        assert ws.receive_json() == {"type": "auth_ok"}
        ws.send_text("not json")
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}


@pytest.mark.parametrize(
    "first_frame",
    [
        {"type": "ping"},
        {"type": "auth"},
        {"type": "auth", "token": "garbage"},
        {"type": "auth", "token": create_access_token(_claims("student", 2))},
        {"type": "auth", "token": create_access_token(_claims("admin", 1))},
        {"type": "auth", "token": create_refresh_token(_claims("student", 1))},
        ["auth"],
    ],
)
def test_websocket_rejects_everyone_else(first_frame):
    with client.websocket_connect("/ws/1") as ws:
        ws.send_json(first_frame)
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
    assert closed.value.code == 1008


# ─── Formerly public routes ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/anon/posts"),
        ("get", "/notes/"),
        ("post", "/notes/1/rate"),
        ("post", "/chat-stream/"),
        ("get", "/search/?q=an"),
        ("get", "/leaderboard/"),
        ("get", "/faculty/directory"),
    ],
)
def test_route_requires_a_valid_token(method, path):
    assert getattr(client, method)(path).status_code in (401, 403)
    assert getattr(client, method)(path, headers=_bearer("garbage")).status_code == 401


def test_signed_in_student_can_use_those_routes():
    _, student = _student()
    note_id = _add(Note(title="Unit 1", file_url="/n.pdf", helpful_count=0, not_helpful_count=0))

    for path in ["/anon/posts", "/notes/", "/search/?q=an", "/leaderboard/", "/faculty/directory"]:
        assert client.get(path, headers=student).status_code == 200, path
    assert client.post(f"/notes/{note_id}/rate", headers=student).status_code == 200
    assert client.post("/chat-stream/?query=hi", headers=student).status_code == 200


def test_anonymous_posts_never_identify_the_author(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.services.anonymity_service.SALT_DIR", tmp_path)
    monkeypatch.setattr("backend.services.anonymity_service.SALT_FILE", tmp_path / "salt.bin")
    _, student = _student()
    text = f"wall post {uuid.uuid4().hex}"

    assert client.post("/anon/posts", json={"content": text}, headers=student).status_code == 200
    post = next(p for p in client.get("/anon/posts", headers=student).json()["posts"] if p["content"] == text)

    assert set(post) == {"id", "content", "category", "reactions", "date"}
    with SessionLocal() as db:
        assert db.query(AnonPost).filter(AnonPost.content == text).one().session_hash
        rows = db.query(AuditLog).filter(AuditLog.action == "POST /anon/posts").all()
        assert rows
        assert all((row.actor_id, row.actor_username, row.ip_address) == (0, "anonymous", None) for row in rows)


def test_writes_are_saved_to_the_audit_log():
    student_id, student = _student()
    client.post("/fees/pay", json={"student_fee_id": 999999, "amount": 1}, headers=student)

    with SessionLocal() as db:
        assert db.query(AuditLog).filter(AuditLog.actor_id == student_id, AuditLog.action == "POST /fees/pay").count() == 1


# ─── Error output ────────────────────────────────────────────────────────────

def test_server_errors_do_not_leak_details(caplog):
    secret = "database password is hunter2"

    def boom():
        raise RuntimeError(secret)

    app.add_api_route("/_test_boom", boom)
    try:
        r = TestClient(app, raise_server_exceptions=False).get("/_test_boom")
    finally:
        app.router.routes.pop()

    assert r.status_code == 500
    assert secret not in r.text
    assert r.json()["error"] == "Internal server error"
    assert secret in caplog.text


def test_report_html_escapes_stored_text():
    name = f"xss_{uuid.uuid4().hex[:10]}"
    student_id = _add(Student(username=name, email=f"{name}@example.test", hashed_password="x",
                              full_name="<script>alert(1)</script>"))

    r = client.get(f"/reports/bonafide/{student_id}", headers=_bearer(create_access_token(_claims("student", student_id))))

    assert r.status_code == 200
    assert "<script>" not in r.text
    assert "&lt;script&gt;" in r.text


# ─── Emotion detection ───────────────────────────────────────────────────────

@pytest.mark.parametrize("message", ["I want to kill myself", "I feel hopeless", "I can’t go on"])
def test_a_single_distress_phrase_is_enough(message):
    from backend.app.chatbot import detect_emotion

    assert detect_emotion(message) == "distressed"


@pytest.mark.parametrize("message", ["Can you send it to me?", "Hello, I studied for the exam"])
def test_ordinary_words_are_not_flagged(message):
    from backend.services.sentiment_service import sentiment_service

    assert sentiment_service.analyze(message)["flagged_keywords"] == []


# --- Admin dashboard ---------------------------------------------------------

def test_admin_dashboard_counts_students_below_75_percent_attendance():
    admin = _admin()

    def at_risk():
        r = client.get("/admin/dashboard/v2", headers=admin)
        assert r.status_code == 200
        return r.json()["academic"]["at_risk_students"]

    before = at_risk()
    # 80% present is above the line, 50% is below it.
    for statuses in ("PPPPA", "PA"):
        student_id, _ = _student()
        for hour, status in enumerate(statuses):
            _add(Attendance(student_id=student_id, subject_id=1, date="2026-01-05", hour=hour, status=status))

    assert at_risk() == before + 1
