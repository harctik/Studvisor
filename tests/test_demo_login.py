"""The test-only admin/admin sign-in must stay shut unless asked for, on this computer."""
import pytest
from fastapi.testclient import TestClient

from backend.app.database import SessionLocal
from backend.app.models import Student
from backend.core.config import get_settings
from backend.core.demo_login import DEMO_STUDENT, demo_login_allowed
from backend.main import app


@pytest.fixture(scope="module", autouse=True)
def demo_student():
    db = SessionLocal()
    if not db.query(Student).filter(Student.username == DEMO_STUDENT).first():
        db.add(Student(username=DEMO_STUDENT, email="demo.student@example.com",
                       hashed_password="unusable", full_name="Aarav Sharma"))
        db.commit()
    db.close()


def sign_in(host="127.0.0.1", headers=None):
    client = TestClient(app, client=(host, 50000))
    return client.post("/login", json={"username": "admin", "password": "admin"}, headers=headers or {})


def test_off_by_default(monkeypatch):
    monkeypatch.setattr(get_settings(), "DEMO_LOGIN", False)
    assert sign_in().status_code == 401


def test_opens_the_demo_student_on_this_computer(monkeypatch):
    monkeypatch.setattr(get_settings(), "DEMO_LOGIN", True)
    response = sign_in()
    assert response.status_code == 200
    assert response.json()["role"] == "student"


@pytest.mark.parametrize("host,headers", [
    ("203.0.113.9", {}),
    ("127.0.0.1", {"X-Forwarded-For": "127.0.0.1"}),
    ("127.0.0.1", {"X-Real-IP": "203.0.113.9"}),
])
def test_refused_from_elsewhere(monkeypatch, host, headers):
    monkeypatch.setattr(get_settings(), "DEMO_LOGIN", True)
    assert sign_in(host, headers).status_code == 401


def test_refused_on_a_hosting_service(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    assert not demo_login_allowed(True, "127.0.0.1", {})
