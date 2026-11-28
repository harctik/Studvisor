"""Test configuration — seed test DB before running tests."""
import os
import pytest

# Force SQLite for tests
os.environ["DATABASE_URL"] = "sqlite:///./test_studvisor.db"
os.environ["SECRET_KEY"] = "test-secret-key-for-ci"

from backend.app.database import Base, engine


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    """Create all tables before tests, drop after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    engine.dispose()  # Windows cannot delete the SQLite file while pooled connections hold it
    if os.path.exists("test_studvisor.db"):
        os.remove("test_studvisor.db")
