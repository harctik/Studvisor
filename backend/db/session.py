# db session — re-export for backward compatibility
from backend.app.database import Base, SessionLocal, engine, get_db

__all__ = ["Base", "engine", "SessionLocal", "get_db"]
