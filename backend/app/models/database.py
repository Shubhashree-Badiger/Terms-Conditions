"""
Database setup (Phase 15)
=========================
SQLite engine + session management via SQLAlchemy.

Usage:
    from app.models.database import get_db, init_db, SessionLocal
    init_db()                      # create tables once at startup
    with SessionLocal() as s: ...  # or use the get_db() dependency in routes
"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app import config

# check_same_thread=False lets the connection be used across FastAPI's threads
engine = create_engine(
    config.DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

Base = declarative_base()


def init_db():
    """Create all tables if they do not exist."""
    # import models so they are registered on Base before create_all
    from app.models import db_models  # noqa: F401
    config.ensure_directories()
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
