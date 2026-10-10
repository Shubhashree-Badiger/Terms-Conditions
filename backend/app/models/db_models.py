"""
SQLAlchemy ORM models (Phase 15)
================================
Three tables persist the analysis:

  documents  - one row per uploaded/analyzed PDF
  clauses    - one row per segmented clause (FK -> documents)
  phrases    - one row per important phrase (FK -> clauses)

bbox coordinates are stored as JSON text (list of {page, bbox}) so a phrase
that maps to several locations keeps them all.
"""
from __future__ import annotations

import datetime
import json

from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, ForeignKey
)
from sqlalchemy.orm import relationship

from app.models.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(64), unique=True, index=True, nullable=False)
    filename = Column(String(512))
    upload_date = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    total_pages = Column(Integer, default=0)
    total_clauses = Column(Integer, default=0)
    status = Column(String(32), default="uploaded")   # uploaded/analyzed/scanned/error
    processing_seconds = Column(Float, default=0.0)
    original_pdf_path = Column(String(1024))
    annotated_pdf_path = Column(String(1024))

    clauses = relationship("Clause", back_populates="document",
                           cascade="all, delete-orphan")


class Clause(Base):
    __tablename__ = "clauses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    clause_id = Column(String(32), index=True, nullable=False)     # e.g. C001
    document_pk = Column(Integer, ForeignKey("documents.id"), index=True)
    clause_number = Column(String(32), default="")
    clause_text = Column(Text, default="")
    page_start = Column(Integer)
    page_end = Column(Integer)
    summary = Column(Text, default="")

    document = relationship("Document", back_populates="clauses")
    phrases = relationship("Phrase", back_populates="clause",
                           cascade="all, delete-orphan")


class Phrase(Base):
    __tablename__ = "phrases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    clause_pk = Column(Integer, ForeignKey("clauses.id"), index=True)
    phrase_text = Column(Text, default="")
    importance_score = Column(Float, default=0.0)
    importance_level = Column(String(16), default="LOW")   # HIGH/MEDIUM/LOW
    category = Column(String(64))
    risk_type = Column(String(64))
    risk_level = Column(String(16))
    mapping_status = Column(String(32), default="failed")
    # all locations for this phrase, JSON: [{"page": int, "bbox": [..]}, ...]
    locations_json = Column(Text, default="[]")

    clause = relationship("Clause", back_populates="phrases")

    # convenience helpers
    @property
    def locations(self):
        try:
            return json.loads(self.locations_json or "[]")
        except Exception:
            return []

    @locations.setter
    def locations(self, value):
        self.locations_json = json.dumps(value or [])
