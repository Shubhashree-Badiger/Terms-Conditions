"""
Database persistence service (Phase 15)
=======================================
Saves a formatted analysis result into the SQLite tables and provides
queries to check existence / fetch metadata without re-running NLP.

The JSON store (results_store) remains the source for full-result serving;
the database adds structured, queryable persistence per the spec.
"""
from __future__ import annotations

import datetime
from typing import Optional, Dict

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.db_models import Document, Clause, Phrase


def get_document(db: Session, document_id: str) -> Optional[Document]:
    return db.execute(
        select(Document).where(Document.document_id == document_id)
    ).scalar_one_or_none()


def document_exists(db: Session, document_id: str) -> bool:
    return get_document(db, document_id) is not None


def register_upload(db: Session, document_id: str, filename: str,
                    total_pages: int, original_pdf_path: str) -> Document:
    """Create (or update) a Document row at upload time."""
    doc = get_document(db, document_id)
    if doc is None:
        doc = Document(document_id=document_id)
        db.add(doc)
    doc.filename = filename
    doc.total_pages = total_pages
    doc.status = "uploaded"
    doc.original_pdf_path = original_pdf_path
    doc.upload_date = datetime.datetime.now(datetime.timezone.utc)
    db.commit()
    db.refresh(doc)
    return doc


def save_analysis(db: Session, result: Dict, annotated_pdf_path: str = None) -> Document:
    """Persist a formatted analysis result (analyzed status) into the DB.

    Existing clauses/phrases for the document are replaced so re-analysis is
    idempotent.
    """
    document_id = result["document_id"]
    doc = get_document(db, document_id)
    if doc is None:
        doc = Document(document_id=document_id)
        db.add(doc)

    doc.filename = result.get("filename")
    doc.total_pages = result.get("total_pages", 0)
    doc.total_clauses = result.get("total_clauses", 0)
    doc.status = result.get("status", "analyzed")
    doc.processing_seconds = result.get("processing_seconds", 0.0)
    if annotated_pdf_path:
        doc.annotated_pdf_path = annotated_pdf_path
    db.flush()  # ensure doc.id is available

    # clear previous clauses (cascade removes phrases)
    for existing in list(doc.clauses):
        db.delete(existing)
    db.flush()

    for c in result.get("clauses", []):
        clause = Clause(
            clause_id=c.get("clause_id"),
            document_pk=doc.id,
            clause_number=c.get("clause_number", ""),
            clause_text=c.get("clause_text", ""),
            page_start=c.get("page_start"),
            page_end=c.get("page_end"),
            summary=c.get("summary", ""),
        )
        db.add(clause)
        db.flush()  # get clause.id
        for p in c.get("important_phrases", []):
            phrase = Phrase(
                clause_pk=clause.id,
                phrase_text=p.get("text", ""),
                importance_score=p.get("score", 0.0),
                importance_level=p.get("level", "LOW"),
                category=p.get("category"),
                risk_type=p.get("risk_type"),
                risk_level=p.get("risk_level"),
                mapping_status=(p.get("locations") and "mapped") or "failed",
            )
            phrase.locations = p.get("locations", [])
            db.add(phrase)

    db.commit()
    db.refresh(doc)
    return doc


def document_metadata(db: Session, document_id: str) -> Optional[Dict]:
    doc = get_document(db, document_id)
    if doc is None:
        return None
    return {
        "document_id": doc.document_id,
        "filename": doc.filename,
        "status": doc.status,
        "analyzed": doc.status == "analyzed",
        "total_pages": doc.total_pages,
        "total_clauses": doc.total_clauses,
        "processing_seconds": doc.processing_seconds,
        "upload_date": doc.upload_date.isoformat() if doc.upload_date else None,
        "annotated_pdf_path": doc.annotated_pdf_path,
    }


def count_phrases(db: Session, document_id: str) -> int:
    doc = get_document(db, document_id)
    if not doc:
        return 0
    return sum(len(c.phrases) for c in doc.clauses)
