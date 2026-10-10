"""
Phase 15 tests: SQLite persistence (app.models + app.services.db_service).
Uses the configured DB; test rows are cleaned up afterward.
Run from the backend directory with the venv pytest.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.models.database import init_db, SessionLocal
from app.services import db_service

_TEST_ID = "EDU_DOC_PYTEST_DB"


def _sample(document_id=_TEST_ID):
    return {
        "document_id": document_id,
        "filename": "t.pdf",
        "total_pages": 2,
        "total_clauses": 1,
        "status": "analyzed",
        "processing_seconds": 1.0,
        "clauses": [{
            "clause_id": "C001", "clause_number": "1.1",
            "page_start": 1, "page_end": 1,
            "clause_text": "Fees are non-refundable.",
            "summary": "Fees non-refundable.",
            "important_phrases": [{
                "text": "non-refundable", "score": 0.85, "level": "HIGH",
                "category": "fees", "risk_type": "financial", "risk_level": "MEDIUM",
                "locations": [{"page": 1, "bbox": [1, 2, 3, 4]}],
            }],
        }],
    }


def setup_module(module):
    init_db()


def teardown_function(function):
    with SessionLocal() as db:
        doc = db_service.get_document(db, _TEST_ID)
        if doc:
            db.delete(doc)
            db.commit()


def test_register_upload():
    with SessionLocal() as db:
        doc = db_service.register_upload(db, _TEST_ID, "t.pdf", 5, "/tmp/t.pdf")
        assert doc.document_id == _TEST_ID
        assert doc.status == "uploaded"
        assert doc.total_pages == 5


def test_save_analysis_persists_clauses_and_phrases():
    with SessionLocal() as db:
        doc = db_service.save_analysis(db, _sample())
        assert doc.status == "analyzed"
        assert len(doc.clauses) == 1
        assert len(doc.clauses[0].phrases) == 1
        ph = doc.clauses[0].phrases[0]
        assert ph.importance_level == "HIGH"
        assert ph.locations == [{"page": 1, "bbox": [1, 2, 3, 4]}]


def test_metadata_query():
    with SessionLocal() as db:
        db_service.save_analysis(db, _sample())
        meta = db_service.document_metadata(db, _TEST_ID)
        assert meta["analyzed"] is True
        assert meta["total_clauses"] == 1


def test_count_phrases():
    with SessionLocal() as db:
        db_service.save_analysis(db, _sample())
        assert db_service.count_phrases(db, _TEST_ID) == 1


def test_reanalysis_is_idempotent():
    with SessionLocal() as db:
        db_service.save_analysis(db, _sample())
        db_service.save_analysis(db, _sample())  # save again
        assert db_service.count_phrases(db, _TEST_ID) == 1  # not doubled


def test_missing_document_metadata_none():
    with SessionLocal() as db:
        assert db_service.document_metadata(db, "NOPE_XYZ_123") is None
        assert db_service.document_exists(db, "NOPE_XYZ_123") is False
