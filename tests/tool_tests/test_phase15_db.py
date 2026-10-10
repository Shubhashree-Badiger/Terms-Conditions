"""
Phase 15 verification: SQLite persistence of analysis results.
Creates tables, saves a sample analysis, and queries it back.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from app.models.database import init_db, SessionLocal
from app.services import db_service


def run():
    print("=" * 66)
    print("PHASE 15: SQLite Database Persistence")
    print("=" * 66)

    init_db()
    print("Tables created.")

    sample = {
        "document_id": "EDU_DOC_DBCHK",
        "filename": "sample.pdf",
        "total_pages": 3,
        "total_clauses": 2,
        "status": "analyzed",
        "processing_seconds": 2.1,
        "clauses": [
            {
                "clause_id": "C001", "clause_number": "3.2",
                "page_start": 1, "page_end": 1,
                "clause_text": "Students must maintain 75% attendance.",
                "summary": "Need 75% attendance.",
                "important_phrases": [
                    {"text": "75% attendance", "score": 0.9, "level": "HIGH",
                     "category": "attendance", "risk_type": "academic",
                     "risk_level": "MEDIUM",
                     "locations": [{"page": 1, "bbox": [10, 20, 30, 40]}]},
                ],
            },
            {
                "clause_id": "C002", "clause_number": "4.1",
                "page_start": 2, "page_end": 2,
                "clause_text": "The fee is non-refundable.",
                "summary": "Fee non-refundable.",
                "important_phrases": [
                    {"text": "non-refundable", "score": 0.8, "level": "HIGH",
                     "category": "fees", "risk_type": "financial",
                     "risk_level": "MEDIUM",
                     "locations": [{"page": 2, "bbox": [15, 25, 35, 45]}]},
                ],
            },
        ],
    }

    with SessionLocal() as db:
        doc = db_service.save_analysis(db, sample, annotated_pdf_path="/tmp/a.pdf")
        print(f"Saved document pk={doc.id}, clauses={len(doc.clauses)}")

        meta = db_service.document_metadata(db, "EDU_DOC_DBCHK")
        print("Metadata:", meta)

        n = db_service.count_phrases(db, "EDU_DOC_DBCHK")
        print("Total phrases stored:", n)

        # verify phrase locations round-trip
        clause = doc.clauses[0]
        ph = clause.phrases[0]
        print("Phrase:", ph.phrase_text, "| level:", ph.importance_level,
              "| locations:", ph.locations)

        # idempotency: save again, counts should stay the same
        db_service.save_analysis(db, sample)
        n2 = db_service.count_phrases(db, "EDU_DOC_DBCHK")
        print("Phrases after re-save (idempotent):", n2)

        # cleanup this test row
        db.delete(db_service.get_document(db, "EDU_DOC_DBCHK"))
        db.commit()
        print("Cleaned up test row.")

    print("\n" + "=" * 66)
    print("[PHASE 15 CHECK COMPLETE]")


if __name__ == "__main__":
    run()
