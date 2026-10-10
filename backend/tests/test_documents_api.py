"""
Phase 14 tests: document analysis + retrieval endpoints.
Uses a tiny in-memory PDF; the pipeline runs with models off (fast).
Run from the backend directory with the venv pytest.
"""
import io
import os

import pymupdf
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _pdf_bytes():
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72
    for ln in [
        "1. Attendance",
        "Students failing to maintain 75% attendance shall be debarred from the exam.",
        "2. Fees",
        "The admission fee is non-refundable and a penalty of Rs. 500 applies.",
    ]:
        page.insert_text((72, y), ln)
        y += 22
    data = doc.tobytes()
    doc.close()
    return data


def _upload():
    files = {"file": ("api_policy.pdf", io.BytesIO(_pdf_bytes()), "application/pdf")}
    r = client.post("/api/upload", files=files)
    assert r.status_code == 200, r.text
    return r.json()["document_id"]


def test_analyze_and_retrieve_flow():
    doc_id = _upload()

    # analyze
    r = client.post(f"/api/analyze/{doc_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "analyzed"
    assert body["total_clauses"] >= 2
    assert body["annotated_pdf"] == f"/api/documents/{doc_id}/annotated"

    # metadata
    r = client.get(f"/api/documents/{doc_id}")
    assert r.status_code == 200
    assert r.json()["analyzed"] is True

    # full results
    r = client.get(f"/api/documents/{doc_id}/results")
    assert r.status_code == 200
    assert "clauses" in r.json()

    # clauses only
    r = client.get(f"/api/documents/{doc_id}/clauses")
    assert r.status_code == 200
    assert r.json()["total_clauses"] >= 2

    # annotated PDF
    r = client.get(f"/api/documents/{doc_id}/annotated")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"


def test_analyze_caches_result():
    doc_id = _upload()
    r1 = client.post(f"/api/analyze/{doc_id}")
    assert r1.status_code == 200
    # second call should return cached (same content)
    r2 = client.post(f"/api/analyze/{doc_id}")
    assert r2.status_code == 200
    assert r1.json()["document_id"] == r2.json()["document_id"]
    assert r1.json()["total_clauses"] == r2.json()["total_clauses"]


def test_analyze_unknown_document_404():
    r = client.post("/api/analyze/EDU_DOC_UNKNOWN_123")
    assert r.status_code == 404


def test_results_before_analysis_404():
    doc_id = _upload()
    r = client.get(f"/api/documents/{doc_id}/results")
    assert r.status_code == 404  # not analyzed yet


def test_unknown_document_metadata_404():
    r = client.get("/api/documents/DOES_NOT_EXIST_999")
    assert r.status_code == 404


def test_annotated_before_analysis_404():
    doc_id = _upload()
    r = client.get(f"/api/documents/{doc_id}/annotated")
    assert r.status_code == 404


def _cleanup():
    """Remove test artifacts created in uploads/outputs during these tests."""
    from app import config
    for d in (config.UPLOADS_DIR, config.BACKEND_OUTPUTS_DIR, config.ANNOTATED_PDFS_DIR):
        if d.exists():
            for p in d.iterdir():
                if p.is_file() and p.name.startswith("EDU_DOC_") and "api_policy" in p.name:
                    p.unlink()


def teardown_module(module):
    _cleanup()


# --- Phase 23: async analysis + progress polling ---
import time as _time


def test_analyze_async_and_progress():
    doc_id = _upload()

    # start background analysis
    r = client.post(f"/api/analyze-async/{doc_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] in ("started", "cached")

    # poll progress until done (bounded)
    done = False
    for _ in range(60):
        pr = client.get(f"/api/documents/{doc_id}/progress")
        assert pr.status_code == 200
        p = pr.json()
        if p.get("done"):
            done = True
            assert p["stage"] == 6
            break
        if p.get("error"):
            break
        _time.sleep(0.5)
    assert done, "analysis did not complete in time"

    # results now available
    rr = client.get(f"/api/documents/{doc_id}/results")
    assert rr.status_code == 200
    assert rr.json()["status"] == "analyzed"


def test_progress_unknown_document_404():
    r = client.get("/api/documents/UNKNOWN_PROGRESS_999/progress")
    assert r.status_code == 404


def teardown_module_async(module=None):
    _cleanup()
