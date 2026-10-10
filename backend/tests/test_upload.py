"""
Phase 3 tests: POST /api/upload
Run from the backend directory with the venv pytest.
"""
import io
import os

import pymupdf
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _make_real_pdf_bytes(text="Students must maintain 75% attendance.") -> bytes:
    """Create a tiny valid one-page PDF in memory."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_upload_valid_pdf():
    pdf_bytes = _make_real_pdf_bytes()
    files = {"file": ("sample_policy.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    r = client.post("/api/upload", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "uploaded"
    assert body["document_id"].startswith("EDU_DOC_")
    assert body["total_pages"] == 1
    assert body["filename"] == "sample_policy.pdf"


def test_reject_non_pdf_extension():
    files = {"file": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")}
    r = client.post("/api/upload", files=files)
    assert r.status_code == 400
    assert "extension" in r.json()["detail"].lower()


def test_reject_fake_pdf():
    # .pdf extension + pdf mime but bytes are not a real PDF
    files = {"file": ("fake.pdf", io.BytesIO(b"not really a pdf"), "application/pdf")}
    r = client.post("/api/upload", files=files)
    assert r.status_code == 400
    assert "pdf" in r.json()["detail"].lower()


def test_reject_empty_file():
    files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}
    r = client.post("/api/upload", files=files)
    assert r.status_code == 400
