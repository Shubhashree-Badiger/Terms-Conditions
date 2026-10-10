"""
Phase 4 tests: PDF extraction with coordinates (ml/preprocessing/pdf_extractor).
Run from the backend directory with the venv pytest.
"""
import os
import sys

import pymupdf

# Make project root importable so `ml` resolves when running from backend/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.preprocessing.pdf_extractor import (
    extract_pdf, extract_page_words, extract_page_text,
    reconstruct_text, is_probably_scanned,
)


def _make_pdf_bytes(lines):
    """Create an in-memory PDF with the given text lines."""
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line)
        y += 20
    data = doc.tobytes()
    doc.close()
    return data


def test_extract_from_bytes_basic():
    data = _make_pdf_bytes([
        "Students must maintain 75% attendance.",
        "Fees once paid are non-refundable.",
    ])
    doc = extract_pdf(data, document_id="EDU_DOC_TEST", file_name="t.pdf")
    assert doc.error is None
    assert doc.total_pages == 1
    assert doc.total_words > 0
    assert "attendance" in doc.full_text.lower()


def test_word_records_have_bbox():
    data = _make_pdf_bytes(["Attendance requirement is 75 percent."])
    doc = extract_pdf(data)
    words = doc.pages[0].words
    assert len(words) >= 5
    for w in words:
        assert set(w.keys()) == {"text", "bbox", "block", "line", "word_index"}
        assert len(w["bbox"]) == 4
        x0, y0, x1, y1 = w["bbox"]
        assert x1 > x0 and y1 > y0  # valid box


def test_reconstruct_text_nonempty():
    data = _make_pdf_bytes(["Hostel gate closes at 9 PM.", "Visitors are not permitted."])
    doc = extract_pdf(data)
    recon = reconstruct_text(doc.pages[0].words)
    assert "hostel" in recon.lower()
    assert "visitors" in recon.lower()


def test_missing_file_returns_error():
    doc = extract_pdf("does_not_exist_12345.pdf")
    assert doc.error is not None
    assert doc.total_pages == 0


def test_scanned_detection_true_for_empty_wordset():
    # A PDF with no text -> should be flagged scanned
    doc = pymupdf.open()
    doc.new_page()  # blank page, no text
    data = doc.tobytes()
    doc.close()
    extracted = extract_pdf(data)
    assert is_probably_scanned(extracted) is True
