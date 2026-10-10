"""
Phase 12 tests: complete analysis pipeline (ml/analysis_pipeline).
Uses small in-memory PDFs and models disabled for speed.
Run from the backend directory with the venv pytest.
"""
import os
import sys
import tempfile

import pymupdf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.analysis_pipeline import AnalysisPipeline

# Build one pipeline for all tests (models off -> fast)
_PIPELINE = AnalysisPipeline(use_semantic=False, use_t5=False)


def _make_pdf(path, pages_lines):
    doc = pymupdf.open()
    for lines in pages_lines:
        page = doc.new_page()
        y = 72
        for ln in lines:
            page.insert_text((72, y), ln)
            y += 22
    doc.save(path)
    doc.close()


def test_analyze_document_basic():
    with tempfile.TemporaryDirectory() as d:
        pdf = os.path.join(d, "policy.pdf")
        out = os.path.join(d, "annotated.pdf")
        _make_pdf(pdf, [[
            "1. Attendance",
            "Students failing to maintain 75% attendance shall be debarred from the exam.",
            "2. Fees",
            "The admission fee is non-refundable under any circumstances.",
        ]])
        result = _PIPELINE.analyze_document(pdf, document_id="EDU_DOC_T1",
                                            annotated_out=out)
        assert result["status"] == "analyzed"
        assert result["document_id"] == "EDU_DOC_T1"
        assert result["total_clauses"] >= 2
        assert result["annotated_pdf"] == out
        assert os.path.exists(out)
        # every clause has the required keys
        for c in result["clauses"]:
            for k in ("clause_id", "clause_number", "clause_text",
                      "summary", "important_phrases", "page_start", "page_end"):
                assert k in c


def test_important_phrases_have_coordinates():
    with tempfile.TemporaryDirectory() as d:
        pdf = os.path.join(d, "p.pdf")
        _make_pdf(pdf, [[
            "1. Discipline",
            "Any student indulging in ragging shall face suspension or expulsion.",
        ]])
        result = _PIPELINE.analyze_document(pdf)
        phrases = [p for c in result["clauses"] for p in c["important_phrases"]]
        assert len(phrases) >= 1
        mapped = [p for p in phrases if p["mapping_status"].startswith("mapped")]
        assert len(mapped) >= 1
        assert mapped[0]["locations"][0]["page"] == 1
        assert len(mapped[0]["locations"][0]["bbox"]) == 4


def test_stats_present():
    with tempfile.TemporaryDirectory() as d:
        pdf = os.path.join(d, "p.pdf")
        _make_pdf(pdf, [["1. Fees", "The fee is non-refundable and a penalty applies."]])
        result = _PIPELINE.analyze_document(pdf)
        for k in ("total_phrases", "mapped_phrases", "failed_phrases",
                  "summary_failures", "clause_errors"):
            assert k in result["stats"]


def test_scanned_pdf_status():
    with tempfile.TemporaryDirectory() as d:
        pdf = os.path.join(d, "blank.pdf")
        # blank pages -> no text -> flagged scanned
        _make_pdf(pdf, [[], [], []])
        result = _PIPELINE.analyze_document(pdf)
        assert result["status"] == "scanned"
        assert "OCR" in result["message"]


def test_missing_file_status():
    result = _PIPELINE.analyze_document("nope_missing_12345.pdf")
    assert result["status"] == "error"
    assert "not found" in result["error"].lower()


def test_original_pdf_untouched_by_pipeline():
    with tempfile.TemporaryDirectory() as d:
        pdf = os.path.join(d, "p.pdf")
        out = os.path.join(d, "annotated.pdf")
        _make_pdf(pdf, [["1. Rule", "Fees are non-refundable."]])
        before = os.path.getsize(pdf)
        _PIPELINE.analyze_document(pdf, annotated_out=out)
        assert os.path.getsize(pdf) == before
        orig = pymupdf.open(pdf)
        assert len(list(orig.load_page(0).annots() or [])) == 0
        orig.close()
