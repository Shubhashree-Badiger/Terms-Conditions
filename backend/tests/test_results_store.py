"""
Phase 13 tests: results formatting + persistence (app.services.results_store).
Run from the backend directory with the venv pytest.
"""
import json
import os

from app.services.results_store import (
    format_result, save_result, load_result, result_json_path,
    _annotated_api_path,
)


def _fake_pipeline_result(document_id="EDU_DOC_T13"):
    return {
        "document_id": document_id,
        "filename": "policy.pdf",
        "total_pages": 3,
        "total_clauses": 1,
        "status": "analyzed",
        "processing_seconds": 1.2,
        "stats": {"total_phrases": 1, "mapped_phrases": 1, "failed_phrases": 0,
                  "summary_failures": 0, "clause_errors": 0},
        "annotated_pdf": "/tmp/whatever.pdf",  # should be replaced by API path
        "clauses": [{
            "clause_id": "C001",
            "clause_number": "3.2",
            "page_start": 1,
            "page_end": 1,
            "clause_text": "Students must maintain 75% attendance.",
            "summary": "Students need 75% attendance.",
            "important_phrases": [{
                "text": "75% attendance",
                "score": 0.9,
                "level": "HIGH",
                "category": "attendance",
                "risk_type": "academic",
                "risk_level": "MEDIUM",
                "mapping_status": "mapped",
                "locations": [{"page": 1, "bbox": [10, 20, 30, 40]}],
            }],
        }],
    }


def test_format_result_shape():
    formatted = format_result(_fake_pipeline_result())
    for k in ("document_id", "filename", "total_pages", "total_clauses",
              "status", "annotated_pdf", "clauses"):
        assert k in formatted
    # annotated_pdf converted to API path
    assert formatted["annotated_pdf"] == "/api/documents/EDU_DOC_T13/annotated"


def test_format_clause_and_phrase_keys():
    formatted = format_result(_fake_pipeline_result())
    c = formatted["clauses"][0]
    for k in ("clause_id", "clause_number", "page_start", "page_end",
              "clause_text", "summary", "important_phrases"):
        assert k in c
    p = c["important_phrases"][0]
    for k in ("text", "score", "level", "locations"):
        assert k in p
    assert p["locations"][0]["page"] == 1
    assert len(p["locations"][0]["bbox"]) == 4


def test_annotated_api_path_helper():
    assert _annotated_api_path("EDU_DOC_009") == "/api/documents/EDU_DOC_009/annotated"


def test_scanned_result_passthrough():
    scanned = {"document_id": "S1", "filename": "scan.pdf",
               "status": "scanned", "message": "OCR required"}
    out = format_result(scanned)
    assert out["status"] == "scanned"
    assert out["message"] == "OCR required"


def test_save_and_load_roundtrip():
    formatted = format_result(_fake_pipeline_result("EDU_DOC_RT"))
    path = save_result(formatted)
    assert os.path.exists(path)
    reloaded = load_result("EDU_DOC_RT")
    assert reloaded is not None
    assert reloaded["document_id"] == "EDU_DOC_RT"
    assert reloaded["clauses"][0]["clause_id"] == "C001"
    # valid JSON on disk
    with open(path, "r", encoding="utf-8") as f:
        json.load(f)
    os.remove(path)


def test_load_missing_returns_none():
    assert load_result("EDU_DOC_DOES_NOT_EXIST_999") is None
