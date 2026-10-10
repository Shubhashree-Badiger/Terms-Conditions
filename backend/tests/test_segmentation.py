"""
Phase 6 tests: clause segmentation (ml/clause_segmentation/clause_segmenter).
Run from the backend directory with the venv pytest.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.clause_segmentation.clause_segmenter import (
    detect_numbered_sections, detect_bullet_sections, detect_headings,
    segment_clauses, assign_clause_ids,
)


def test_detect_numbered():
    assert detect_numbered_sections("1. Title and Commencement") == "1"
    assert detect_numbered_sections("3.2 Students must attend") == "3.2"
    assert detect_numbered_sections("1.2.3 Sub clause here") == "1.2.3"
    assert detect_numbered_sections("Students must attend") is None


def test_detect_bullets_and_letters():
    assert detect_bullet_sections("(a) first point") == "(a)"
    assert detect_bullet_sections("(i) roman point") == "(i)"
    assert detect_bullet_sections("- a bullet point") == "-"
    assert detect_bullet_sections("normal text line") is None


def test_detect_headings():
    assert detect_headings("ATTENDANCE REQUIREMENTS")
    assert detect_headings("Definitions:")
    assert not detect_headings("Students must maintain 75% attendance in each course.")


def test_segment_numbered_clauses():
    pages = [{
        "page_number": 1,
        "normalized_text": (
            "1. Attendance\n"
            "Students must maintain a minimum of 75% attendance.\n"
            "Shortage will lead to debarment.\n"
            "2. Fees\n"
            "Fees once paid are non-refundable."
        ),
    }]
    clauses = assign_clause_ids(segment_clauses(pages))
    assert len(clauses) == 2
    assert clauses[0]["clause_number"] == "1"
    assert "attendance" in clauses[0]["clause_text"].lower()
    # multi-sentence clause preserved
    assert "debarment" in clauses[0]["clause_text"].lower()
    assert clauses[1]["clause_number"] == "2"
    assert clauses[0]["clause_id"] == "C001"


def test_clause_has_required_fields():
    pages = [{"page_number": 3, "normalized_text": "5.1 A rule about exams."}]
    clauses = assign_clause_ids(segment_clauses(pages))
    assert len(clauses) == 1
    c = clauses[0]
    for key in ("clause_id", "clause_number", "clause_text",
                "page_start", "page_end", "start_position", "end_position"):
        assert key in c
    assert c["page_start"] == 3
    assert c["end_position"] > c["start_position"]


def test_plain_text_fallback():
    # No structural markers -> sentence-grouped clauses
    pages = [{
        "page_number": 1,
        "normalized_text": (
            "The hostel gate closes at nine PM. Visitors are not permitted after that. "
            "Students must sign the register. Late entry requires warden approval."
        ),
    }]
    clauses = assign_clause_ids(segment_clauses(pages))
    assert len(clauses) >= 1
    assert all(c["clause_number"] == "" for c in clauses)


# --- Phase 18: TOC filtering ---
from ml.clause_segmentation.clause_segmenter import is_toc_entry, filter_toc


def test_is_toc_entry_detects_toc_lines():
    assert is_toc_entry({"clause_text": "1.0 Title & Commencement 04"})
    assert is_toc_entry({"clause_text": "Admission .......... 05"})
    assert is_toc_entry({"clause_text": "Section Particulars Page No. Preamble 03"})


def test_is_toc_entry_keeps_real_clauses():
    real = {"clause_text": ("Students failing to maintain the minimum attendance "
                            "requirement of 75% shall be debarred from the examination.")}
    assert not is_toc_entry(real)


def test_filter_toc_removes_toc_only():
    clauses = [
        {"clause_text": "1.0 Title & Commencement 04"},
        {"clause_text": ("Students must maintain 75% attendance to be eligible for "
                         "the end semester examination as per regulations.")},
        {"clause_text": "2.0 Definitions 05"},
    ]
    kept = filter_toc(clauses)
    assert len(kept) == 1
    assert "attendance" in kept[0]["clause_text"].lower()
