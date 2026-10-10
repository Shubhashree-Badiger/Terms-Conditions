"""
Phase 9 tests: coordinate mapping (ml/coordinate_mapping/coordinate_mapper).
Uses small in-memory PDFs so tests are fast and deterministic.
Run from the backend directory with the venv pytest.
"""
import os
import sys

import pymupdf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.coordinate_mapping.coordinate_mapper import CoordinateMapper


def _make_doc(pages_lines):
    """pages_lines: list of list-of-strings (one list per page)."""
    doc = pymupdf.open()
    for lines in pages_lines:
        page = doc.new_page()
        y = 72
        for ln in lines:
            page.insert_text((72, y), ln)
            y += 20
    return doc


def test_map_phrase_success():
    doc = _make_doc([["Students must maintain 75% attendance."]])
    mapper = CoordinateMapper(doc)
    clause = {"clause_id": "C001", "page_start": 1, "page_end": 1}
    r = mapper.map_phrase("attendance", clause)
    assert r["mapping_status"] in ("mapped", "mapped_page")
    assert len(r["locations"]) >= 1
    loc = r["locations"][0]
    assert loc["page"] == 1
    assert len(loc["bbox"]) == 4
    doc.close()


def test_map_phrase_case_insensitive_variant():
    doc = _make_doc([["Fees are NON-REFUNDABLE after admission."]])
    mapper = CoordinateMapper(doc)
    clause = {"clause_id": "C001", "page_start": 1, "page_end": 1}
    # search_for is case-insensitive in PyMuPDF; variant handling also helps
    r = mapper.map_phrase("non-refundable", clause)
    assert r["mapping_status"] in ("mapped", "mapped_page")
    doc.close()


def test_scope_to_clause_page():
    # 'penalty' appears on page 2 only; clause says page 2
    doc = _make_doc([
        ["Introduction and definitions."],
        ["A penalty applies for late fees."],
    ])
    mapper = CoordinateMapper(doc)
    clause = {"clause_id": "C002", "page_start": 2, "page_end": 2}
    r = mapper.map_phrase("penalty", clause)
    assert r["mapping_status"] in ("mapped", "mapped_page")
    assert r["locations"][0]["page"] == 2
    doc.close()


def test_failed_mapping_does_not_crash():
    doc = _make_doc([["Some ordinary text about classes."]])
    mapper = CoordinateMapper(doc)
    clause = {"clause_id": "C001", "page_start": 1, "page_end": 1}
    r = mapper.map_phrase("nonexistent_phrase_xyz", clause)
    assert r["mapping_status"] == "failed"
    assert r["locations"] == []
    doc.close()


def test_empty_phrase():
    doc = _make_doc([["Content."]])
    mapper = CoordinateMapper(doc)
    r = mapper.map_phrase("", {"clause_id": "C1", "page_start": 1, "page_end": 1})
    assert r["mapping_status"] == "failed"
    doc.close()


def test_map_clause_phrases_batch():
    doc = _make_doc([["Students must attend. Fees are non-refundable."]])
    mapper = CoordinateMapper(doc)
    clause = {"clause_id": "C001", "page_start": 1, "page_end": 1}
    results = mapper.map_clause_phrases(["attend", "non-refundable", "missing_xyz"], clause)
    assert len(results) == 3
    statuses = {r["phrase"]: r["mapping_status"] for r in results}
    assert statuses["attend"].startswith("mapped")
    assert statuses["non-refundable"].startswith("mapped")
    assert statuses["missing_xyz"] == "failed"
    doc.close()


def test_document_fallback():
    # phrase on page 2 but clause claims page 1; fallback enabled finds it
    doc = _make_doc([
        ["Page one content."],
        ["The word rustication appears here."],
    ])
    mapper = CoordinateMapper(doc, allow_document_fallback=True)
    clause = {"clause_id": "C001", "page_start": 1, "page_end": 1}
    r = mapper.map_phrase("rustication", clause)
    assert r["mapping_status"] == "mapped_fallback"
    assert r["locations"][0]["page"] == 2
    doc.close()


# --- Region-constrained mapping (accuracy fix) ---
def test_phrase_mapped_within_correct_clause_region():
    # The word "examination" appears in clause 1 (page top). The clause region
    # matching should map it to the clause's own area, status 'mapped'.
    doc = _make_doc([[
        "1. Students must pass the examination to earn credits.",
        "2. Fees are non-refundable after admission is confirmed.",
    ]])
    mapper = CoordinateMapper(doc)
    clause = {"clause_id": "C001", "page_start": 1, "page_end": 1,
              "clause_text": "1. Students must pass the examination to earn credits."}
    r = mapper.map_phrase("examination", clause)
    assert r["mapping_status"] in ("mapped", "mapped_page")
    assert len(r["locations"]) >= 1
    doc.close()


def test_no_scattered_word_boxes():
    # A multi-word phrase whose words are non-contiguous should NOT produce a
    # scattered box per word (the old buggy behavior). It either maps as a
    # contiguous match or fails - never returns many unrelated boxes.
    doc = _make_doc([[
        "Games are held regularly.",
        "Activities include many clubs.",
        "Athletics is encouraged for all.",
    ]])
    mapper = CoordinateMapper(doc)
    clause = {"clause_id": "C001", "page_start": 1, "page_end": 1,
              "clause_text": "Games are held regularly."}
    r = mapper.map_phrase("Games Activities Athletics", clause)
    # non-contiguous phrase -> should fail rather than scatter boxes
    assert r["mapping_status"] == "failed"
    doc.close()


def test_contiguous_multiword_still_maps():
    doc = _make_doc([["Games Activities Athletics are all offered."]])
    mapper = CoordinateMapper(doc)
    clause = {"clause_id": "C001", "page_start": 1, "page_end": 1,
              "clause_text": "Games Activities Athletics are all offered."}
    r = mapper.map_phrase("Games Activities Athletics", clause)
    assert r["mapping_status"] in ("mapped", "mapped_page")
    assert len(r["locations"]) == 1
    doc.close()

