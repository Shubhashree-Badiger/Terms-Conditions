"""
Phase 10 tests: PDF highlighting (ml/annotation/pdf_highlighter).
Run from the backend directory with the venv pytest.
"""
import os
import sys
import tempfile

import pymupdf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.annotation.pdf_highlighter import (
    highlight_phrase, highlight_page, generate_annotated_pdf, _color_for_level,
)


def _make_pdf(path, text="Students must maintain 75% attendance. Fees are non-refundable."):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def _bbox_of(path, word):
    doc = pymupdf.open(path)
    rects = doc.load_page(0).search_for(word)
    box = [round(v, 2) for v in rects[0]] if rects else None
    doc.close()
    return box


def test_color_for_level():
    assert _color_for_level("HIGH") == (1.0, 0.55, 0.55)
    assert _color_for_level("MEDIUM") == (1.0, 0.80, 0.40)
    assert _color_for_level("LOW") == (1.0, 0.95, 0.45)
    assert _color_for_level("unknown") == (1.0, 0.95, 0.45)


def test_highlight_phrase_adds_annotation():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src.pdf")
        _make_pdf(src)
        box = _bbox_of(src, "attendance")
        doc = pymupdf.open(src)
        page = doc.load_page(0)
        annot = highlight_phrase(page, box, "HIGH")
        assert annot is not None
        doc.close()


def test_generate_annotated_pdf():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src.pdf")
        out = os.path.join(d, "annotated.pdf")
        _make_pdf(src)
        box = _bbox_of(src, "non-refundable")
        mappings = [{
            "phrase": "non-refundable", "level": "HIGH",
            "mapping_status": "mapped",
            "locations": [{"page": 1, "bbox": box}],
        }]
        result = generate_annotated_pdf(src, mappings, out)
        assert os.path.exists(result)
        # annotation present
        ann = pymupdf.open(out)
        annots = list(ann.load_page(0).annots() or [])
        assert len(annots) >= 1
        # text still selectable (not rasterized)
        assert "non-refundable" in ann.load_page(0).get_text("text").lower()
        ann.close()


def test_original_pdf_not_modified():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src.pdf")
        out = os.path.join(d, "annotated.pdf")
        _make_pdf(src)
        size_before = os.path.getsize(src)
        box = _bbox_of(src, "attendance")
        mappings = [{
            "phrase": "attendance", "level": "MEDIUM",
            "mapping_status": "mapped",
            "locations": [{"page": 1, "bbox": box}],
        }]
        generate_annotated_pdf(src, mappings, out)
        # original file unchanged and has no annotations
        assert os.path.getsize(src) == size_before
        orig = pymupdf.open(src)
        assert len(list(orig.load_page(0).annots() or [])) == 0
        orig.close()


def test_failed_mappings_skipped():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src.pdf")
        out = os.path.join(d, "annotated.pdf")
        _make_pdf(src)
        mappings = [
            {"phrase": "x", "mapping_status": "failed", "locations": []},
            {"phrase": "y", "mapping_status": "failed", "locations": []},
        ]
        generate_annotated_pdf(src, mappings, out)
        ann = pymupdf.open(out)
        assert len(list(ann.load_page(0).annots() or [])) == 0
        ann.close()


def test_missing_source_raises():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "annotated.pdf")
        try:
            generate_annotated_pdf(os.path.join(d, "nope.pdf"), [], out)
            assert False, "should have raised"
        except FileNotFoundError:
            pass
