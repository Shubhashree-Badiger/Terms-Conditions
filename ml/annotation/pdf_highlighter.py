"""
PDF Highlighter (Phase 10)
==========================
Adds highlight annotations to a COPY of the PDF for each successfully mapped
phrase, producing an annotated PDF that preserves the original text, layout
and searchability (NOT rasterized).

Rules honored:
  * Never modify the original PDF - we open it and save to a new path.
  * Highlight only meaningful, mapped phrases.
  * Color-code by importance level so HIGH clauses stand out.

Public API:
    highlight_phrase(page, bbox, level) -> annotation
    highlight_page(page, boxes) -> int                 # boxes: [(bbox, level), ...]
    generate_annotated_pdf(src_path, mappings, out_path) -> str

`mappings` is a flat list of phrase-mapping records, each:
    {
      "phrase": "...",
      "level": "HIGH" | "MEDIUM" | "LOW",   # importance level (optional)
      "locations": [ {"page": 4, "bbox": [x0,y0,x1,y1]}, ... ],
      "mapping_status": "mapped" | "failed"
    }
Only records with mapping_status starting 'mapped' are drawn.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

import pymupdf

logger = logging.getLogger("analyzer.highlight")

# RGB colours in 0..1 range for highlight annotations
_LEVEL_COLORS = {
    "HIGH":   (1.0, 0.55, 0.55),   # red / salmon
    "MEDIUM": (1.0, 0.80, 0.40),   # orange
    "LOW":    (1.0, 0.95, 0.45),   # yellow
}
_DEFAULT_COLOR = (1.0, 0.95, 0.45)  # yellow


def _color_for_level(level: str) -> Tuple[float, float, float]:
    return _LEVEL_COLORS.get((level or "").upper(), _DEFAULT_COLOR)


def highlight_phrase(page: "pymupdf.Page", bbox: List[float], level: str = "LOW"):
    """Add a single highlight annotation on the given bbox.
    Returns the created annotation (or None on failure)."""
    try:
        rect = pymupdf.Rect(bbox)
        annot = page.add_highlight_annot(rect)
        annot.set_colors(stroke=_color_for_level(level))
        annot.update()
        return annot
    except Exception as e:
        logger.debug("Failed to highlight bbox %s on page %s: %s",
                     bbox, page.number, e)
        return None


def highlight_page(page: "pymupdf.Page", boxes: List[Tuple[List[float], str]]) -> int:
    """Highlight multiple boxes on one page. boxes = [(bbox, level), ...].
    Returns the count of successfully added annotations."""
    count = 0
    for bbox, level in boxes:
        if highlight_phrase(page, bbox, level) is not None:
            count += 1
    return count


def generate_annotated_pdf(src_path: str, mappings: List[Dict], out_path: str) -> str:
    """Create an annotated copy of src_path with all mapped phrases highlighted.

    Args:
        src_path: path to the ORIGINAL PDF (never modified).
        mappings: flat list of phrase-mapping records (see module docstring).
        out_path: where to write the annotated PDF.

    Returns:
        out_path on success.

    Raises:
        FileNotFoundError if src_path does not exist.
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source PDF not found: {src_path}")

    # Group boxes by page index (0-based) so we open each page once.
    by_page: Dict[int, List[Tuple[List[float], str]]] = {}
    for m in mappings:
        if not str(m.get("mapping_status", "")).startswith("mapped"):
            continue
        level = m.get("level", "LOW")
        for loc in m.get("locations", []):
            page_no = loc.get("page")
            bbox = loc.get("bbox")
            if page_no is None or not bbox:
                continue
            by_page.setdefault(page_no - 1, []).append((bbox, level))

    doc = pymupdf.open(src_path)   # a fresh handle; original file untouched
    total = 0
    try:
        for page_index, boxes in by_page.items():
            if 0 <= page_index < len(doc):
                page = doc.load_page(page_index)
                total += highlight_page(page, boxes)

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        # incremental=False + clean save; text stays selectable (no rasterization)
        doc.save(out_path, garbage=4, deflate=True)
    finally:
        doc.close()

    logger.info("Annotated PDF written: %s (%d highlights)", out_path, total)
    return out_path
