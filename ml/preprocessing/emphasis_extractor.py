"""
Emphasis Extractor (Phase 25)
=============================
Reads the ORIGINAL emphasis the document author already applied - BOLD and
ITALIC text - straight from the PDF. If a policy author bolded a phrase (e.g.
"internal evaluation of 100 marks", "another 50 marks", "mandatory audit
courses"), that is a strong human signal that the phrase matters, so the
analyzer must highlight it FIRST and never skip it.

PyMuPDF exposes per-span font info via page.get_text("dict"). A span is bold if
its font flags have the bold bit set (flags & 2**4) or the font name contains
"bold"/"black"/"semibold"; italic if the italic bit (flags & 2**1) is set or
the name contains "italic"/"oblique".

Public API:
    extract_emphasized_spans(doc) -> list[dict]
        [{ "text": "another 50 marks", "page": 3, "bold": True, "italic": False }, ...]

The pipeline matches these spans against each clause's text and injects the
matches as high-priority candidate phrases (see analysis_pipeline).
"""
from __future__ import annotations

import re
from typing import Dict, List

# PyMuPDF span flag bits
_FLAG_ITALIC = 1 << 1   # 2
_FLAG_BOLD = 1 << 4     # 16

_BOLD_NAME = re.compile(r"bold|black|semibold|heavy", re.IGNORECASE)
_ITALIC_NAME = re.compile(r"italic|oblique", re.IGNORECASE)


def _span_is_bold(span: Dict) -> bool:
    flags = span.get("flags", 0)
    if flags & _FLAG_BOLD:
        return True
    return bool(_BOLD_NAME.search(span.get("font", "")))


def _span_is_italic(span: Dict) -> bool:
    flags = span.get("flags", 0)
    if flags & _FLAG_ITALIC:
        return True
    return bool(_ITALIC_NAME.search(span.get("font", "")))


def _looks_like_body_default(spans_bold_ratio: float) -> bool:
    """If nearly the whole document is 'bold', the base font is just heavy -
    emphasis then carries no signal, so we ignore it."""
    return spans_bold_ratio > 0.6


def extract_emphasized_spans(doc: "object") -> List[Dict]:
    """Extract bold / italic runs from a PyMuPDF document.

    Consecutive emphasized spans on the same line are merged into one phrase so
    a bolded sentence fragment comes back whole rather than word-by-word.
    Returns [] if emphasis is not a useful signal for this document.
    """
    out: List[Dict] = []
    total_spans = 0
    bold_spans = 0

    for pidx in range(len(doc)):
        page = doc.load_page(pidx)
        try:
            data = page.get_text("dict")
        except Exception:
            continue

        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                # merge adjacent emphasized spans within a line
                run_text: List[str] = []
                run_bold = False
                run_italic = False

                def flush_run():
                    nonlocal run_text, run_bold, run_italic
                    if run_text:
                        text = re.sub(r"\s+", " ", " ".join(run_text)).strip()
                        # keep only meaningful emphasized runs (>=2 chars,
                        # contains a letter/number, not a lone symbol)
                        if len(text) >= 2 and re.search(r"[A-Za-z0-9]", text):
                            out.append({
                                "text": text,
                                "page": pidx + 1,
                                "bold": run_bold,
                                "italic": run_italic,
                            })
                    run_text = []
                    run_bold = False
                    run_italic = False

                for span in line.get("spans", []):
                    stext = span.get("text", "")
                    if not stext.strip():
                        continue
                    total_spans += 1
                    b = _span_is_bold(span)
                    it = _span_is_italic(span)
                    if b:
                        bold_spans += 1
                    if b or it:
                        run_text.append(stext)
                        run_bold = run_bold or b
                        run_italic = run_italic or it
                    else:
                        flush_run()
                flush_run()

    if total_spans == 0:
        return []
    # If most of the doc is "bold", emphasis is not discriminative -> drop it.
    if _looks_like_body_default(bold_spans / max(1, total_spans)):
        return []
    return out
