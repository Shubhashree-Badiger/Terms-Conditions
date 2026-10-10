"""
Document Preprocessing (Phase 5)
================================
Cleans extracted PDF text for NLP while preserving the original text and
word coordinates for later PDF mapping.

Handles:
  - extra / repeated whitespace
  - broken lines (soft-wrapped mid-sentence)
  - hyphenated words split across line breaks  ("attend-\nance" -> "attendance")
  - empty lines
  - repeated headers / footers (detected across pages)
  - page-number / boilerplate lines
  - unicode normalization of quotes and dashes

Design principle (per project rules):
  * NEVER modify the original PDF.
  * Keep `original_text` (verbatim, for coordinate mapping) separate from
    `normalized_text` (cleaned, for NLP).

Public API:
    normalize_unicode(text) -> str
    dehyphenate(text) -> str
    collapse_whitespace(text) -> str
    detect_repeated_lines(pages_text) -> set[str]     # header/footer candidates
    is_boilerplate_line(line) -> bool
    preprocess_page(text, repeated_lines) -> str
    preprocess_document(extracted) -> PreprocessedDocument
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Set


# --------------------------------------------------------------------------
# Data container
# --------------------------------------------------------------------------
@dataclass
class PreprocessedPage:
    page_number: int
    original_text: str      # verbatim page text (from extractor)
    normalized_text: str    # cleaned text for NLP


@dataclass
class PreprocessedDocument:
    document_id: str
    pages: List[PreprocessedPage] = field(default_factory=list)
    normalized_text: str = ""    # full cleaned document text
    original_text: str = ""      # full verbatim document text
    removed_lines: List[str] = field(default_factory=list)  # headers/footers removed

    def to_dict(self) -> Dict:
        return {
            "document_id": self.document_id,
            "normalized_text": self.normalized_text,
            "removed_lines": self.removed_lines,
            "pages": [
                {
                    "page_number": p.page_number,
                    "original_text": p.original_text,
                    "normalized_text": p.normalized_text,
                }
                for p in self.pages
            ],
        }


# --------------------------------------------------------------------------
# Regex patterns
# --------------------------------------------------------------------------
_PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),                       # bare "12"
    re.compile(r"^\s*page\s*\d+\s*(of\s*\d+)?\s*$", re.I),  # "Page 3 of 40"
    re.compile(r"^\s*\d+\s*\|\s*p\s*a\s*g\s*e\s*$", re.I),  # "1 | P a g e"
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),               # "- 4 -"
]

_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*(printed on|generated on|downloaded from)\b.*$", re.I),
    re.compile(r"^\s*\(this page (is )?intentionally left blank\)\s*$", re.I),
    re.compile(r"^\s*(all rights reserved|copyright)\b.*$", re.I),
]

# hyphen at end of line: "attend-\nance" -> join
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
# a line break that is NOT a paragraph break (lowercase continues on next line)
_SOFT_LINEBREAK = re.compile(r"([a-z,;])\n([a-z])")


# --------------------------------------------------------------------------
# Cleaning helpers
# --------------------------------------------------------------------------
def normalize_unicode(text: str) -> str:
    """Normalize smart quotes, dashes and stray glyphs to ASCII equivalents."""
    if not text:
        return ""
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
        "\u201c": '"', "\u201d": '"', "\u201e": '"',
        "\u2013": "-", "\u2014": "-", "\u2212": "-",
        "\u2026": "...",
        "\u00a0": " ",   # non-breaking space
        "\u2022": "-",   # bullet
        "\ufb01": "fi", "\ufb02": "fl",
        "\u00e6": "ae",
    }
    for a, b in replacements.items():
        text = text.replace(a, b)
    return text


def dehyphenate(text: str) -> str:
    """Join words split by a hyphen at a line break: 'attend-\\nance' -> 'attendance'."""
    return _HYPHEN_LINEBREAK.sub(r"\1\2", text)


def join_soft_linebreaks(text: str) -> str:
    """Join lines that were soft-wrapped mid-sentence (lowercase -> lowercase)."""
    return _SOFT_LINEBREAK.sub(r"\1 \2", text)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs, trim lines, and limit blank lines."""
    # normalize spaces within lines
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    # collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_page_number_line(line: str) -> bool:
    s = line.strip()
    return any(p.match(s) for p in _PAGE_NUMBER_PATTERNS)


def is_boilerplate_line(line: str) -> bool:
    s = line.strip()
    return any(p.match(s) for p in _BOILERPLATE_PATTERNS)


# --------------------------------------------------------------------------
# Header / footer detection across pages
# --------------------------------------------------------------------------
def detect_repeated_lines(pages_text: List[str], min_repeat_ratio: float = 0.5,
                          max_line_len: int = 90) -> Set[str]:
    """Detect lines that repeat across many pages (likely headers/footers).

    A short line appearing on >= min_repeat_ratio of pages is treated as a
    header/footer and removed from the normalized text.
    """
    if len(pages_text) < 3:
        return set()

    counter: Counter = Counter()
    for text in pages_text:
        seen_on_page = set()
        for raw in text.split("\n"):
            ln = raw.strip()
            if not ln or len(ln) > max_line_len:
                continue
            # normalize for comparison (case-insensitive, page numbers stripped)
            key = re.sub(r"\d+", "#", ln.lower())
            if key not in seen_on_page:
                counter[key] += 1
                seen_on_page.add(key)

    threshold = max(2, int(len(pages_text) * min_repeat_ratio))
    repeated = {key for key, count in counter.items() if count >= threshold}
    return repeated


def _line_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


# --------------------------------------------------------------------------
# Page + document preprocessing
# --------------------------------------------------------------------------
def preprocess_page(text: str, repeated_lines: Set[str]) -> str:
    """Produce normalized text for one page."""
    if not text:
        return ""

    text = normalize_unicode(text)
    text = dehyphenate(text)

    # remove header/footer, page-number and boilerplate lines
    kept: List[str] = []
    for raw in text.split("\n"):
        ln = raw.strip()
        if not ln:
            kept.append("")
            continue
        if _line_key(ln) in repeated_lines:
            continue
        if is_page_number_line(ln) or is_boilerplate_line(ln):
            continue
        kept.append(raw)

    text = "\n".join(kept)
    text = join_soft_linebreaks(text)
    text = collapse_whitespace(text)
    return text


def preprocess_document(extracted) -> PreprocessedDocument:
    """Preprocess an ExtractedDocument (from pdf_extractor.extract_pdf).

    Returns a PreprocessedDocument with per-page original + normalized text
    and the full normalized document text.
    """
    pages_text = [p.text for p in extracted.pages]
    repeated = detect_repeated_lines(pages_text)

    pp_pages: List[PreprocessedPage] = []
    for p in extracted.pages:
        normalized = preprocess_page(p.text, repeated)
        pp_pages.append(PreprocessedPage(
            page_number=p.page_number,
            original_text=p.text,
            normalized_text=normalized,
        ))

    full_norm = "\n\n".join(pp.normalized_text for pp in pp_pages if pp.normalized_text.strip())
    full_orig = "\n\n".join(pp.original_text for pp in pp_pages if pp.original_text.strip())

    # human-readable list of removed header/footer keys
    removed = sorted(repeated)

    return PreprocessedDocument(
        document_id=extracted.document_id,
        pages=pp_pages,
        normalized_text=full_norm,
        original_text=full_orig,
        removed_lines=removed,
    )
