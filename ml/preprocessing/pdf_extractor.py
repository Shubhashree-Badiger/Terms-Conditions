"""
PDF Text Extraction with Coordinates (Phase 4)
===============================================
Uses PyMuPDF to extract, for every page:
  - readable page text
  - every word with its bounding box (x0, y0, x1, y1)
  - block / line / word indices

The word-level coordinates are essential for later phases (coordinate mapping
and PDF highlighting). The ORIGINAL PDF is never modified.

Public API:
    extract_page_words(page)      -> list[dict]   word records for one page
    extract_page_text(page)       -> str          readable text for one page
    reconstruct_text(words)       -> str          rebuild text from word records
    extract_pdf(path|bytes)       -> ExtractedDocument

Word record shape:
    {
        "text": "Students",
        "bbox": [72.0, 100.0, 115.0, 112.0],   # x0, y0, x1, y1
        "block": 1,
        "line": 2,
        "word_index": 0
    }
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Union

import pymupdf  # PyMuPDF


@dataclass
class PageContent:
    """Extracted content for a single page."""
    page_number: int                       # 1-based
    text: str                              # readable page text
    words: List[Dict] = field(default_factory=list)  # word records with bbox
    width: float = 0.0
    height: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "text": self.text,
            "words": self.words,
        }


@dataclass
class ExtractedDocument:
    """Fully extracted PDF document."""
    document_id: str
    file_name: str
    total_pages: int
    pages: List[PageContent] = field(default_factory=list)
    full_text: str = ""
    extraction_method: str = "PyMuPDF"
    error: Optional[str] = None

    @property
    def total_words(self) -> int:
        return sum(len(p.words) for p in self.pages)

    def to_dict(self) -> Dict:
        return {
            "document_id": self.document_id,
            "file_name": self.file_name,
            "total_pages": self.total_pages,
            "total_words": self.total_words,
            "extraction_method": self.extraction_method,
            "error": self.error,
            "pages": [p.to_dict() for p in self.pages],
        }


# --------------------------------------------------------------------------
# Page-level helpers
# --------------------------------------------------------------------------
def extract_page_words(page: "pymupdf.Page") -> List[Dict]:
    """Return every word on the page with its bounding box and indices.

    PyMuPDF page.get_text("words") yields tuples:
        (x0, y0, x1, y1, "word", block_no, line_no, word_no)
    sorted in natural reading order when sort=True.
    """
    raw = page.get_text("words", sort=True)
    records: List[Dict] = []
    for w in raw:
        x0, y0, x1, y1, text, block_no, line_no, word_no = w
        text = text.strip()
        if not text:
            continue
        records.append({
            "text": text,
            "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
            "block": int(block_no),
            "line": int(line_no),
            "word_index": int(word_no),
        })
    return records


def extract_page_text(page: "pymupdf.Page") -> str:
    """Return the readable text of a page in natural reading order."""
    return page.get_text("text", sort=True)


def reconstruct_text(words: List[Dict]) -> str:
    """Rebuild a readable text string from word records.

    Words on the same (block, line) are joined by spaces; a new line starts a
    new text line; a new block adds a blank line. This gives a normalized text
    whose ordering matches the word-coordinate list, useful for aligning NLP
    output back to coordinates.
    """
    if not words:
        return ""

    lines: List[str] = []
    current_line: List[str] = []
    prev_block = words[0]["block"]
    prev_line = words[0]["line"]

    for w in words:
        b, ln = w["block"], w["line"]
        if b != prev_block:
            # flush line, add blank line between blocks
            if current_line:
                lines.append(" ".join(current_line))
                current_line = []
            lines.append("")  # block separator
            prev_block, prev_line = b, ln
        elif ln != prev_line:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = []
            prev_line = ln
        current_line.append(w["text"])

    if current_line:
        lines.append(" ".join(current_line))

    # collapse 3+ blank lines to a single blank line
    text = "\n".join(lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


# --------------------------------------------------------------------------
# Document-level extraction
# --------------------------------------------------------------------------
def _extract_from_doc(doc: "pymupdf.Document", document_id: str, file_name: str) -> ExtractedDocument:
    pages: List[PageContent] = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        rect = page.rect
        pages.append(PageContent(
            page_number=i + 1,
            text=extract_page_text(page),
            words=extract_page_words(page),
            width=round(rect.width, 2),
            height=round(rect.height, 2),
        ))

    full_text = "\n\n".join(p.text for p in pages if p.text.strip())

    return ExtractedDocument(
        document_id=document_id,
        file_name=file_name,
        total_pages=len(pages),
        pages=pages,
        full_text=full_text,
    )


def extract_pdf(source: Union[str, bytes], document_id: Optional[str] = None,
                file_name: Optional[str] = None) -> ExtractedDocument:
    """Extract text + word coordinates from a PDF.

    Args:
        source: a filesystem path (str) or raw PDF bytes.
        document_id: identifier for the document (defaults to file stem).
        file_name: original file name (defaults to basename of path).

    Returns:
        ExtractedDocument. If extraction fails, .error is set and pages is empty.
    """
    # Resolve identifiers
    if isinstance(source, str):
        if not os.path.exists(source):
            return ExtractedDocument(
                document_id=document_id or "",
                file_name=file_name or os.path.basename(source),
                total_pages=0,
                error=f"File not found: {source}",
            )
        file_name = file_name or os.path.basename(source)
        document_id = document_id or os.path.splitext(os.path.basename(source))[0]
    else:
        file_name = file_name or "uploaded.pdf"
        document_id = document_id or "document"

    # Open
    try:
        if isinstance(source, str):
            doc = pymupdf.open(source)
        else:
            doc = pymupdf.open(stream=source, filetype="pdf")
    except Exception as e:
        return ExtractedDocument(
            document_id=document_id, file_name=file_name, total_pages=0,
            error=f"Failed to open PDF: {e}",
        )

    try:
        if doc.needs_pass:
            return ExtractedDocument(
                document_id=document_id, file_name=file_name, total_pages=0,
                error="PDF is password-protected.",
            )
        result = _extract_from_doc(doc, document_id, file_name)
    except Exception as e:
        result = ExtractedDocument(
            document_id=document_id, file_name=file_name, total_pages=0,
            error=f"Extraction failed: {e}",
        )
    finally:
        doc.close()

    return result


def is_probably_scanned(extracted: ExtractedDocument, min_words_per_page: float = 5.0) -> bool:
    """Heuristic: if the average words-per-page is very low, the PDF is likely
    a scanned/image document that needs OCR (not supported)."""
    if extracted.total_pages == 0:
        return True
    return (extracted.total_words / extracted.total_pages) < min_words_per_page
