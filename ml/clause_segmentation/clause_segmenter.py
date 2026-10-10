"""
Clause Segmenter (Phase 6)
==========================
Hybrid clause segmentation for educational regulation documents.

Strategy
--------
PRIMARY  : regex-based structural detection of clause markers such as
           1.  1.1  1.2.3  2.3(a)  (a)  (b)  (i)  (ii)  A.  B.  and bullets.
SECONDARY: when a page (or region) has no structural markers, fall back to
           spaCy sentence segmentation, grouping sentences into clauses.

A clause MAY span multiple sentences. We do NOT split every sentence into a
separate clause. Content following a marker (including wrapped lines and
sub-points) belongs to that clause until the next same-or-higher-level marker.

Each clause:
    {
      "clause_id":      "C001",
      "clause_number":  "3.2" | "(a)" | "-" | "",
      "clause_text":    "...",
      "page_start":     4,
      "page_end":       4,
      "start_position": <char offset within the document normalized text>,
      "end_position":   <char offset>
    }

Public API:
    detect_numbered_sections(line) -> str | None
    detect_bullet_sections(line)   -> str | None
    detect_headings(line)          -> bool
    segment_clauses(pages)         -> list[dict]
    assign_clause_ids(clauses)     -> list[dict]
"""
from __future__ import annotations

import re
from typing import List, Dict, Optional

try:
    import spacy
    _NLP = None
except ImportError:  # pragma: no cover
    spacy = None
    _NLP = None


# --------------------------------------------------------------------------
# Marker detection
# --------------------------------------------------------------------------
# Numbered / decimal: "1." "1.1" "1.2.3" "12)" "2.3(a)"
_NUMBERED = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*[\.\)]?\s*(\(?[a-z]\)?)?\s+\S")
# Lettered:  "(a)" "a)" "a."  "(i)" "(ii)"  "A." "B)"
_PAREN_ALPHA = re.compile(r"^\s*\(?([a-zA-Z]|[ivxlcdm]+)\)\s+\S")
_ALPHA_DOT = re.compile(r"^\s*([A-Za-z])[\.\)]\s+\S")
_ROMAN = re.compile(r"^\s*\(?((?:i{1,3}|iv|v|vi{0,3}|ix|x{1,3}))\)?[\.\)]?\s+\S", re.I)
# Bullets
_BULLET = re.compile(r"^\s*([\-\*\u2022\u25cf\u25aa\u2023\u2043o])\s+\S")

# Heading: short, mostly title/upper case, no terminal period
_HEADING_MAXLEN = 70


def detect_numbered_sections(line: str) -> Optional[str]:
    """Return the numeric clause marker (e.g. '3.2') if the line starts one."""
    m = _NUMBERED.match(line)
    if m:
        num = m.group(1)
        sub = m.group(2)
        if sub:
            sub = sub.strip()
            return f"{num}{sub}" if sub.startswith("(") else f"{num}.{sub}"
        return num
    return None


def detect_bullet_sections(line: str) -> Optional[str]:
    """Return a normalized bullet/letter marker if the line starts one."""
    if _BULLET.match(line):
        return "-"
    m = _PAREN_ALPHA.match(line)
    if m:
        return f"({m.group(1)})"
    m = _ROMAN.match(line)
    if m:
        return f"({m.group(1).lower()})"
    m = _ALPHA_DOT.match(line)
    if m:
        # Avoid treating a normal capitalized sentence as a marker: require the
        # letter to be a single char followed by . or ) and short leading token.
        return f"{m.group(1)}."
    return None


def detect_marker(line: str) -> Optional[str]:
    """Return any clause marker at the start of the line, or None."""
    return detect_numbered_sections(line) or detect_bullet_sections(line)


def detect_headings(line: str) -> bool:
    """Heuristic: is this line a section heading (not a clause body)?"""
    s = line.strip()
    if not s or len(s) > _HEADING_MAXLEN:
        return False
    if s.endswith(('.', ':', ';', ',')):
        # headings usually don't end with a period; a colon can be a heading
        if not s.endswith(':'):
            return False
    words = s.split()
    if len(words) > 9:
        return False
    # A short line ending with a colon is treated as a heading
    if s.endswith(':'):
        return True
    # ALL CAPS heading
    letters = [c for c in s if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        return True
    return False


# --------------------------------------------------------------------------
# spaCy fallback
# --------------------------------------------------------------------------
def _get_nlp():
    global _NLP
    if _NLP is None and spacy is not None:
        try:
            _NLP = spacy.load("en_core_web_sm")
            _NLP.max_length = 2_000_000
        except Exception:
            _NLP = False  # mark unavailable
    return _NLP or None


def _sentence_split(text: str) -> List[str]:
    nlp = _get_nlp()
    if nlp:
        return [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    # regex fallback
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if s.strip()]


def _group_sentences_into_clauses(sentences: List[str],
                                  max_sentences: int = 2,
                                  max_chars: int = 320) -> List[str]:
    """Group sentences into multi-sentence clauses of reasonable size."""
    clauses: List[str] = []
    buf: List[str] = []
    buf_len = 0
    for sent in sentences:
        buf.append(sent)
        buf_len += len(sent)
        if len(buf) >= max_sentences or buf_len >= max_chars:
            clauses.append(" ".join(buf))
            buf, buf_len = [], 0
    if buf:
        clauses.append(" ".join(buf))
    return clauses


# --------------------------------------------------------------------------
# Core segmentation
# --------------------------------------------------------------------------
def _clause_level(marker: str) -> int:
    """Rough nesting level: top-level number = 0, decimals deeper, letters/bullets deepest."""
    if not marker or marker == "-":
        return 9
    if re.match(r"^\d", marker):
        return marker.count(".")
    return 8  # (a) (i) A.


def segment_clauses(pages: List[Dict]) -> List[Dict]:
    """Segment a document into clauses.

    Args:
        pages: list of {"page_number": int, "normalized_text": str}
               (as produced by PreprocessedDocument.pages -> dicts, or built
               directly). Each page's normalized text is processed in order.

    Returns:
        list of clause dicts (without ids yet; call assign_clause_ids()).
    """
    clauses: List[Dict] = []
    doc_offset = 0  # running char offset across the whole normalized document

    # State for the currently-open structural clause
    cur_marker: Optional[str] = None
    cur_lines: List[str] = []
    cur_page_start: Optional[int] = None
    cur_page_end: Optional[int] = None
    cur_start_pos: int = 0

    # buffer of non-structural lines (become sentence-split clauses)
    plain_buf: List[str] = []
    plain_page_start: Optional[int] = None
    plain_start_pos: int = 0

    def flush_structural():
        nonlocal cur_marker, cur_lines, cur_page_start, cur_page_end, cur_start_pos
        if cur_marker is not None and cur_lines:
            text = " ".join(l.strip() for l in cur_lines).strip()
            if text:
                clauses.append({
                    "clause_number": cur_marker,
                    "clause_text": text,
                    "page_start": cur_page_start,
                    "page_end": cur_page_end,
                    "start_position": cur_start_pos,
                    "end_position": cur_start_pos + len(text),
                })
        cur_marker, cur_lines = None, []
        cur_page_start, cur_page_end = None, None

    def flush_plain():
        nonlocal plain_buf, plain_page_start, plain_start_pos
        if plain_buf:
            block = " ".join(l.strip() for l in plain_buf).strip()
            if block:
                for piece in _group_sentences_into_clauses(_sentence_split(block)):
                    clauses.append({
                        "clause_number": "",
                        "clause_text": piece,
                        "page_start": plain_page_start,
                        "page_end": plain_page_start,
                        "start_position": plain_start_pos,
                        "end_position": plain_start_pos + len(piece),
                    })
        plain_buf = []
        plain_page_start = None

    for page in pages:
        page_num = page["page_number"]
        text = page.get("normalized_text", "") or ""
        for raw in text.split("\n"):
            line = raw.rstrip()
            stripped = line.strip()
            line_len = len(raw) + 1  # +1 for newline

            if not stripped:
                doc_offset += line_len
                continue

            marker = detect_marker(stripped)
            heading = detect_headings(stripped) if marker is None else False

            if marker is not None:
                # New structural clause begins -> flush any open buffers
                flush_plain()
                flush_structural()
                cur_marker = marker
                cur_lines = [stripped]
                cur_page_start = page_num
                cur_page_end = page_num
                cur_start_pos = doc_offset
            elif heading:
                # Heading ends the current clause; headings are not clauses
                flush_plain()
                flush_structural()
            else:
                # Continuation line
                if cur_marker is not None:
                    cur_lines.append(stripped)
                    cur_page_end = page_num
                else:
                    if plain_page_start is None:
                        plain_page_start = page_num
                        plain_start_pos = doc_offset
                    plain_buf.append(stripped)

            doc_offset += line_len

    # flush leftovers
    flush_plain()
    flush_structural()

    return clauses


def assign_clause_ids(clauses: List[Dict]) -> List[Dict]:
    """Assign sequential clause_id values (C001, C002, ...)."""
    for i, c in enumerate(clauses, 1):
        c["clause_id"] = f"C{i:03d}"
    # reorder keys for readability
    ordered = []
    for c in clauses:
        ordered.append({
            "clause_id": c["clause_id"],
            "clause_number": c.get("clause_number", ""),
            "clause_text": c["clause_text"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "start_position": c["start_position"],
            "end_position": c["end_position"],
        })
    return ordered


# A short clause whose text ends in a bare page number (optionally after
# dot leaders) is almost always a table-of-contents entry, e.g.
#   "1.0 Title & Commencement 04"  or  "Admission .......... 05"
_TOC_TRAILING_NUM = re.compile(r".*?[\.\s]+\d{1,3}\s*$")


def is_toc_entry(clause: Dict) -> bool:
    """Heuristic: does this clause look like a table-of-contents line?"""
    text = (clause.get("clause_text") or "").strip()
    if len(text) > 80:
        return False  # real clauses are longer than TOC lines
    # ends in a bare page number and has few words
    if _TOC_TRAILING_NUM.match(text) and len(text.split()) <= 10:
        # dot leaders are a strong TOC signal
        if "...." in text or ". ." in text:
            return True
        # short heading-like line ending in a number
        if len(text.split()) <= 8:
            return True
    return False


def filter_toc(clauses: List[Dict]) -> List[Dict]:
    """Remove clauses that look like table-of-contents / front-matter entries."""
    return [c for c in clauses if not is_toc_entry(c)]


def segment_document(preprocessed, drop_toc: bool = True) -> List[Dict]:
    """Convenience: segment a PreprocessedDocument into clauses with ids.

    drop_toc=True removes table-of-contents entries (Phase 18 improvement).
    """
    pages = [
        {"page_number": p.page_number, "normalized_text": p.normalized_text}
        for p in preprocessed.pages
    ]
    clauses = segment_clauses(pages)
    if drop_toc:
        clauses = filter_toc(clauses)
    return assign_clause_ids(clauses)
