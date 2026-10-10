"""
Coordinate Mapper (Phase 9, hardened in Phase 18/23)
=====================================================
Locates an important phrase (found by NLP) at its exact position(s) in the
ORIGINAL PDF, returning page numbers and bounding boxes for highlighting.

Accuracy strategy
-----------------
The naive approach - searching the whole page for a phrase and taking the
first match - misplaces highlights, because a common word (e.g. "examination",
"of the") occurs many times on a page. To fix this we:

  1. First locate the CLAUSE'S REGION on the page: the rectangle spanning the
     PDF words that belong to this clause. We do this by matching the clause's
     leading words against the page word list.
  2. Then search for the phrase using page.search_for(..., clip=region), so a
     match is only accepted INSIDE the clause's own area. This keeps a
     highlight on the right occurrence.
  3. If the region can't be determined, fall back to a whole-page search but
     only accept a CONTIGUOUS phrase match (never scattered single words).

The previous scattered word-by-word fallback (which highlighted each word's
first occurrence anywhere on the page) has been removed - it was the cause of
misplaced highlights.

On failure: log, set mapping_status = "failed", and continue.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import pymupdf

logger = logging.getLogger("analyzer.coordmap")


def _clean_variants(phrase: str) -> List[str]:
    """Progressively relaxed search variants of a phrase (deduped)."""
    variants: List[str] = []

    def add(v: str):
        v = v.strip()
        if v and v not in variants:
            variants.append(v)

    add(phrase)
    add(re.sub(r"\s+", " ", phrase))
    add(phrase.strip(" .,;:()[]\"'"))
    add(phrase.rstrip("."))
    return variants


def _rect_to_bbox(rect: "pymupdf.Rect") -> List[float]:
    return [round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)]


def _norm_word(w: str) -> str:
    return re.sub(r"[^a-z0-9%]", "", w.lower())


class CoordinateMapper:
    def __init__(self, doc: "pymupdf.Document", allow_document_fallback: bool = False):
        self.doc = doc
        self.allow_document_fallback = allow_document_fallback
        self._page_count = len(doc)
        # cache of page_index -> word list (x0,y0,x1,y1,text,block,line,word)
        self._words_cache: Dict[int, list] = {}

    # -- page words -------------------------------------------------------
    def _page_words(self, page_index: int) -> list:
        if page_index not in self._words_cache:
            page = self.doc.load_page(page_index)
            self._words_cache[page_index] = page.get_text("words", sort=True)
        return self._words_cache[page_index]

    # -- clause region ----------------------------------------------------
    def _find_clause_region(self, page_index: int, clause_text: str
                            ) -> Optional["pymupdf.Rect"]:
        """Find the rectangle on the page that contains this clause, by matching
        a run of the clause's leading words against the page word sequence.

        Returns a Rect covering the matched run (with a little vertical
        padding), or None if the clause could not be located on this page.
        """
        words = self._page_words(page_index)
        if not words:
            return None

        # normalized page tokens
        page_tokens = [_norm_word(w[4]) for w in words]

        # clause "anchor" = its first several meaningful tokens
        clause_tokens = [_norm_word(t) for t in clause_text.split()]
        clause_tokens = [t for t in clause_tokens if t]
        anchor = clause_tokens[:6]
        if len(anchor) < 3:
            return None

        # slide over the page tokens looking for the anchor sequence
        n = len(page_tokens)
        a = len(anchor)
        best_start = -1
        for i in range(0, n - a + 1):
            # allow small mismatches: require >=3 of the anchor tokens to match
            # in order at this position
            matches = sum(1 for k in range(a) if page_tokens[i + k] == anchor[k])
            if matches >= max(3, a - 1):
                best_start = i
                break

        if best_start < 0:
            return None

        # how many page words does this clause span? estimate by clause length
        span = min(len(clause_tokens) + 4, n - best_start)
        run = words[best_start: best_start + span]
        x0 = min(w[0] for w in run)
        y0 = min(w[1] for w in run)
        x1 = max(w[2] for w in run)
        y1 = max(w[3] for w in run)
        # small padding so search_for clip doesn't cut edge glyphs
        return pymupdf.Rect(x0 - 1, y0 - 1, x1 + 1, y1 + 1)

    # -- searching --------------------------------------------------------
    def _search_in_clip(self, page_index: int, phrase: str,
                        clip: Optional["pymupdf.Rect"]) -> List[List[float]]:
        """Search a phrase on a page, optionally restricted to a clip rect.
        Returns bboxes of matches that fall inside the clip (if given)."""
        page = self.doc.load_page(page_index)
        for variant in _clean_variants(phrase):
            try:
                rects = page.search_for(variant, quads=False, clip=clip)
            except TypeError:
                # older signature without clip: search then filter manually
                try:
                    rects = page.search_for(variant, quads=False)
                    if clip is not None:
                        rects = [r for r in rects if clip.intersects(r)]
                except Exception:
                    rects = []
            except Exception:
                rects = []
            if rects:
                return [_rect_to_bbox(r) for r in rects]
        return []

    def _search_multiline(self, page_index: int, phrase: str,
                          clip: Optional["pymupdf.Rect"]) -> List[List[float]]:
        """Locate a phrase that WRAPS ACROSS LINES by matching its word
        sequence against the page words, then return ONE bbox per line the
        phrase spans (so the highlight follows the text across the wrap).

        page.search_for() often fails on long spans that break across lines
        (e.g. a whole consequence sentence). This recovers those cases while
        staying constrained to the clause region for accuracy.
        """
        words = self._page_words(page_index)
        if not words:
            return []

        if clip is not None:
            idx_words = [(i, w) for i, w in enumerate(words)
                         if clip.intersects(pymupdf.Rect(w[0], w[1], w[2], w[3]))]
        else:
            idx_words = list(enumerate(words))
        if not idx_words:
            return []

        page_tokens = [_norm_word(w[4]) for (_, w) in idx_words]
        target = [_norm_word(t) for t in phrase.split()]
        target = [t for t in target if t]
        if len(target) < 2:
            return []

        n, m = len(page_tokens), len(target)
        # find the best contiguous run allowing a few token mismatches
        best_start = -1
        for i in range(0, n - m + 1):
            matches = sum(1 for k in range(m) if page_tokens[i + k] == target[k])
            if matches >= max(2, int(0.75 * m)):
                best_start = i
                break
        if best_start < 0:
            return []

        run = [idx_words[best_start + k][1] for k in range(m)]
        # group the run's words into lines (by y position) -> one box per line
        lines: Dict[int, List] = {}
        for w in run:
            key = int(round((w[1] + w[3]) / 2 / 3.0))  # ~line bucket
            lines.setdefault(key, []).append(w)

        boxes: List[List[float]] = []
        for _, lw in sorted(lines.items()):
            x0 = min(w[0] for w in lw)
            y0 = min(w[1] for w in lw)
            x1 = max(w[2] for w in lw)
            y1 = max(w[3] for w in lw)
            boxes.append([round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)])
        return boxes

    def _page_range(self, clause: Dict) -> List[int]:
        ps = clause.get("page_start")
        pe = clause.get("page_end", ps)
        if ps is None:
            return list(range(self._page_count))
        start = max(0, ps - 1)
        end = min(self._page_count - 1, (pe or ps) - 1)
        return list(range(start, end + 1))

    # -- public -----------------------------------------------------------
    def map_phrase(self, phrase: str, clause: Dict) -> Dict:
        """Map a phrase to PDF location(s), constrained to the clause region."""
        result = {"phrase": phrase, "locations": [], "mapping_status": "failed"}
        if not phrase or not phrase.strip():
            return result

        clause_text = clause.get("clause_text", "")

        for pidx in self._page_range(clause):
            # 1) locate the clause region on this page and search within it
            region = self._find_clause_region(pidx, clause_text)
            if region is not None:
                boxes = self._search_in_clip(pidx, phrase, region)
                if boxes:
                    result["locations"] = [{"page": pidx + 1, "bbox": b} for b in boxes]
                    result["mapping_status"] = "mapped"
                    return result
                # 1b) long / line-wrapping phrase: map its word sequence and
                #     return one box per line it spans (still region-constrained)
                boxes = self._search_multiline(pidx, phrase, region)
                if boxes:
                    result["locations"] = [{"page": pidx + 1, "bbox": b} for b in boxes]
                    result["mapping_status"] = "mapped_multiline"
                    return result

        # 2) region not found on any clause page -> whole-page CONTIGUOUS search
        #    (accept only if the phrase is multi-character / specific enough)
        for pidx in self._page_range(clause):
            boxes = self._search_in_clip(pidx, phrase, None)
            if boxes:
                # only take the first match to avoid highlighting every
                # occurrence across the page
                result["locations"] = [{"page": pidx + 1, "bbox": boxes[0]}]
                result["mapping_status"] = "mapped_page"
                return result

        # 3) optional document-wide fallback (off by default)
        if self.allow_document_fallback:
            for pidx in range(self._page_count):
                boxes = self._search_in_clip(pidx, phrase, None)
                if boxes:
                    result["locations"] = [{"page": pidx + 1, "bbox": boxes[0]}]
                    result["mapping_status"] = "mapped_fallback"
                    return result

        logger.info("Phrase not mapped: %r (clause %s, pages %s-%s)",
                    phrase, clause.get("clause_id"),
                    clause.get("page_start"), clause.get("page_end"))
        return result

    def map_clause_phrases(self, phrases: List[str], clause: Dict) -> List[Dict]:
        """Map every phrase for a clause. Failures are recorded, never raised."""
        mapped: List[Dict] = []
        for ph in phrases:
            try:
                mapped.append(self.map_phrase(ph, clause))
            except Exception as e:
                logger.warning("Unexpected mapping error for %r: %s", ph, e)
                mapped.append({"phrase": ph, "locations": [],
                               "mapping_status": "failed"})
        return mapped
