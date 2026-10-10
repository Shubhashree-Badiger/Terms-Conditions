"""
Complete Analysis Pipeline (Phase 12)
=====================================
Orchestrates the full backend analysis for one PDF, WITHOUT any frontend:

    PDF
     -> extract (text + word coordinates)
     -> preprocess (clean, keep original + normalized)
     -> clause segmentation
     -> FOR EACH CLAUSE:
            YAKE + domain lexicon  (phrase extraction)
            importance scoring     (+ risk)
            coordinate mapping     (phrase -> pdf bbox)
            one-line summary
     -> annotated PDF (highlight mapped phrases)
     -> structured result dict (Phase 13 JSON shape)

Design rules honored:
  * ML models (SBERT, T5) are loaded ONCE via the shared components, not per
    clause. The AnalysisPipeline builds them once and reuses them.
  * If one clause fails, log and continue; never crash the whole document.
  * The original PDF is never modified; the annotated copy is written separately.

Phrase selection intelligence
-----------------------------
Which words get highlighted is decided by the ImportanceScorer, whose PRIMARY
signal is CLAUSE CENTRALITY - cosine(phrase, whole-clause) via MiniLM - so the
highlights land on the words the clause is actually about, not on generic YAKE
n-grams. This requires use_semantic=True (the backend enables it).

Public API:
    class AnalysisPipeline:
        __init__(use_semantic=True, use_t5=False)   # backend uses semantic=True
        analyze_document(pdf_path, document_id=None,
                         annotated_out=None, min_score=0.30,
                         top_phrases_per_clause=10) -> dict
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

import pymupdf

from ml.preprocessing.pdf_extractor import extract_pdf, is_probably_scanned
from ml.preprocessing.emphasis_extractor import extract_emphasized_spans
from ml.preprocessing.text_preprocessor import preprocess_document
from ml.clause_segmentation.clause_segmenter import segment_document
from ml.importance.phrase_extractor import PhraseExtractor
from ml.importance.importance_scorer import ImportanceScorer
from ml.coordinate_mapping.coordinate_mapper import CoordinateMapper
from ml.annotation.pdf_highlighter import generate_annotated_pdf
from ml.summarization.summarizer import ClauseSummarizer
from ml.importance.phrase_extractor import trim_phrase

import re as _re

logger = logging.getLogger("analyzer.pipeline")

SCANNED_MESSAGE = ("This PDF appears to contain scanned images. "
                   "OCR support is required.")


class AnalysisPipeline:
    """Reusable analysis pipeline. Build once, analyze many documents."""

    def __init__(self, use_semantic: bool = True, use_t5: bool = False):
        # Models loaded ONCE here and reused across documents / clauses.
        # use_semantic defaults ON because the clause-centrality signal (the
        # primary driver of which words get highlighted) needs the SBERT model.
        logger.info("Initializing analysis pipeline (semantic=%s, t5=%s)...",
                    use_semantic, use_t5)
        self.phrase_extractor = PhraseExtractor()
        self.scorer = ImportanceScorer(use_semantic=use_semantic)
        self.summarizer = ClauseSummarizer(use_model=use_t5)
        logger.info("Pipeline ready.")

    def analyze_document(self, pdf_path: str, document_id: Optional[str] = None,
                         annotated_out: Optional[str] = None,
                         min_score: float = 0.30,
                         top_phrases_per_clause: int = 10,
                         progress_cb=None) -> Dict:
        """Analyze a single PDF and return a structured result dict.

        Args:
            pdf_path: path to the PDF to analyze.
            document_id: id to embed in the result (defaults to file stem).
            annotated_out: where to write the annotated PDF (optional).
            min_score: only phrases with importance >= this are highlighted.
            top_phrases_per_clause: cap on highlighted phrases per clause.

        Returns:
            result dict (see Phase 13 shape). On fatal error, result has
            "status": "error" and an "error" message. Scanned PDFs return
            "status": "scanned".
        """
        def _report(stage):
            if progress_cb:
                try:
                    progress_cb(stage)
                except Exception:
                    pass

        t0 = time.time()
        file_name = os.path.basename(pdf_path)
        document_id = document_id or os.path.splitext(file_name)[0]

        # --- 1. extract -------------------------------------------------
        _report(1)
        extracted = extract_pdf(pdf_path, document_id=document_id, file_name=file_name)
        if extracted.error:
            _report(-1)
            return {"document_id": document_id, "filename": file_name,
                    "status": "error", "error": extracted.error}
        if is_probably_scanned(extracted):
            return {"document_id": document_id, "filename": file_name,
                    "total_pages": extracted.total_pages,
                    "status": "scanned", "message": SCANNED_MESSAGE}

        # --- 2. preprocess ---------------------------------------------
        pp = preprocess_document(extracted)

        # --- 3. segment -------------------------------------------------
        _report(2)
        clauses = segment_document(pp)

        # --- 4-8 per clause --------------------------------------------
        _report(3)
        doc = pymupdf.open(pdf_path)
        mapper = CoordinateMapper(doc)

        # Original author emphasis (bold/italic) - a strong human signal of what
        # matters. Extracted once, then matched per clause below.
        try:
            emphasis_spans = extract_emphasized_spans(doc)
        except Exception as e:
            logger.debug("Emphasis extraction failed: %s", e)
            emphasis_spans = []
        emphasis_by_page: Dict[int, List[Dict]] = {}
        for sp in emphasis_spans:
            emphasis_by_page.setdefault(sp["page"], []).append(sp)

        clause_results: List[Dict] = []
        all_highlight_mappings: List[Dict] = []
        stats = {
            "total_phrases": 0, "mapped_phrases": 0, "failed_phrases": 0,
            "summary_failures": 0, "clause_errors": 0,
        }

        for clause in clauses:
            try:
                # emphasized author spans that fall on this clause's page(s)
                cl_emphasis = []
                for pg in range(clause.get("page_start") or 1,
                                 (clause.get("page_end") or clause.get("page_start") or 1) + 1):
                    cl_emphasis.extend(emphasis_by_page.get(pg, []))
                clause_result = self._analyze_clause(
                    clause, mapper, min_score, top_phrases_per_clause, stats,
                    all_highlight_mappings, emphasis_spans=cl_emphasis)
                clause_results.append(clause_result)
            except Exception as e:  # never crash the whole document
                stats["clause_errors"] += 1
                logger.warning("Clause %s failed: %s", clause.get("clause_id"), e)
                clause_results.append({
                    "clause_id": clause.get("clause_id"),
                    "clause_number": clause.get("clause_number", ""),
                    "page_start": clause.get("page_start"),
                    "page_end": clause.get("page_end"),
                    "clause_text": clause.get("clause_text", ""),
                    "summary": "",
                    "important_phrases": [],
                    "error": str(e),
                })

        doc.close()
        _report(5)  # summaries were produced within per-clause analysis

        # --- 9. annotated PDF ------------------------------------------
        annotated_path = None
        if annotated_out:
            _report(4)
            try:
                generate_annotated_pdf(pdf_path, all_highlight_mappings, annotated_out)
                annotated_path = annotated_out
            except Exception as e:
                logger.error("Annotation failed: %s", e)

        elapsed = round(time.time() - t0, 2)
        # Note: stage 6 (done) is intentionally NOT reported here. The caller
        # marks completion only AFTER the result JSON has been persisted, to
        # avoid a race where the client fetches /results before it exists.

        return {
            "document_id": document_id,
            "filename": file_name,
            "total_pages": extracted.total_pages,
            "total_clauses": len(clause_results),
            "status": "analyzed",
            "processing_seconds": elapsed,
            "stats": stats,
            "annotated_pdf": annotated_path,
            "clauses": clause_results,
        }

    # -- emphasis candidates ---------------------------------------------
    @staticmethod
    def _emphasis_candidates(clause_text: str, spans: List[Dict]) -> List[Dict]:
        """Turn the author's bold/italic spans that occur in this clause into
        high-priority candidate phrases (so original emphasis is prioritized).

        Only spans whose text actually appears in the clause are used, so page
        emphasis from other clauses on the same page is not misattributed.
        """
        if not clause_text or not spans:
            return []
        lower_clause = clause_text.lower()
        out: List[Dict] = []
        seen = set()
        for sp in spans:
            raw = (sp.get("text") or "").strip()
            phrase = trim_phrase(raw)
            if len(phrase) < 3:
                continue
            key = phrase.lower()
            if key in seen:
                continue
            # the emphasized text must appear in THIS clause
            if key not in lower_clause:
                # allow a looser match: most words of the span present in order
                words = [w for w in _re.sub(r"[^a-z0-9%\s]", " ", key).split() if w]
                if not words or sum(1 for w in words if w in lower_clause) < max(2, int(0.8 * len(words))):
                    continue
            seen.add(key)
            out.append({
                "phrase": phrase,
                "sources": ["emphasis"],
                "yake_score": None,
                "category": None,
                "constraint_type": "emphasis",
                # bold outranks italic; both get a strong boost + score floor
                "constraint_boost": 0.97 if sp.get("bold") else 0.9,
                "emphasis": True,
            })
        return out

    # -- per-clause helper -----------------------------------------------
    def _analyze_clause(self, clause: Dict, mapper: CoordinateMapper,
                        min_score: float, top_k: int, stats: Dict,
                        highlight_sink: List[Dict],
                        emphasis_spans: Optional[List[Dict]] = None) -> Dict:
        clause_text = clause.get("clause_text", "")

        # phrase extraction + scoring
        candidates = self.phrase_extractor.extract(clause_text)

        # Inject the author's OWN emphasis (bold/italic) that appears in this
        # clause as high-priority candidates so it is prioritized, not skipped.
        emph_cands = self._emphasis_candidates(clause_text, emphasis_spans or [])
        if emph_cands:
            candidates = emph_cands + candidates

        scored = self.scorer.score_candidates(candidates, clause_text)

        # keep the important ones for highlighting
        important = [s for s in scored if s["score"] >= min_score][:top_k]

        important_phrases = []
        for s in important:
            mapping = mapper.map_phrase(s["phrase"], clause)
            stats["total_phrases"] += 1
            if mapping["mapping_status"].startswith("mapped"):
                stats["mapped_phrases"] += 1
            else:
                stats["failed_phrases"] += 1

            # record for highlighting
            highlight_sink.append({
                "phrase": s["phrase"],
                "level": s["level"],
                "mapping_status": mapping["mapping_status"],
                "locations": mapping["locations"],
            })

            important_phrases.append({
                "text": s["phrase"],
                "score": s["score"],
                "level": s["level"],
                "category": s.get("category"),
                "constraint_type": s.get("constraint_type"),
                "risk_type": s.get("risk_type"),
                "risk_level": s.get("risk_level"),
                "mapping_status": mapping["mapping_status"],
                "locations": mapping["locations"],
            })

        # one-line summary
        try:
            summary = self.summarizer.summarize(clause_text)
        except Exception as e:
            stats["summary_failures"] += 1
            logger.debug("Summary failed for %s: %s", clause.get("clause_id"), e)
            summary = ""

        return {
            "clause_id": clause.get("clause_id"),
            "clause_number": clause.get("clause_number", ""),
            "page_start": clause.get("page_start"),
            "page_end": clause.get("page_end"),
            "clause_text": clause_text,
            "summary": summary,
            "important_phrases": important_phrases,
        }
