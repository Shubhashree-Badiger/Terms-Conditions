# Error Analysis Report (Phase 18)

Generated from actual results of the batch run (48 real institutional PDFs)
and the human-labeled evaluation. All numbers are measured, not estimated.

## 1. Corpus Overview

| Metric | Value |
|--------|-------|
| Documents processed | 48 |
| Successfully analyzed | 43 |
| Scanned (flagged, not analyzed) | 5 |
| Hard errors / crashes | 0 |
| Overall mapping success rate | 98.6% (1080 / 1095 phrases) |
| Avg time / document | 28.0 s |
| Avg time / clause | 0.11 s |

No document crashed the pipeline — the per-clause error isolation held across
1,129 pages and 12,055 clauses.

## 2. Error Categories

### E1 — Scanned / image-only PDFs (5 docs)
**Files:** 1575.pdf, CU Regulation Student Code of Conduct (1).pdf,
Guidelines.pdf, socMed_Disciplinary.pdf, socMed_Guidelines.pdf
**Impact:** These have no extractable text, so no analysis is possible.
**Status:** Handled gracefully — detected and returned "This PDF appears to
contain scanned images. OCR support is required." (never crashes).
**Recommendation (future):** Add an OCR fallback (Tesseract / PyMuPDF OCR) to
process image-only PDFs. Out of scope for the current text-based pipeline.

### E2 — Coordinate mapping failures (15 of 1,095 phrases = 1.37%)
**Affected docs:** 2013_UG_Ordinances_IITR, Academic_Programm-2012,
UGAdmManual2024, UG_Ordinance_BoG_2025.
**Root cause (confirmed by drill-down):** The failing phrases are multi-word
spans whose words are **non-contiguous in the PDF layout** — they come from
table columns or reflowed lines (e.g. "Games Activities Athletics",
"Centre Academic Committee", "Commencement five-year Integrated"). PyMuPDF's
`search_for()` looks for a contiguous string and cannot match words that are
visually separated across columns.
**Status:** Handled gracefully — `mapping_status = "failed"`, logged, the
document continues. The phrase still appears in the JSON without coordinates.
**Recommendation:** Add a word-coordinate fallback that assembles a bounding
box from the extractor's per-word list when `search_for()` misses, and/or skip
mapping for phrases that YAKE assembled from non-adjacent table cells.

### E3 — Important-phrase recall gap (measured)
**Measured (human-labeled, tuned settings):** Precision 0.50, Recall 0.31,
F1 0.38 (tp=10, fp=10, fn=22 over 15 clauses).
**Root cause:** YAKE + the domain lexicon favor short keyword phrases. Many
gold phrases are **descriptive multi-word spans** ("criminal proceeding",
"financial extortion", "generating a sense of shame", "regular academic")
that are never generated as candidates, so they can never be scored or
selected. Threshold tuning (Phase 17) raised F1 from 0.29 to 0.38 but hit a
ceiling because the limitation is candidate generation, not filtering.
**Recommendation:** (a) Expand the education/risk lexicons with more legal and
anti-ragging terminology; (b) add spaCy noun-chunk candidates so descriptive
noun phrases are proposed; (c) allow 4-5 word n-grams in YAKE for policy text.

### E4 — Table-of-contents / front-matter as clauses
**Impact:** On documents with a TOC, lines like "1.0 Title & Commencement 04"
are segmented as clauses (with a trailing page number), inflating clause
counts and adding low-value entries.
**Status:** Handled gracefully — these produce no/low-importance phrases and do
not break anything.
**Recommendation:** Detect TOC pages (dot-leaders, lines ending in a page
number) and skip them, or drop very short clauses that end in a bare number.

### E5 — Summaries echo the clause (limited compression)
**Impact:** One-line summaries are always faithful (no hallucination) but for
single-sentence clauses the "summary" can equal the clause because there is
nothing to condense.
**Root cause:** Base t5-small is not fine-tuned for summarization; the
extractive fallback returns the most informative sentence.
**Recommendation:** For genuinely abstractive one-liners, swap t5-small for a
summarization-tuned checkpoint (e.g. flan-t5-base or a fine-tuned model). The
pipeline interface stays the same.

## 3. Priority of Fixes

| Priority | Item | Effort | Payoff |
|----------|------|--------|--------|
| High | E3 candidate generation (noun-chunks + lexicon) | Medium | Directly raises recall/F1 |
| Medium | E2 word-coordinate mapping fallback | Medium | Recovers ~1.4% of highlights |
| Medium | E4 TOC filtering | Low | Cleaner clause list |
| Low | E1 OCR fallback | High | Enables ~10% of the corpus |
| Low | E5 fine-tuned summarizer | High | Nicer abstractive summaries |

## 4. What Already Works Well

- Zero crashes across 48 diverse documents (robust error isolation).
- 98.6% coordinate-mapping success on text-based PDFs.
- High precision (0.83 at conservative settings) — flagged phrases are usually
  correct.
- Scanned PDFs detected and reported instead of failing silently.
- Faithful summaries (numbers, percentages, penalties preserved; no
  hallucination).
