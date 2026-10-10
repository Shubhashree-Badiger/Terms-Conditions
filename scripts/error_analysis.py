"""
Phase 18: Error Analysis.
=========================
Mines the ACTUAL results from the batch run (data/metadata/batch_report.csv,
batch_summary.json) and the evaluation report to categorize real error types,
then writes a structured error report.

It also drills into specific coordinate-mapping failures on a chosen document
to identify WHY phrases failed to map (multi-line, reflow, glyph issues, etc.).

Outputs:
    data/metadata/error_analysis.json     (structured findings)
Console: human-readable error report.
"""
import csv
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

META = os.path.join(PROJECT_ROOT, "data", "metadata")
RAW = os.path.join(PROJECT_ROOT, "data", "raw")
CSV_PATH = os.path.join(META, "batch_report.csv")
SUMMARY_PATH = os.path.join(META, "batch_summary.json")
OUT_PATH = os.path.join(META, "error_analysis.json")


def load_batch():
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        summary = json.load(f)
    return rows, summary


def drill_mapping_failures(filename, document_id):
    """Re-run mapping on one doc and return the phrases that failed + why."""
    import pymupdf
    from ml.preprocessing.pdf_extractor import extract_pdf
    from ml.preprocessing.text_preprocessor import preprocess_document
    from ml.clause_segmentation.clause_segmenter import segment_document
    from ml.importance.phrase_extractor import PhraseExtractor
    from ml.importance.importance_scorer import ImportanceScorer
    from ml.coordinate_mapping.coordinate_mapper import CoordinateMapper

    path = os.path.join(RAW, filename)
    pp = preprocess_document(extract_pdf(path))
    clauses = segment_document(pp)
    ex = PhraseExtractor()
    scorer = ImportanceScorer(use_semantic=False)
    doc = pymupdf.open(path)
    mapper = CoordinateMapper(doc)

    failures = []
    for c in clauses:
        cands = ex.extract(c["clause_text"])
        scored = scorer.score_candidates(cands, c["clause_text"])
        top = [s for s in scored if s["score"] >= 0.30][:10]
        for s in top:
            m = mapper.map_phrase(s["phrase"], c)
            if not m["mapping_status"].startswith("mapped"):
                # classify the likely cause
                phrase = s["phrase"]
                cause = "unknown"
                if len(phrase.split()) >= 4:
                    cause = "long multi-word phrase (likely line-wrapped/reflowed)"
                elif any(ch in phrase for ch in "-/()"):
                    cause = "contains punctuation/hyphen (tokenization mismatch)"
                elif phrase.lower() != phrase and " " in phrase:
                    cause = "mixed case multi-word (spacing/case mismatch)"
                else:
                    cause = "not found on clause page (segmentation/page scope)"
                failures.append({"clause_id": c["clause_id"], "phrase": phrase,
                                 "page": c["page_start"], "likely_cause": cause})
    doc.close()
    return failures


def main():
    print("=" * 74)
    print("PHASE 18: ERROR ANALYSIS")
    print("=" * 74)

    rows, summary = load_batch()

    scanned = [r for r in rows if r["status"] == "scanned"]
    errored = [r for r in rows if r["status"] == "error"]
    mapping_fail_docs = [r for r in rows
                         if r["status"] == "analyzed" and int(r["failed"]) > 0]

    findings = {
        "corpus": {
            "total_docs": summary["total_docs"],
            "analyzed": summary["analyzed"],
            "scanned": summary["scanned"],
            "errors": summary["error"],
            "overall_mapping_success_rate": summary["mapping_success_rate"],
        },
        "error_categories": [],
    }

    # --- Category 1: Scanned / image-only PDFs ---
    findings["error_categories"].append({
        "id": "E1",
        "name": "Scanned / image-only PDFs (no extractable text)",
        "count": len(scanned),
        "examples": [r["filename"] for r in scanned],
        "impact": "Document cannot be analyzed at all; returns 'OCR required'.",
        "handled_gracefully": True,
        "recommendation": "Integrate an OCR fallback (e.g. Tesseract via pytesseract "
                          "or PyMuPDF OCR) for image-only PDFs. Currently flagged, not processed.",
    })

    # --- Category 2: Coordinate mapping failures ---
    total_ph = summary["total_phrases"]
    failed_ph = summary["failed_phrases"]
    findings["error_categories"].append({
        "id": "E2",
        "name": "Coordinate mapping failures",
        "count": failed_ph,
        "rate": f"{round(failed_ph/total_ph*100,2)}% of {total_ph} phrases",
        "affected_docs": [r["filename"] for r in mapping_fail_docs],
        "impact": "Phrase identified but cannot be highlighted in the PDF.",
        "handled_gracefully": True,
        "recommendation": "Improve multi-line / reflowed phrase search; add a "
                          "word-coordinate fallback that assembles bboxes from the "
                          "extractor's word list when search_for() misses.",
    })

    # --- Category 3: Phrase recall gap (from evaluation) ---
    eval_report = None
    ev_path = os.path.join(META, "evaluation_report.json")
    if os.path.exists(ev_path):
        with open(ev_path, encoding="utf-8") as f:
            eval_report = json.load(f)
    prf = (eval_report or {}).get("phrase_prf")
    findings["error_categories"].append({
        "id": "E3",
        "name": "Important-phrase recall gap",
        "measured": prf,
        "impact": "System misses descriptive multi-word gold phrases.",
        "root_cause": "YAKE + domain lexicon favor short keyword phrases; long "
                     "descriptive spans (e.g. 'criminal proceeding', 'financial "
                     "extortion') are never generated as candidates.",
        "handled_gracefully": True,
        "recommendation": "Expand education/risk lexicons (esp. legal/anti-ragging "
                          "terms); add spaCy noun-chunk candidates; allow 4-5 word "
                          "n-grams in YAKE for policy language.",
    })

    # --- Category 4: TOC / front-matter lines segmented as clauses ---
    findings["error_categories"].append({
        "id": "E4",
        "name": "Table-of-contents / front-matter captured as clauses",
        "impact": "Noisy short 'clauses' (e.g. '1.0 Title & Commencement 04') "
                  "with trailing page numbers inflate clause counts.",
        "handled_gracefully": True,
        "recommendation": "Detect and skip TOC pages (lines ending in a page number, "
                          "dot-leaders); or drop clauses shorter than N chars with a "
                          "trailing number pattern.",
    })

    # --- Category 5: Weak abstractive summaries (echo) ---
    findings["error_categories"].append({
        "id": "E5",
        "name": "Summaries echo the clause (limited compression)",
        "impact": "One-line summaries are faithful but not always shorter than "
                  "the source for single-sentence clauses.",
        "root_cause": "Base t5-small is not fine-tuned; extractive fallback returns "
                     "the key sentence which for atomic clauses equals the clause.",
        "handled_gracefully": True,
        "recommendation": "Swap t5-small for a summarization-tuned checkpoint "
                          "(e.g. flan-t5-base or a fine-tuned model) if abstractive "
                          "compression is required.",
    })

    # --- drill into a real mapping-failure doc ---
    drill = []
    if mapping_fail_docs:
        target = mapping_fail_docs[0]["filename"]
        print(f"Drilling into mapping failures for: {target} ...")
        try:
            drill = drill_mapping_failures(target, mapping_fail_docs[0]["document_id"])
        except Exception as e:
            drill = [{"error": str(e)}]
    findings["mapping_failure_examples"] = drill[:15]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(findings, f, ensure_ascii=False, indent=2)

    # ---- console report ----
    print(f"\nCorpus: {findings['corpus']}")
    print("\nERROR CATEGORIES:")
    for cat in findings["error_categories"]:
        print(f"\n[{cat['id']}] {cat['name']}")
        for k in ("count", "rate", "measured", "impact", "root_cause",
                  "affected_docs", "examples", "recommendation"):
            if k in cat and cat[k]:
                print(f"    {k}: {cat[k]}")

    if drill:
        print("\nSAMPLE MAPPING FAILURES (drill-down):")
        for d in drill[:8]:
            if "error" in d:
                print("   drill error:", d["error"]); break
            print(f"   [{d['clause_id']} p{d['page']}] {d['phrase'][:40]!r} "
                  f"-> {d['likely_cause']}")

    print(f"\nSaved: {OUT_PATH}")
    print("=" * 74)
    print("[PHASE 18 CHECK COMPLETE]")


if __name__ == "__main__":
    main()
