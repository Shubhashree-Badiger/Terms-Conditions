"""
Phase 10 verification + FIRST MAJOR MILESTONE.
Full chain: PDF -> extract -> preprocess -> segment -> phrases -> score ->
map -> highlight -> annotated PDF.

Produces an annotated PDF in outputs/annotated_pdfs/ that can be opened and
inspected visually.
"""
import os
import sys
import time

import pymupdf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml.preprocessing.pdf_extractor import extract_pdf, is_probably_scanned
from ml.preprocessing.text_preprocessor import preprocess_document
from ml.clause_segmentation.clause_segmenter import segment_document
from ml.importance.phrase_extractor import PhraseExtractor
from ml.importance.importance_scorer import ImportanceScorer
from ml.coordinate_mapping.coordinate_mapper import CoordinateMapper
from ml.annotation.pdf_highlighter import generate_annotated_pdf

RAW = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "annotated_pdfs")
TEST_FILE = "Christ_University_Anti_Ragging_Regulations_Cleaned.pdf"

# Only highlight reasonably important phrases to keep the output meaningful
MIN_SCORE = 0.45
TOP_PHRASES_PER_CLAUSE = 4


def run():
    print("=" * 70)
    print("PHASE 10: PDF Highlighting  (FIRST MAJOR MILESTONE - full pipeline)")
    print("=" * 70)

    path = os.path.join(RAW, TEST_FILE)
    t0 = time.time()

    # 1. extract
    extracted = extract_pdf(path)
    if extracted.error:
        print("ERROR:", extracted.error); return
    if is_probably_scanned(extracted):
        print("Scanned PDF - OCR required."); return
    print(f"1. Extracted: {extracted.total_pages} pages, {extracted.total_words} words")

    # 2. preprocess
    pp = preprocess_document(extracted)
    print(f"2. Preprocessed (removed {len(pp.removed_lines)} header/footer patterns)")

    # 3. segment
    clauses = segment_document(pp)
    print(f"3. Segmented: {len(clauses)} clauses")

    # 4/5. phrases + scoring
    extractor = PhraseExtractor()
    scorer = ImportanceScorer(use_semantic=False)  # fast; semantic optional
    print("4/5. Extracting + scoring phrases...")

    # 6. map + collect highlight records
    doc = pymupdf.open(path)
    mapper = CoordinateMapper(doc)

    all_mappings = []
    kept = 0
    for cl in clauses:
        cands = extractor.extract(cl["clause_text"])
        scored = scorer.score_candidates(cands, cl["clause_text"])
        # keep only sufficiently important phrases
        top = [s for s in scored if s["score"] >= MIN_SCORE][:TOP_PHRASES_PER_CLAUSE]
        for s in top:
            m = mapper.map_phrase(s["phrase"], cl)
            m["level"] = s["level"]
            m["score"] = s["score"]
            all_mappings.append(m)
            if m["mapping_status"].startswith("mapped"):
                kept += 1
    doc.close()

    mapped_ct = sum(1 for m in all_mappings if m["mapping_status"].startswith("mapped"))
    print(f"6. Phrases to highlight: {len(all_mappings)}  (mapped: {mapped_ct})")

    # 7. highlight -> annotated PDF
    out_path = os.path.join(OUT_DIR, "milestone_annotated.pdf")
    generate_annotated_pdf(path, all_mappings, out_path)
    elapsed = time.time() - t0

    size_kb = os.path.getsize(out_path) / 1024
    print(f"7. Annotated PDF: {out_path} ({size_kb:.0f} KB)")
    print(f"\nTotal pipeline time: {elapsed:.1f}s")

    # verify annotations exist and text is still selectable
    ann = pymupdf.open(out_path)
    annot_count = sum(1 for pg in ann for _ in (pg.annots() or []))
    sample_text = ann.load_page(0).get_text("text")[:60].replace("\n", " ")
    ann.close()
    print(f"Annotations in output: {annot_count}")
    print(f"Text still selectable (page 1 sample): {sample_text!r}")

    print("\n" + "=" * 70)
    print("[MILESTONE COMPLETE] Open the annotated PDF to inspect highlights.")


if __name__ == "__main__":
    run()
