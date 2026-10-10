"""
Phase 9 verification: coordinate mapping of important phrases back to the PDF.
Runs extract -> preprocess -> segment -> phrases -> map, reporting success rate.
"""
import os
import sys

import pymupdf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml.preprocessing.pdf_extractor import extract_pdf
from ml.preprocessing.text_preprocessor import preprocess_document
from ml.clause_segmentation.clause_segmenter import segment_document
from ml.importance.phrase_extractor import PhraseExtractor
from ml.coordinate_mapping.coordinate_mapper import CoordinateMapper

RAW = os.path.join(PROJECT_ROOT, "data", "raw")
TEST_FILE = "Christ_University_Anti_Ragging_Regulations_Cleaned.pdf"


def run():
    print("=" * 68)
    print("PHASE 9: Coordinate Mapping (phrase -> PDF bbox)")
    print("=" * 68)

    path = os.path.join(RAW, TEST_FILE)
    extracted = extract_pdf(path)
    pp = preprocess_document(extracted)
    clauses = segment_document(pp)
    extractor = PhraseExtractor()

    doc = pymupdf.open(path)
    mapper = CoordinateMapper(doc)

    total = 0
    mapped = 0
    shown = 0

    # Process a subset of substantive clauses
    subset = [c for c in clauses if len(c["clause_text"]) > 80][:15]
    for cl in subset:
        cands = extractor.extract(cl["clause_text"])
        phrases = [c["phrase"] for c in cands[:5]]  # top few per clause
        results = mapper.map_clause_phrases(phrases, cl)
        for r in results:
            total += 1
            if r["mapping_status"].startswith("mapped"):
                mapped += 1
                if shown < 10:
                    loc = r["locations"][0]
                    print(f"  [{cl['clause_id']} p{loc['page']}] {r['phrase'][:34]:<34} "
                          f"-> bbox={loc['bbox']}")
                    shown += 1

    doc.close()
    rate = (mapped / total * 100) if total else 0
    print(f"\n  Mapped {mapped}/{total} phrases  ({rate:.1f}% success)")
    print("=" * 68)
    print("[PHASE 9 CHECK COMPLETE]")


if __name__ == "__main__":
    run()
