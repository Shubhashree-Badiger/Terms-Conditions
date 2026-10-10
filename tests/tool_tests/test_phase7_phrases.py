"""
Phase 7 verification: important phrase extraction on real clauses.
Runs extract -> preprocess -> segment -> phrase-extract and shows candidates.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml.preprocessing.pdf_extractor import extract_pdf
from ml.preprocessing.text_preprocessor import preprocess_document
from ml.clause_segmentation.clause_segmenter import segment_document
from ml.importance.phrase_extractor import PhraseExtractor

RAW = os.path.join(PROJECT_ROOT, "data", "raw")
TEST_FILE = "2024-Academic-Regulations-Updated-1.pdf"


def run():
    print("=" * 66)
    print("PHASE 7: Important Phrase Extraction (YAKE + Domain Lexicon)")
    print("=" * 66)

    extractor = PhraseExtractor()
    print(f"Lexicon categories loaded: {len(extractor.lexicon.get('categories', {}))}")

    # Standalone sample first
    sample = ("Students failing to maintain the minimum attendance requirement of 75% "
              "shall not be permitted to appear for the end semester examination and "
              "the fee paid is non-refundable.")
    print("\n--- Standalone sample clause ---")
    print(sample)
    print("\nCandidates:")
    for c in extractor.extract(sample):
        src = "+".join(c["sources"])
        cat = c["category"] or "-"
        ys = f"{c['yake_score']:.4f}" if c["yake_score"] is not None else "  -   "
        print(f"   [{src:<12}] cat={cat:<12} yake={ys}  {c['phrase']}")

    # Real clauses from the PDF
    extracted = extract_pdf(os.path.join(RAW, TEST_FILE))
    pp = preprocess_document(extracted)
    clauses = segment_document(pp)

    # Pick a few substantive clauses (longer text, numbered)
    substantive = [c for c in clauses
                   if c["clause_number"] and len(c["clause_text"]) > 120][:3]

    print("\n--- Real clauses from PDF ---")
    for cl in substantive:
        print(f"\n[{cl['clause_id']} | {cl['clause_number']} | p{cl['page_start']}]")
        print(f"  TEXT: {cl['clause_text'][:150]}...")
        cands = extractor.extract(cl["clause_text"])
        print(f"  {len(cands)} candidates (top 6):")
        for c in cands[:6]:
            src = "+".join(c["sources"])
            cat = c["category"] or "-"
            print(f"     [{src:<12}] cat={cat:<12} {c['phrase']}")

    print("\n" + "=" * 66)
    print("[PHASE 7 CHECK COMPLETE]")


if __name__ == "__main__":
    run()
