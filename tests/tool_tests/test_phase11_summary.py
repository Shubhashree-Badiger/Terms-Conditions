"""
Phase 11 verification: one-line clause summarization.
Shows original vs summary for crafted samples and real clauses.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml.preprocessing.pdf_extractor import extract_pdf
from ml.preprocessing.text_preprocessor import preprocess_document
from ml.clause_segmentation.clause_segmenter import segment_document
from ml.summarization.summarizer import ClauseSummarizer

RAW = os.path.join(PROJECT_ROOT, "data", "raw")
TEST_FILE = "2024-Academic-Regulations-Updated-1.pdf"


def run():
    print("=" * 72)
    print("PHASE 11: One-Line Summarization (T5-small + faithful fallback)")
    print("=" * 72)

    print("Loading T5-small (first run may download ~240MB)...")
    summ = ClauseSummarizer(use_model=True)
    print(f"T5 model active: {summ.use_model}\n")

    samples = [
        "Students failing to maintain the minimum attendance requirement of 75% "
        "shall not be permitted to appear for the end semester examination.",
        "The fee once paid at the time of admission is non-refundable under any "
        "circumstances, and a late payment attracts a penalty of Rs. 500 per day.",
        "Any student found indulging in ragging shall be liable for suspension or "
        "expulsion from the institution as per the anti-ragging regulations.",
        "Students must carry their identity card while on campus.",
    ]

    print("--- Crafted samples ---")
    for s in samples:
        out = summ.summarize(s)
        print(f"\nORIGINAL : {s}")
        print(f"SUMMARY  : {out}")

    # Real clauses
    print("\n--- Real clauses from PDF ---")
    pp = preprocess_document(extract_pdf(os.path.join(RAW, TEST_FILE)))
    clauses = segment_document(pp)
    substantive = [c for c in clauses
                   if c["clause_number"] and 120 < len(c["clause_text"]) < 400][:4]
    for cl in substantive:
        out = summ.summarize(cl["clause_text"])
        print(f"\n[{cl['clause_id']} | {cl['clause_number']}]")
        print(f"ORIGINAL : {cl['clause_text'][:180]}")
        print(f"SUMMARY  : {out}")

    print("\n" + "=" * 72)
    print("[PHASE 11 CHECK COMPLETE]")


if __name__ == "__main__":
    run()
