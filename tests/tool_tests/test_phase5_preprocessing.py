"""
Phase 5 verification: preprocessing on real PDFs.
Shows detected headers/footers and a before/after text sample.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml.preprocessing.pdf_extractor import extract_pdf
from ml.preprocessing.text_preprocessor import preprocess_document

RAW = os.path.join(PROJECT_ROOT, "data", "raw")
TEST_FILE = "2024-Academic-Regulations-Updated-1.pdf"


def run():
    print("=" * 64)
    print("PHASE 5: Document Preprocessing")
    print("=" * 64)

    path = os.path.join(RAW, TEST_FILE)
    extracted = extract_pdf(path)
    print(f"File: {TEST_FILE}  pages={extracted.total_pages}")

    pp = preprocess_document(extracted)

    print(f"\nDetected repeated header/footer patterns ({len(pp.removed_lines)}):")
    for r in pp.removed_lines[:10]:
        print(f"   - {r!r}")

    # Show before/after for a sample page
    page = pp.pages[4]  # 5th page
    print(f"\n--- Page {page.page_number} ORIGINAL (first 300 chars) ---")
    print(page.original_text[:300].strip())
    print(f"\n--- Page {page.page_number} NORMALIZED (first 300 chars) ---")
    print(page.normalized_text[:300].strip())

    print(f"\nFull normalized length: {len(pp.normalized_text)} chars")
    print(f"Full original length:   {len(pp.original_text)} chars")
    print("\n[PHASE 5 CHECK COMPLETE]")


if __name__ == "__main__":
    run()
