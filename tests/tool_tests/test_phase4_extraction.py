"""
Phase 4 verification: PDF extraction with coordinates.
Runs the extractor on a few real PDFs from data/raw and prints inspectable
output: page count, word count, sample word records with bboxes, and a
scanned-PDF check.
"""
import os
import sys

# Make the project root importable so `ml` package resolves
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml.preprocessing.pdf_extractor import extract_pdf, reconstruct_text, is_probably_scanned

RAW = os.path.join(PROJECT_ROOT, "data", "raw")

# A text-based PDF (good) and a scanned PDF (should be flagged)
TEST_FILES = [
    "2024-Academic-Regulations-Updated-1.pdf",   # text-based, 45 pages
    "Christ_University_Anti_Ragging_Regulations_Cleaned.pdf",  # text-based, small
    "1575.pdf",                                  # scanned (0 text expected)
]


def run():
    print("=" * 64)
    print("PHASE 4: PDF Extraction with Coordinates")
    print("=" * 64)

    for fname in TEST_FILES:
        path = os.path.join(RAW, fname)
        if not os.path.exists(path):
            print(f"\n[SKIP] {fname} not found")
            continue

        doc = extract_pdf(path)
        print(f"\n### {fname}")
        if doc.error:
            print(f"  ERROR: {doc.error}")
            continue

        print(f"  pages={doc.total_pages}  total_words={doc.total_words}  "
              f"method={doc.extraction_method}")
        scanned = is_probably_scanned(doc)
        print(f"  scanned/OCR-needed? {scanned}")

        if scanned:
            print("  -> would return: 'This PDF appears to contain scanned "
                  "images. OCR support is required.'")
            continue

        # Pick a page with a decent number of words
        page = max(doc.pages, key=lambda p: len(p.words))
        print(f"  Sample page {page.page_number}: {len(page.words)} words, "
              f"size {page.width} x {page.height}")
        print("  First 5 word records:")
        for w in page.words[:5]:
            print(f"    {w['text']!r:<18} bbox={w['bbox']}  "
                  f"blk={w['block']} ln={w['line']} w={w['word_index']}")

        # Verify reconstruct_text works and roughly matches
        recon = reconstruct_text(page.words)
        print(f"  reconstruct_text() -> {len(recon)} chars; first 120:")
        print("   ", recon[:120].replace("\n", " ").strip())

    print("\n" + "=" * 64)
    print("[PHASE 4 CHECK COMPLETE]")
    print("=" * 64)


if __name__ == "__main__":
    run()
