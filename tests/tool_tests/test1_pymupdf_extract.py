"""
TEST 1: PyMuPDF
- Open a PDF
- Extract page text
- Extract words with coordinates (bounding boxes)
Produces inspectable output printed to console.
"""
import os
import pymupdf  # PyMuPDF (modern import name)

# Path to a known text-based PDF in data/raw
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PDF_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "2024-Academic-Regulations-Updated-1.pdf")


def run():
    print("=" * 60)
    print("TEST 1: PyMuPDF - Text + Word Coordinate Extraction")
    print("=" * 60)
    print(f"PDF: {os.path.basename(PDF_PATH)}")

    doc = pymupdf.open(PDF_PATH)
    print(f"Total pages: {len(doc)}")

    # --- Extract text from page index 4 (5th page) ---
    page_index = 4
    page = doc.load_page(page_index)
    text = page.get_text("text")
    print(f"\n--- Page {page_index + 1}: first 400 chars of text ---")
    print(text[:400].strip())

    # --- Extract words with coordinates ---
    # page.get_text("words") returns tuples:
    # (x0, y0, x1, y1, "word", block_no, line_no, word_no)
    words = page.get_text("words")
    print(f"\n--- Page {page_index + 1}: total words = {len(words)} ---")
    print("First 8 words with bounding boxes (x0, y0, x1, y1):")
    for w in words[:8]:
        x0, y0, x1, y1, word, block, line, wno = w
        print(f"  '{word}'  bbox=({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})  "
              f"block={block} line={line} word={wno}")

    doc.close()
    print("\n[TEST 1 PASSED] Text and coordinates extracted successfully.")


if __name__ == "__main__":
    run()
