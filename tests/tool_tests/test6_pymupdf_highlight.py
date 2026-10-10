"""
TEST 6: PyMuPDF Highlighting
- Open a PDF
- Search for a phrase on a page
- Add a highlight annotation
- Save an annotated copy (original is never modified)
"""
import os
import pymupdf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PDF_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "2024-Academic-Regulations-Updated-1.pdf")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "test6_annotated.pdf")


def run():
    print("=" * 60)
    print("TEST 6: PyMuPDF - Phrase Search + Highlight Annotation")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    doc = pymupdf.open(PDF_PATH)

    # Find a phrase that actually occurs in the document.
    # We search a common word likely to appear: "attendance"
    target_phrase = "attendance"
    total_hits = 0
    highlighted_pages = []

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        rects = page.search_for(target_phrase)
        if rects:
            for r in rects:
                page.add_highlight_annot(r)
            total_hits += len(rects)
            highlighted_pages.append(page_index + 1)
            # Limit to first few pages with hits for a quick test
            if len(highlighted_pages) >= 3:
                break

    print(f"Searched for: '{target_phrase}'")
    print(f"Total highlights added: {total_hits}")
    print(f"Pages highlighted: {highlighted_pages}")

    # Save annotated copy - original PDF is untouched
    doc.save(OUTPUT_PDF, garbage=4, deflate=True)
    doc.close()

    assert total_hits > 0, "No matches found to highlight"
    assert os.path.exists(OUTPUT_PDF), "Annotated PDF was not created"
    size_kb = os.path.getsize(OUTPUT_PDF) / 1024
    print(f"\nAnnotated PDF saved: {OUTPUT_PDF} ({size_kb:.0f} KB)")
    print("[TEST 6 PASSED] Highlighting works. Open the file to inspect visually.")


if __name__ == "__main__":
    run()
