"""
Utility: extract a raw PDF to a JSON file (text + word coordinates).

Saves <document_id>.json into data/extracted/. Useful for inspecting the
Phase 4 output and for feeding later phases without re-running extraction.

Usage (from project root, with venv python):
    python scripts/extract_to_json.py "data/raw/Some-Policy.pdf"
    python scripts/extract_to_json.py "data/raw/Some-Policy.pdf" EDU_DOC_001
"""
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml.preprocessing.pdf_extractor import extract_pdf, is_probably_scanned

EXTRACTED_DIR = os.path.join(PROJECT_ROOT, "data", "extracted")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/extract_to_json.py <pdf_path> [document_id]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    document_id = sys.argv[2] if len(sys.argv) > 2 else None

    doc = extract_pdf(pdf_path, document_id=document_id)
    if doc.error:
        print(f"ERROR: {doc.error}")
        sys.exit(2)

    if is_probably_scanned(doc):
        print("This PDF appears to contain scanned images. OCR support is required.")
        sys.exit(3)

    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    out_path = os.path.join(EXTRACTED_DIR, f"{doc.document_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"Extracted {doc.total_pages} pages, {doc.total_words} words.")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
