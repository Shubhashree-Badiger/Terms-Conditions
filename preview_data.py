"""Quick preview of raw data files to assess content and suitability."""
import os
import json
import pymupdf  # PyMuPDF

RAW_DIR = os.path.join("data", "raw")

def preview_pdf(path, max_pages=2, max_chars=1200):
    doc = pymupdf.open(path)
    n_pages = len(doc)
    text = ""
    for i in range(min(max_pages, n_pages)):
        text += doc.load_page(i).get_text("text")
    doc.close()
    total_text = ""
    d2 = pymupdf.open(path)
    for p in d2:
        total_text += p.get_text("text")
    d2.close()
    return n_pages, len(total_text), text[:max_chars]

def main():
    for fname in sorted(os.listdir(RAW_DIR)):
        path = os.path.join(RAW_DIR, fname)
        print("=" * 70)
        print(f"FILE: {fname}")
        print("-" * 70)
        if fname.lower().endswith(".pdf"):
            try:
                pages, total_chars, snippet = preview_pdf(path)
                print(f"Pages: {pages} | Total extracted chars: {total_chars}")
                print("--- First ~1200 chars ---")
                print(snippet.strip())
            except Exception as e:
                print(f"ERROR reading PDF: {e}")
        elif fname.lower().endswith(".json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"JSON type: {type(data).__name__}")
                if isinstance(data, list):
                    print(f"Number of entries: {len(data)}")
                    if data:
                        print("--- First entry ---")
                        print(json.dumps(data[0], indent=2, ensure_ascii=False)[:1200])
                elif isinstance(data, dict):
                    print(f"Top-level keys: {list(data.keys())[:20]}")
                    print(json.dumps(data, indent=2, ensure_ascii=False)[:1200])
            except Exception as e:
                print(f"ERROR reading JSON: {e}")
        print()

if __name__ == "__main__":
    main()
