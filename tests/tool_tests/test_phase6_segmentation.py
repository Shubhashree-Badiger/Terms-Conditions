"""
Phase 6 verification: clause segmentation on real PDFs.
Runs extract -> preprocess -> segment and shows sample clauses + stats.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml.preprocessing.pdf_extractor import extract_pdf
from ml.preprocessing.text_preprocessor import preprocess_document
from ml.clause_segmentation.clause_segmenter import segment_document

RAW = os.path.join(PROJECT_ROOT, "data", "raw")
TEST_FILES = [
    "2024-Academic-Regulations-Updated-1.pdf",
    "Christ_University_Anti_Ragging_Regulations_Cleaned.pdf",
]


def run():
    print("=" * 66)
    print("PHASE 6: Clause Segmentation")
    print("=" * 66)

    for fname in TEST_FILES:
        path = os.path.join(RAW, fname)
        if not os.path.exists(path):
            print(f"\n[SKIP] {fname}")
            continue

        extracted = extract_pdf(path)
        pp = preprocess_document(extracted)
        clauses = segment_document(pp)

        numbered = [c for c in clauses if c["clause_number"] and c["clause_number"][0].isdigit()]
        lettered = [c for c in clauses if c["clause_number"].startswith("(")]
        plain = [c for c in clauses if c["clause_number"] == ""]

        print(f"\n### {fname}")
        print(f"  pages={extracted.total_pages}  total_clauses={len(clauses)}")
        print(f"  numbered={len(numbered)}  lettered={len(lettered)}  plain/sentence={len(plain)}")

        print("  --- Sample numbered clauses ---")
        for c in numbered[:4]:
            txt = c["clause_text"][:110].replace("\n", " ")
            print(f"    [{c['clause_id']} | {c['clause_number']:>6} | p{c['page_start']}] {txt}")

    print("\n" + "=" * 66)
    print("[PHASE 6 CHECK COMPLETE]")


if __name__ == "__main__":
    run()
