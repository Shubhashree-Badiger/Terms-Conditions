"""
Phase 12 verification: complete analysis pipeline via analyze_document().
Runs the whole chain on a real PDF and prints the structured result summary.
"""
import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml.analysis_pipeline import AnalysisPipeline

RAW = os.path.join(PROJECT_ROOT, "data", "raw")
OUT = os.path.join(PROJECT_ROOT, "outputs")
TEST_FILE = "Christ_University_Anti_Ragging_Regulations_Cleaned.pdf"


def run():
    print("=" * 72)
    print("PHASE 12: Complete Analysis Pipeline (analyze_document)")
    print("=" * 72)

    # Build once (semantic + t5 off for speed; the pipeline supports both)
    pipeline = AnalysisPipeline(use_semantic=False, use_t5=False)

    pdf = os.path.join(RAW, TEST_FILE)
    annotated = os.path.join(OUT, "annotated_pdfs", "pipeline_annotated.pdf")

    result = pipeline.analyze_document(
        pdf, document_id="EDU_DOC_DEMO", annotated_out=annotated)

    print(f"\nstatus            : {result['status']}")
    print(f"document_id       : {result['document_id']}")
    print(f"total_pages       : {result['total_pages']}")
    print(f"total_clauses     : {result['total_clauses']}")
    print(f"processing_seconds: {result['processing_seconds']}")
    print(f"stats             : {result['stats']}")
    print(f"annotated_pdf     : {result['annotated_pdf']}")

    # Show a few clauses that have important phrases
    withphrases = [c for c in result["clauses"] if c["important_phrases"]][:3]
    print("\n--- Sample analyzed clauses ---")
    for c in withphrases:
        print(f"\n[{c['clause_id']} | {c['clause_number']} | p{c['page_start']}]")
        print(f"  SUMMARY: {c['summary'][:120]}")
        print(f"  PHRASES:")
        for p in c["important_phrases"][:4]:
            loc = p["locations"][0] if p["locations"] else None
            locstr = f"p{loc['page']} {loc['bbox']}" if loc else "unmapped"
            print(f"     [{p['level']:<6} {p['score']:.2f}] {p['text'][:32]:<32} -> {locstr}")

    # Save the full JSON result for inspection (Phase 13 will formalize this)
    out_json = os.path.join(OUT, "json", "EDU_DOC_DEMO.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nFull JSON saved: {out_json}")

    print("\n" + "=" * 72)
    print("[PHASE 12 CHECK COMPLETE]")


if __name__ == "__main__":
    run()
