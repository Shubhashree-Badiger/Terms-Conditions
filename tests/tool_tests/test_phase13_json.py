"""
Phase 13 verification: JSON output formatting + persistence.
Runs the pipeline, formats to the spec shape, saves and reloads.
"""
import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from ml.analysis_pipeline import AnalysisPipeline
from app.services.results_store import format_result, save_result, load_result

RAW = os.path.join(PROJECT_ROOT, "data", "raw")
TEST_FILE = "Christ_University_Anti_Ragging_Regulations_Cleaned.pdf"


def run():
    print("=" * 72)
    print("PHASE 13: JSON Output (spec shape + persistence)")
    print("=" * 72)

    pipeline = AnalysisPipeline(use_semantic=False, use_t5=False)
    raw_result = pipeline.analyze_document(
        os.path.join(RAW, TEST_FILE), document_id="EDU_DOC_001")

    formatted = format_result(raw_result)

    # validate top-level keys
    top_keys = ["document_id", "filename", "total_pages", "total_clauses",
                "status", "annotated_pdf", "clauses"]
    print("Top-level keys present:")
    for k in top_keys:
        print(f"   {k:<20} {'OK' if k in formatted else 'MISSING'}")

    print(f"\nannotated_pdf -> {formatted['annotated_pdf']}")

    # validate a clause with phrases
    withp = next((c for c in formatted["clauses"] if c["important_phrases"]), None)
    if withp:
        print("\nSample clause structure:")
        clause_keys = ["clause_id", "clause_number", "page_start", "page_end",
                       "clause_text", "summary", "important_phrases"]
        for k in clause_keys:
            print(f"   {k:<20} {'OK' if k in withp else 'MISSING'}")
        p = withp["important_phrases"][0]
        print("\nSample phrase structure:")
        for k in ["text", "score", "level", "locations"]:
            print(f"   {k:<20} {'OK' if k in p else 'MISSING'}  = {p.get(k)}")

    # persist + reload
    path = save_result(formatted)
    print(f"\nSaved to: {path}")
    reloaded = load_result("EDU_DOC_001")
    print(f"Reloaded OK: {reloaded is not None and reloaded['document_id'] == 'EDU_DOC_001'}")

    # confirm valid JSON serialization
    json.dumps(formatted)
    print("JSON serializable: True")

    print("\n" + "=" * 72)
    print("[PHASE 13 CHECK COMPLETE]")


if __name__ == "__main__":
    run()
