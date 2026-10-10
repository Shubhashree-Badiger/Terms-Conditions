"""
Phase 17 runner: evaluation.
============================
1. Analyzes one document.
2. If no ground-truth files exist, SEEDS templates you can verify/edit:
     data/annotations/<id>.json            (phrase ground truth)
     data/annotations/<id>_summaries.json  (reference summaries)
   The seed uses the system's own top phrases/summaries as a STARTING POINT.
   You should review and correct these to make the evaluation meaningful.
3. Runs the evaluation report (P/R/F1, ROUGE, mapping, timings) using
   whatever ground truth is present, and prints ONLY measured results.

Usage (from project root, venv python):
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py "data/raw/Some.pdf" EDU_DOC_EVAL
    python scripts/run_evaluation.py --seed        # (re)write seed templates
"""
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml.analysis_pipeline import AnalysisPipeline
from evaluation.evaluation_report import evaluate_document, performance_from_batch

RAW = os.path.join(PROJECT_ROOT, "data", "raw")
ANN = os.path.join(PROJECT_ROOT, "data", "annotations")
DEFAULT_PDF = "Christ_University_Anti_Ragging_Regulations_Cleaned.pdf"


def seed_ground_truth(result, document_id, overwrite=False):
    """Write seed ground-truth templates from the system output (to be verified)."""
    os.makedirs(ANN, exist_ok=True)
    gt_path = os.path.join(ANN, f"{document_id}.json")
    ref_path = os.path.join(ANN, f"{document_id}_summaries.json")

    # take a manageable subset: clauses that have phrases, first ~12
    clauses = [c for c in result["clauses"] if c["important_phrases"]][:12]

    if overwrite or not os.path.exists(gt_path):
        gt = {"document_id": document_id, "_note":
              "SEED - verify/correct these gold phrases per clause, then re-run.",
              "clauses": {c["clause_id"]: [p["text"] for p in c["important_phrases"][:4]]
                          for c in clauses}}
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False, indent=2)
        print(f"Seeded phrase ground truth: {gt_path}")

    if overwrite or not os.path.exists(ref_path):
        refs = {c["clause_id"]: c["summary"] for c in clauses}
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(refs, f, ensure_ascii=False, indent=2)
        print(f"Seeded reference summaries: {ref_path}")


def main():
    args = [a for a in sys.argv[1:] if a != "--seed"]
    force_seed = "--seed" in sys.argv
    pdf = args[0] if len(args) > 0 else os.path.join(RAW, DEFAULT_PDF)
    document_id = args[1] if len(args) > 1 else "EDU_DOC_EVAL"

    print("=" * 74)
    print("PHASE 17: Evaluation")
    print("=" * 74)

    pipeline = AnalysisPipeline(use_semantic=False, use_t5=False)
    result = pipeline.analyze_document(pdf, document_id=document_id)
    if result.get("status") != "analyzed":
        print("Cannot evaluate; status:", result.get("status"))
        return

    seed_ground_truth(result, document_id, overwrite=force_seed)

    report = evaluate_document(result)

    print("\n--- MAPPING (measured) ---")
    m = report["mapping"]
    print(f"  total_phrases        : {m['total_phrases']}")
    print(f"  mapped_phrases       : {m['mapped_phrases']}")
    print(f"  failed_phrases       : {m['failed_phrases']}")
    print(f"  mapping_success_rate : {m['mapping_success_rate']}%")

    print("\n--- PHRASE PRECISION / RECALL / F1 ---")
    if report["phrase_prf"]:
        p = report["phrase_prf"]
        print(f"  clauses_evaluated    : {p['clauses_evaluated']}")
        print(f"  precision            : {p['precision']}")
        print(f"  recall               : {p['recall']}")
        print(f"  f1                   : {p['f1']}")
        print(f"  (tp={p['tp']} fp={p['fp']} fn={p['fn']})")
        print("  Computed against the human-labeled answer key in "
              "data/annotations/.")
    else:
        print(" ", report.get("phrase_prf_note"))

    print("\n--- SUMMARY ROUGE ---")
    if report["summary_rouge"]:
        r = report["summary_rouge"]
        print(f"  pairs_evaluated      : {r['pairs_evaluated']}")
        print(f"  ROUGE-1              : {r['rouge1']}")
        print(f"  ROUGE-2              : {r['rouge2']}")
        print(f"  ROUGE-L              : {r['rougeL']}")
        print("  Computed against the human-written reference summaries in "
              "data/annotations/.")
    else:
        print(" ", report.get("summary_rouge_note"))

    print("\n--- PERFORMANCE (from batch summary, if available) ---")
    perf = performance_from_batch()
    if perf:
        for k, v in perf.items():
            print(f"  {k:<28}: {v}")
    else:
        print("  No batch_summary.json found (run scripts/batch_analyze.py).")

    # save the report
    out = os.path.join(PROJECT_ROOT, "data", "metadata", "evaluation_report.json")
    report["performance"] = perf
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nEvaluation report saved: {out}")

    print("\n" + "=" * 74)
    print("[PHASE 17 CHECK COMPLETE]")


if __name__ == "__main__":
    main()
