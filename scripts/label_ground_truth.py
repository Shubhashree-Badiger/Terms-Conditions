"""
Interactive Ground-Truth Labeling Helper (Phase 17)
===================================================
Makes creating an evaluation answer key easy: it shows you each clause one at
a time along with the system's SUGGESTED important phrases, and you either
accept them or type the correct ones. It also lets you write a short reference
summary for each clause.

Your answers are saved as the ground-truth "answer key" used to compute real
Precision / Recall / F1 and ROUGE scores.

OUTPUT FILES (this is your answer key):
    data/annotations/EDU_DOC_EVAL.json            <- correct important phrases per clause
    data/annotations/EDU_DOC_EVAL_summaries.json  <- your reference summaries per clause

HOW TO USE (from the project root):
    tc-analyzer\venv\Scripts\python.exe scripts\label_ground_truth.py

For EACH clause you will see:
    - the clause text
    - the system's suggested phrases
  Then you answer two prompts:
    1) PHRASES: press ENTER to accept suggestions, OR type the correct
       important phrases separated by commas (or type "none" if the clause
       has no important phrase).
    2) SUMMARY: press ENTER to accept the system summary, OR type your own
       one-line summary.

Commands at any prompt:
    skip   -> skip this clause (not added to the answer key)
    quit   -> stop and save everything answered so far

You can run it again anytime; it resumes and lets you re-do clauses.
"""
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml.analysis_pipeline import AnalysisPipeline

RAW = os.path.join(PROJECT_ROOT, "data", "raw")
ANN = os.path.join(PROJECT_ROOT, "data", "annotations")

DEFAULT_PDF = "Christ_University_Anti_Ragging_Regulations_Cleaned.pdf"
DOC_ID = "EDU_DOC_EVAL"
MAX_CLAUSES = 15          # how many clauses to label
MIN_CLAUSE_LEN = 60       # skip trivially short clauses


def _load(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def main():
    pdf_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF
    pdf_path = pdf_name if os.path.isabs(pdf_name) else os.path.join(RAW, pdf_name)

    print("=" * 70)
    print(" GROUND-TRUTH LABELING HELPER")
    print("=" * 70)
    print(f"Document : {os.path.basename(pdf_path)}")
    print("Analyzing (please wait ~15-45s)...\n")

    pipeline = AnalysisPipeline(use_semantic=False, use_t5=False)
    result = pipeline.analyze_document(pdf_path, document_id=DOC_ID)
    if result.get("status") != "analyzed":
        print("Cannot label; document status:", result.get("status"))
        return

    os.makedirs(ANN, exist_ok=True)
    gt_path = os.path.join(ANN, f"{DOC_ID}.json")
    ref_path = os.path.join(ANN, f"{DOC_ID}_summaries.json")

    # resume from any existing answers
    gt = _load(gt_path) or {"document_id": DOC_ID, "clauses": {}}
    refs = _load(ref_path) or {}
    gt.setdefault("clauses", {})

    # pick substantive clauses to label
    clauses = [c for c in result["clauses"]
               if len(c["clause_text"]) >= MIN_CLAUSE_LEN][:MAX_CLAUSES]

    print(f"You'll label up to {len(clauses)} clauses.")
    print("At any prompt type 'skip' to skip, or 'quit' to stop and save.\n")

    def save():
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False, indent=2)
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(refs, f, ensure_ascii=False, indent=2)

    for i, c in enumerate(clauses, 1):
        cid = c["clause_id"]
        suggested = [p["text"] for p in c["important_phrases"][:5]]
        sys_summary = c["summary"]

        print("-" * 70)
        print(f"CLAUSE {i}/{len(clauses)}  [{cid} | number {c['clause_number'] or '-'} "
              f"| page {c['page_start']}]")
        print("-" * 70)
        print("TEXT:")
        print("  " + c["clause_text"][:400])
        print()
        print("SYSTEM SUGGESTED PHRASES:", ", ".join(suggested) if suggested else "(none)")

        # --- phrases prompt ---
        ans = input("\n[1] Correct important phrases? "
                    "(ENTER=accept, comma-separated=replace, 'none', 'skip', 'quit'): ").strip()
        if ans.lower() == "quit":
            break
        if ans.lower() == "skip":
            print()
            continue
        if ans == "":
            gold_phrases = suggested
        elif ans.lower() == "none":
            gold_phrases = []
        else:
            gold_phrases = [p.strip() for p in ans.split(",") if p.strip()]

        # --- summary prompt ---
        print(f"\nSYSTEM SUMMARY: {sys_summary}")
        s = input("[2] Reference summary? (ENTER=accept, type your own, 'skip', 'quit'): ").strip()
        if s.lower() == "quit":
            gt["clauses"][cid] = gold_phrases   # still record the phrases just answered
            break
        if s.lower() == "skip":
            gt["clauses"][cid] = gold_phrases
            print()
            save()
            continue
        ref_summary = sys_summary if s == "" else s

        gt["clauses"][cid] = gold_phrases
        refs[cid] = ref_summary
        save()   # save after each clause so progress is never lost
        print("  saved.\n")

    save()
    print("=" * 70)
    print(f"Saved answer key:")
    print(f"  {gt_path}   ({len(gt['clauses'])} clauses labeled)")
    print(f"  {ref_path}   ({len(refs)} summaries)")
    print("\nNext: run  scripts\\run_evaluation.py  to get real P/R/F1 + ROUGE scores.")
    print("=" * 70)


if __name__ == "__main__":
    main()
