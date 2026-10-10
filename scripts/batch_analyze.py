"""
Phase 16: Batch analysis of many PDFs.
=====================================
Runs the complete pipeline over every PDF in data/raw and records per-document
statistics required by the spec:

  - number of pages
  - number of clauses
  - number of important phrases
  - number of successfully mapped phrases
  - number of mapping failures
  - processing time
  - summary failures
  - status (analyzed / scanned / error)

Outputs:
  - console summary table
  - data/metadata/batch_report.csv   (per-document rows)
  - data/metadata/batch_summary.json (aggregate totals)

Usage (from project root, with venv python):
    python scripts/batch_analyze.py
    python scripts/batch_analyze.py 10        # limit to first 10 PDFs
"""
import csv
import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml.analysis_pipeline import AnalysisPipeline

RAW = os.path.join(PROJECT_ROOT, "data", "raw")
META = os.path.join(PROJECT_ROOT, "data", "metadata")
OUT_ANN = os.path.join(PROJECT_ROOT, "outputs", "annotated_pdfs")

CSV_PATH = os.path.join(META, "batch_report.csv")
SUMMARY_PATH = os.path.join(META, "batch_summary.json")


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    pdfs = sorted([f for f in os.listdir(RAW) if f.lower().endswith(".pdf")])
    if limit:
        pdfs = pdfs[:limit]

    print("=" * 78)
    print(f"PHASE 16: Batch analysis of {len(pdfs)} PDFs")
    print("=" * 78)

    os.makedirs(META, exist_ok=True)
    os.makedirs(OUT_ANN, exist_ok=True)

    # Build the pipeline ONCE (models loaded once, reused for all docs)
    pipeline = AnalysisPipeline(use_semantic=False, use_t5=False)

    rows = []
    agg = {
        "total_docs": len(pdfs), "analyzed": 0, "scanned": 0, "error": 0,
        "total_pages": 0, "total_clauses": 0, "total_phrases": 0,
        "mapped_phrases": 0, "failed_phrases": 0, "summary_failures": 0,
        "clause_errors": 0, "total_seconds": 0.0,
    }

    for i, fname in enumerate(pdfs, 1):
        path = os.path.join(RAW, fname)
        doc_id = f"EDU_DOC_{i:03d}"
        annotated = os.path.join(OUT_ANN, f"{doc_id}_annotated.pdf")

        t0 = time.time()
        try:
            result = pipeline.analyze_document(path, document_id=doc_id,
                                               annotated_out=annotated)
        except Exception as e:
            result = {"status": "error", "error": str(e)}
        secs = round(time.time() - t0, 2)

        status = result.get("status", "error")
        stats = result.get("stats", {})
        row = {
            "document_id": doc_id,
            "filename": fname,
            "status": status,
            "pages": result.get("total_pages", 0),
            "clauses": result.get("total_clauses", 0),
            "phrases": stats.get("total_phrases", 0),
            "mapped": stats.get("mapped_phrases", 0),
            "failed": stats.get("failed_phrases", 0),
            "summary_failures": stats.get("summary_failures", 0),
            "clause_errors": stats.get("clause_errors", 0),
            "seconds": secs,
            "note": result.get("error") or result.get("message") or "",
        }
        rows.append(row)

        # aggregate
        agg["total_seconds"] += secs
        if status == "analyzed":
            agg["analyzed"] += 1
            agg["total_pages"] += row["pages"]
            agg["total_clauses"] += row["clauses"]
            agg["total_phrases"] += row["phrases"]
            agg["mapped_phrases"] += row["mapped"]
            agg["failed_phrases"] += row["failed"]
            agg["summary_failures"] += row["summary_failures"]
            agg["clause_errors"] += row["clause_errors"]
        elif status == "scanned":
            agg["scanned"] += 1
        else:
            agg["error"] += 1

        print(f"[{i:>2}/{len(pdfs)}] {status:<9} {row['pages']:>3}p "
              f"{row['clauses']:>4}cl {row['phrases']:>4}ph "
              f"map{row['mapped']:>4}/{row['failed']:<3}fail {secs:>6.1f}s  {fname[:38]}")

    # write CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # derived aggregate metrics
    if agg["total_phrases"] > 0:
        agg["mapping_success_rate"] = round(
            agg["mapped_phrases"] / agg["total_phrases"] * 100, 1)
    else:
        agg["mapping_success_rate"] = 0.0
    if agg["analyzed"] > 0:
        agg["avg_seconds_per_doc"] = round(agg["total_seconds"] / len(pdfs), 2)
        agg["avg_clauses_per_doc"] = round(agg["total_clauses"] / agg["analyzed"], 1)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    # console summary
    print("\n" + "=" * 78)
    print("AGGREGATE SUMMARY")
    print("=" * 78)
    print(f"  Documents         : {agg['total_docs']}")
    print(f"    analyzed        : {agg['analyzed']}")
    print(f"    scanned (OCR)   : {agg['scanned']}")
    print(f"    error           : {agg['error']}")
    print(f"  Total pages       : {agg['total_pages']}")
    print(f"  Total clauses     : {agg['total_clauses']}")
    print(f"  Total phrases     : {agg['total_phrases']}")
    print(f"  Mapped phrases    : {agg['mapped_phrases']}")
    print(f"  Failed mappings   : {agg['failed_phrases']}")
    print(f"  Mapping success   : {agg['mapping_success_rate']}%")
    print(f"  Summary failures  : {agg['summary_failures']}")
    print(f"  Clause errors     : {agg['clause_errors']}")
    print(f"  Total time        : {round(agg['total_seconds'],1)}s")
    print(f"  Avg time/doc      : {agg.get('avg_seconds_per_doc','-')}s")
    print(f"  Avg clauses/doc   : {agg.get('avg_clauses_per_doc','-')}")
    print(f"\nReports written:\n  {CSV_PATH}\n  {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
