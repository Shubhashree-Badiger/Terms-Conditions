"""
Evaluation report generator (Phase 17)
=======================================
Aggregates:
  * phrase P/R/F1        (vs manual ground truth, if provided)
  * summary ROUGE-1/2/L  (vs reference summaries, if provided)
  * coordinate mapping success rate (measured from output)
  * performance timings  (avg PDF time, avg clause time) from the batch summary

Reports ONLY actual measured results. Where no ground truth exists, the report
clearly states "no ground truth available" rather than inventing numbers.

Ground-truth files (optional), per document, under data/annotations/:
    <document_id>.json            -> phrase ground truth
    <document_id>_summaries.json  -> reference summaries
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

from evaluation import phrase_metrics, summary_metrics, mapping_metrics

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "data", "annotations")
METADATA_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")


def _predicted_phrases(result: Dict) -> Dict[str, list]:
    return {c["clause_id"]: [p["text"] for p in c.get("important_phrases", [])]
            for c in result.get("clauses", [])}


def _predicted_summaries(result: Dict) -> Dict[str, str]:
    return {c["clause_id"]: c.get("summary", "") for c in result.get("clauses", [])}


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def evaluate_document(result: Dict) -> Dict:
    """Build an evaluation report for one analyzed-document result."""
    document_id = result.get("document_id", "unknown")

    report: Dict = {"document_id": document_id}

    # --- mapping (always measurable) ---
    report["mapping"] = mapping_metrics.evaluate_mapping(result)

    # --- phrase P/R/F1 (needs ground truth) ---
    gt_path = os.path.join(ANNOTATIONS_DIR, f"{document_id}.json")
    gt = _load_json(gt_path)
    if gt and gt.get("clauses"):
        report["phrase_prf"] = phrase_metrics.evaluate_document(
            _predicted_phrases(result), gt["clauses"])
    else:
        report["phrase_prf"] = None
        report["phrase_prf_note"] = ("No phrase ground truth at "
                                     f"data/annotations/{document_id}.json")

    # --- summary ROUGE (needs references) ---
    ref_path = os.path.join(ANNOTATIONS_DIR, f"{document_id}_summaries.json")
    refs = _load_json(ref_path)
    if refs:
        report["summary_rouge"] = summary_metrics.evaluate_summaries(
            _predicted_summaries(result), refs)
    else:
        report["summary_rouge"] = None
        report["summary_rouge_note"] = ("No reference summaries at "
                                        f"data/annotations/{document_id}_summaries.json")

    return report


def performance_from_batch() -> Optional[Dict]:
    """Read timing metrics from the Phase 16 batch summary, if present."""
    summary = _load_json(os.path.join(METADATA_DIR, "batch_summary.json"))
    if not summary:
        return None
    analyzed = summary.get("analyzed", 0)
    total_clauses = summary.get("total_clauses", 0)
    total_seconds = summary.get("total_seconds", 0.0)
    avg_clause = round(total_seconds / total_clauses, 4) if total_clauses else None
    return {
        "documents_analyzed": analyzed,
        "avg_seconds_per_document": summary.get("avg_seconds_per_doc"),
        "avg_seconds_per_clause": avg_clause,
        "overall_mapping_success_rate": summary.get("mapping_success_rate"),
        "total_pages": summary.get("total_pages"),
        "total_clauses": total_clauses,
        "total_phrases": summary.get("total_phrases"),
    }
