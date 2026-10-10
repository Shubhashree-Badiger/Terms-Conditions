"""
Results formatting + persistence (Phase 13)
===========================================
Formats the AnalysisPipeline output into the exact JSON structure specified
by the project, and persists it to backend/outputs/<document_id>.json.

Spec JSON shape:
    {
      "document_id": "EDU_DOC_001",
      "filename": "academic_rules.pdf",
      "total_pages": 20,
      "total_clauses": 45,
      "status": "analyzed",
      "processing_seconds": 12.3,
      "stats": { ... },
      "clauses": [
        {
          "clause_id": "C001",
          "clause_number": "3.2",
          "page_start": 4,
          "page_end": 4,
          "clause_text": "...",
          "summary": "Students need at least 75% attendance to take the exam.",
          "important_phrases": [
            {
              "text": "minimum attendance of 75%",
              "score": 0.93,
              "level": "HIGH",
              "category": "attendance",
              "risk_type": "academic",
              "risk_level": "MEDIUM",
              "locations": [ { "page": 4, "bbox": [100,250,300,270] } ]
            }
          ]
        }
      ],
      "annotated_pdf": "/api/documents/EDU_DOC_001/annotated"
    }
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

from app import config


def _annotated_api_path(document_id: str) -> str:
    return f"/api/documents/{document_id}/annotated"


def format_result(pipeline_result: Dict) -> Dict:
    """Shape a pipeline result dict into the final spec JSON structure.

    Non-analyzed results (scanned / error) are passed through with their
    status so the API can return an appropriate response.
    """
    status = pipeline_result.get("status")
    document_id = pipeline_result.get("document_id")

    if status != "analyzed":
        # scanned / error: return as-is (already minimal + informative)
        return pipeline_result

    clauses_out = []
    for c in pipeline_result.get("clauses", []):
        phrases_out = []
        for p in c.get("important_phrases", []):
            phrases_out.append({
                "text": p.get("text"),
                "score": p.get("score"),
                "level": p.get("level"),
                "category": p.get("category"),
                "constraint_type": p.get("constraint_type"),
                "risk_type": p.get("risk_type"),
                "risk_level": p.get("risk_level"),
                "locations": p.get("locations", []),
            })
        clauses_out.append({
            "clause_id": c.get("clause_id"),
            "clause_number": c.get("clause_number", ""),
            "page_start": c.get("page_start"),
            "page_end": c.get("page_end"),
            "clause_text": c.get("clause_text", ""),
            "summary": c.get("summary", ""),
            "important_phrases": phrases_out,
        })

    return {
        "document_id": document_id,
        "filename": pipeline_result.get("filename"),
        "total_pages": pipeline_result.get("total_pages"),
        "total_clauses": pipeline_result.get("total_clauses", len(clauses_out)),
        "status": "analyzed",
        "processing_seconds": pipeline_result.get("processing_seconds"),
        "stats": pipeline_result.get("stats", {}),
        "annotated_pdf": _annotated_api_path(document_id),
        "clauses": clauses_out,
    }


def result_json_path(document_id: str) -> str:
    return str(config.BACKEND_OUTPUTS_DIR / f"{document_id}.json")


def save_result(result: Dict) -> str:
    """Persist a formatted result to backend/outputs/<document_id>.json."""
    config.ensure_directories()
    document_id = result.get("document_id", "unknown")
    path = result_json_path(document_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path


def load_result(document_id: str) -> Optional[Dict]:
    """Load a previously saved result, or None if not present.

    Enables the API to serve cached results without re-running NLP
    (supports the 'do not rerun NLP unnecessarily' rule)."""
    path = result_json_path(document_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
