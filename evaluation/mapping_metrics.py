"""
Coordinate mapping evaluation (Phase 17)
========================================
Computes the coordinate-mapping success rate (mapped / total phrases) from an
analysis result, and lists failures for error analysis (Phase 18).

This metric is fully measurable from actual pipeline output -- no human
labelling required.
"""
from __future__ import annotations

from typing import Dict, List


def evaluate_mapping(result: Dict) -> Dict:
    """Return mapping stats for one analyzed-document result dict.

    Uses each phrase's `locations` (present = mapped) across all clauses.
    """
    total = 0
    mapped = 0
    failures: List[Dict] = []

    for clause in result.get("clauses", []):
        for p in clause.get("important_phrases", []):
            total += 1
            if p.get("locations"):
                mapped += 1
            else:
                failures.append({
                    "clause_id": clause.get("clause_id"),
                    "phrase": p.get("text"),
                    "page_start": clause.get("page_start"),
                })

    rate = round(mapped / total * 100, 2) if total else 0.0
    return {
        "total_phrases": total,
        "mapped_phrases": mapped,
        "failed_phrases": total - mapped,
        "mapping_success_rate": rate,
        "failures": failures,
    }
