"""
Phrase extraction evaluation (Phase 17)
=======================================
Computes Precision, Recall and F1 of predicted important phrases against a
manually labelled ground-truth subset.

Ground-truth format (data/annotations/<document_id>.json):
    {
      "document_id": "EDU_DOC_001",
      "clauses": {
         "C001": ["ragging", "criminal offence"],   # gold important phrases
         "C005": ["75% attendance", "debarred"]
      }
    }

Matching is case-insensitive with light normalization. A predicted phrase
counts as a true positive if it matches (or is contained in / contains) a gold
phrase for the same clause -- this tolerates minor boundary differences
(e.g. "attendance" vs "75% attendance").

Only ACTUAL measured results are reported; nothing is fabricated. If no
ground truth exists, the functions return None so the report can say so.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set, Optional


def _norm(p: str) -> str:
    return re.sub(r"\s+", " ", p.strip().lower()).strip(" .,:;()[]\"'")


def _match(pred: str, gold_set: Set[str]) -> bool:
    p = _norm(pred)
    if not p:
        return False
    for g in gold_set:
        if p == g or p in g or g in p:
            return True
    return False


def evaluate_clause(pred_phrases: List[str], gold_phrases: List[str]) -> Dict:
    """Return tp/fp/fn counts for one clause."""
    gold_set = {_norm(g) for g in gold_phrases if _norm(g)}
    matched_gold: Set[str] = set()
    tp = 0
    fp = 0
    for pred in pred_phrases:
        if _match(pred, gold_set):
            tp += 1
            # mark which gold matched (best-effort)
            pn = _norm(pred)
            for g in gold_set:
                if pn == g or pn in g or g in pn:
                    matched_gold.add(g)
        else:
            fp += 1
    fn = len(gold_set - matched_gold)
    return {"tp": tp, "fp": fp, "fn": fn}


def prf(tp: int, fp: int, fn: int) -> Dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


def evaluate_document(predicted: Dict[str, List[str]],
                      ground_truth: Dict[str, List[str]]) -> Optional[Dict]:
    """Aggregate P/R/F1 over all annotated clauses of a document.

    Args:
        predicted: {clause_id: [predicted phrase strings]}
        ground_truth: {clause_id: [gold phrase strings]}

    Returns:
        metrics dict, or None if ground_truth is empty.
    """
    if not ground_truth:
        return None
    tp = fp = fn = 0
    for clause_id, gold in ground_truth.items():
        preds = predicted.get(clause_id, [])
        c = evaluate_clause(preds, gold)
        tp += c["tp"]; fp += c["fp"]; fn += c["fn"]
    result = prf(tp, fp, fn)
    result["clauses_evaluated"] = len(ground_truth)
    return result
