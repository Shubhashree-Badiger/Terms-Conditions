"""
Summary evaluation (Phase 17)
=============================
Computes ROUGE-1, ROUGE-2 and ROUGE-L (F1) for generated one-line summaries
against reference summaries. Implemented from scratch (no extra dependency) to
keep the project lightweight.

ROUGE-N (F1): n-gram overlap between candidate and reference.
ROUGE-L (F1): based on the Longest Common Subsequence.

Reference format (data/annotations/<document_id>_summaries.json):
    { "C001": "reference summary text", "C005": "..." }

Only actual measured results are reported.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9%]+", (text or "").lower())


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _rouge_n(candidate: str, reference: str, n: int) -> float:
    cand = _ngrams(_tokens(candidate), n)
    ref = _ngrams(_tokens(reference), n)
    if not cand or not ref:
        return 0.0
    ref_counts = {}
    for g in ref:
        ref_counts[g] = ref_counts.get(g, 0) + 1
    overlap = 0
    cand_counts = {}
    for g in cand:
        cand_counts[g] = cand_counts.get(g, 0) + 1
    for g, c in cand_counts.items():
        if g in ref_counts:
            overlap += min(c, ref_counts[g])
    precision = overlap / len(cand)
    recall = overlap / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _rouge_l(candidate: str, reference: str) -> float:
    cand = _tokens(candidate)
    ref = _tokens(reference)
    if not cand or not ref:
        return 0.0
    lcs = _lcs_length(cand, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def score_pair(candidate: str, reference: str) -> Dict:
    """ROUGE-1/2/L F1 for a single candidate/reference pair."""
    return {
        "rouge1": round(_rouge_n(candidate, reference, 1), 4),
        "rouge2": round(_rouge_n(candidate, reference, 2), 4),
        "rougeL": round(_rouge_l(candidate, reference), 4),
    }


def evaluate_summaries(predicted: Dict[str, str],
                       references: Dict[str, str]) -> Optional[Dict]:
    """Average ROUGE-1/2/L over all clauses with a reference summary.

    Args:
        predicted: {clause_id: generated summary}
        references: {clause_id: reference summary}

    Returns:
        {rouge1, rouge2, rougeL, pairs_evaluated} or None if no references.
    """
    if not references:
        return None
    r1 = r2 = rl = 0.0
    count = 0
    for clause_id, ref in references.items():
        cand = predicted.get(clause_id, "")
        s = score_pair(cand, ref)
        r1 += s["rouge1"]; r2 += s["rouge2"]; rl += s["rougeL"]
        count += 1
    if count == 0:
        return None
    return {
        "rouge1": round(r1 / count, 4),
        "rouge2": round(r2 / count, 4),
        "rougeL": round(rl / count, 4),
        "pairs_evaluated": count,
    }
