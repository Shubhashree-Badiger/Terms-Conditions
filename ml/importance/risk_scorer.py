"""
Risk Scorer (Phase 8, supporting)
=================================
Maps a phrase (in the context of its clause) to a risk type and level using
the risk lexicon (configs/risk_lexicon.json).

"Important" and "risky" are DIFFERENT concepts:
  - An important clause states a key rule (e.g. "carry your ID card").
  - A risky clause imposes a penalty / financial loss / rights waiver
    (e.g. "fees are non-refundable", "will be expelled").

This module returns the risk view; the importance scorer returns the
importance view. Both are reported so the UI can distinguish them.

Public API:
    load_risk_lexicon(path) -> dict
    class RiskScorer:
        score(phrase, clause_text) -> dict
            { "risk_type": <category|None>, "risk_level": HIGH|MEDIUM|LOW|NONE,
              "evidence": [matched terms] }
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List

try:
    from app import config as _cfg
    _DEFAULT_RISK = str(_cfg.RISK_LEXICON_PATH)
except Exception:
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _DEFAULT_RISK = os.path.join(_PROJECT_ROOT, "configs", "risk_lexicon.json")


# Category weights map to how severe a risk category is. Higher = riskier.
# Derived from the "weight" field in risk_lexicon.json.
def load_risk_lexicon(path: str = _DEFAULT_RISK) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


class RiskScorer:
    def __init__(self, risk_lexicon_path: str = _DEFAULT_RISK):
        self.lexicon = load_risk_lexicon(risk_lexicon_path)
        # Precompile term -> (category, weight)
        self._terms: List[tuple] = []
        max_w = 1.0
        for category, block in self.lexicon.items():
            weight = float(block.get("weight", 1.0))
            max_w = max(max_w, weight)
            for kw in block.get("keywords", []):
                self._terms.append((kw.lower(), category, weight))
        self._max_weight = max_w

    def score(self, phrase: str, clause_text: str = "") -> Dict:
        """Return risk assessment for a phrase within its clause context.

        We search both the phrase itself and the surrounding clause text so a
        short phrase like "75%" can still inherit risk from nearby words such
        as "debarred" only when they belong to the same clause. Matching is on
        the phrase first; clause context is used to boost/confirm.
        """
        haystack = f"{phrase} {clause_text}".lower()
        matched: List[tuple] = []  # (term, category, weight)
        for term, category, weight in self._terms:
            pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
            if re.search(pattern, haystack):
                matched.append((term, category, weight))

        if not matched:
            return {"risk_type": None, "risk_level": "NONE", "evidence": []}

        # highest-weight category wins
        matched.sort(key=lambda t: t[2], reverse=True)
        top_term, top_cat, top_w = matched[0]
        norm = top_w / self._max_weight  # 0..1

        if norm >= 0.85:
            level = "HIGH"
        elif norm >= 0.6:
            level = "MEDIUM"
        else:
            level = "LOW"

        # unique evidence terms, preserve order by weight
        evidence = []
        for term, _, _ in matched:
            if term not in evidence:
                evidence.append(term)

        return {
            "risk_type": top_cat,
            "risk_level": level,
            "evidence": evidence[:6],
        }
