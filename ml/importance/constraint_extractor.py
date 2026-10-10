"""
Constraint / Condition Extractor (Phase 24)
===========================================
The centrality + YAKE + domain signals are good at finding what a clause is
ABOUT (e.g. "attendance"), but they miss the most decision-critical parts of a
policy clause: the concrete BOUNDARIES and CONSEQUENCES, such as

    "80%"                                  (a threshold / limit)
    "less than 80%"                        (a comparison boundary)
    "not less than 75%"                    (a minimum)
    "Rs. 500"                              (a monetary amount)
    "within 15 days"                       (a deadline / duration)
    "shall not be permitted to appear for the Semester End Examinations"
                                           (a prohibition / consequence)
    "if his attendance is less than 80%"   (a condition that triggers a rule)

These are exactly the phrases a student needs highlighted. This module finds
them with deterministic patterns (no training, no external model) and returns
them as candidate phrases carrying:

    {
        "phrase": "<matched span>",
        "sources": ["constraint"],
        "yake_score": None,
        "category": <domain category or None>,
        "constraint_type": "threshold" | "comparison" | "amount" |
                           "duration" | "consequence" | "condition",
        "constraint_boost": 0.0..1.0,   # how strongly to prioritize it
    }

The importance scorer treats a non-None constraint_type as a strong signal, so
these boundaries/consequences reliably surface at the top of the highlights.

Public API:
    extract_constraints(clause_text) -> list[dict]
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# --- consequence / obligation verbs that start an actionable span ----------
# Used to capture the FULL consequence phrase, e.g.
# "shall not be permitted to appear for the Semester End Examinations".
# Single-verb consequences that don't need a "shall/will" lead, e.g.
# "he has to reappear", "will forfeit the seat".
_CONSEQUENCE_WORDS = re.compile(
    r"\b(?:reappear|re-appear|forfeit(?:ed|s)?|debarred|detained|expelled|"
    r"suspended|rusticated|disqualified|penalis(?:ed|e)|penaliz(?:ed|e)|"
    r"cancelled|terminated|rejected|not\s+be\s+made|not\s+be\s+eligible)"
    r"[^.;\n]{0,70}",
    re.IGNORECASE,
)

# "minimum/maximum of 50% of marks", "minimum of 50%" - a required level.
_MIN_MAX_RE = re.compile(
    r"\b(?:a\s+)?(?:minimum|maximum|min\.?|max\.?)\s+of\s+"
    r"\d[\d,]*(?:\.\d+)?\s*(?:%|percent|per\s*cent|marks?|credits?)?"
    r"(?:\s+of\s+marks)?",
    re.IGNORECASE,
)

# Bare quantity + unit that carries weight in academics: "100 marks",
# "50 marks", "3 credits", "two periods".
_QUANTITY_RE = re.compile(
    r"\b(?:\d[\d,]*(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s+(?:marks?|credits?|periods?|attempts?|semesters?|hours?)\b",
    re.IGNORECASE,
)

# STRONG consequence leads: prohibitions / obligations that actually impose a
# rule. These carry the real "what happens" and get the top boost.
_STRONG_LEAD = (
    r"(?:shall\s+not|will\s+not|must\s+not|may\s+not|cannot|can\s+not|"
    r"is\s+not\s+permitted|are\s+not\s+permitted|is\s+liable|are\s+liable|"
    r"is\s+prohibited|are\s+prohibited|is\s+debarred|are\s+debarred|"
    r"is\s+detained|are\s+detained|has\s+to|have\s+to|shall\s+have\s+to|"
    r"will\s+have\s+to|must|shall\s+be\s+(?:debarred|liable|penali|suspended|"
    r"expelled|cancelled|forfeited|rejected|required))"
)
# WEAK leads: "shall be / will be / shall" used descriptively. Lower boost and
# tighter span so we don't grab whole descriptive sentences.
_WEAK_LEAD = r"(?:shall(?:\s+be)?|will\s+be|would\s+be|is\s+required|are\s+required)"

# A consequence span runs from the lead verb up to the next natural break
# (sentence end, comma, or connective) so it stays a concise core, not a whole
# sentence.
_CONSEQUENCE_RE = re.compile(
    _STRONG_LEAD + r"[^.;,\n]{0,70}",
    re.IGNORECASE,
)
_WEAK_CONSEQUENCE_RE = re.compile(
    _WEAK_LEAD + r"[^.;,\n]{0,50}",
    re.IGNORECASE,
)

# --- conditional triggers: "if ...", "unless ...", "in case of ...", etc. --
_CONDITION_RE = re.compile(
    r"\b(?:if|unless|provided that|in case of|in the event of|failing which|"
    r"subject to|only if|except when|whenever)\b[^.;,\n]{0,80}",
    re.IGNORECASE,
)

# --- numeric comparisons / boundaries: "less than 80%", "not less than 75%",
#     "at least 3", "minimum of 75%", "up to Rs. 5000", "not exceeding 10 days"
_COMPARATOR = (
    r"(?:less than|greater than|more than|not less than|not more than|"
    r"at least|at most|minimum of|maximum of|min\.?|max\.?|"
    r"up to|upto|not exceeding|exceeding|below|above|under|over|"
    r"no later than|not later than|on or before|within|before|after)"
)
_NUM = r"(?:rs\.?\s*|inr\s*|₹\s*|\$\s*)?\d[\d,]*(?:\.\d+)?\s*(?:%|percent|per\s*cent|days?|weeks?|months?|years?|hours?|marks?|credits?|rupees?)?"

_COMPARISON_RE = re.compile(
    r"\b" + _COMPARATOR + r"\s+" + _NUM,
    re.IGNORECASE,
)

# --- ranges: "between 60% and 80%", "60% to 80%", "60-80%" -----------------
_RANGE_RE = re.compile(
    r"\b(?:between\s+)?" + _NUM + r"\s*(?:to|-|–|and)\s*" + _NUM,
    re.IGNORECASE,
)

# --- standalone monetary amounts: "Rs. 500", "₹5,000", "$100" --------------
_AMOUNT_RE = re.compile(
    r"(?:rs\.?|inr|₹|\$)\s*\d[\d,]*(?:\.\d+)?",
    re.IGNORECASE,
)

# --- standalone percentages: "80%", "75 percent" --------------------------
_PERCENT_RE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s*(?:%|percent|per\s*cent)\b",
    re.IGNORECASE,
)

# --- durations / counts with a unit: "15 days", "two periods", "3 credits" -
_DURATION_RE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s*(?:days?|weeks?|months?|years?|hours?|"
    r"working\s+days?|semesters?|credits?|marks?|attempts?|periods?)\b",
    re.IGNORECASE,
)


def _clean_span(span: str) -> str:
    """Trim whitespace and dangling connectors from a matched span.

    Also drops a trailing dangling connector word (e.g. a span that ends in
    "... is less" because the window cut off before "than 80%") so highlights
    end on a meaningful token.
    """
    s = re.sub(r"\s+", " ", span).strip()
    s = s.strip(" ,;:.-")
    # If the regex window cut the last token mid-word (no trailing space/punct
    # in the source), drop that partial trailing token.
    if s and not span.endswith((" ", ".", ",", ";", ":")) and " " in s:
        head, _, last = s.rpartition(" ")
        # a very short trailing fragment is almost always a cut word
        if len(last) <= 4 and last.lower() not in {"marks", "days", "%"}:
            s = head
    # remove a trailing dangling connective that leaves the phrase incomplete
    s = re.sub(r"\s+(?:is|are|be|to|of|the|a|an|and|or|for|in|on|than|that|"
               r"with|as|at|by|less|more|not|before|after|during)$", "",
               s, flags=re.IGNORECASE)
    return s.strip(" ,;:.-")


def _add(out: Dict[str, Dict], phrase: str, ctype: str, boost: float):
    """Add/merge a constraint candidate, keeping the strongest boost."""
    phrase = _clean_span(phrase)
    # trim filler articles/prepositions from the edges (keep numbers/%)
    try:
        from ml.importance.phrase_extractor import trim_phrase
        phrase = trim_phrase(phrase)
    except Exception:
        pass
    # need at least one alphanumeric char and reasonable length
    if not phrase or not re.search(r"[a-z0-9]", phrase, re.IGNORECASE):
        return
    if len(phrase) > 160:
        return
    key = phrase.lower()
    rec = out.get(key)
    if rec is None:
        out[key] = {
            "phrase": phrase,
            "sources": ["constraint"],
            "yake_score": None,
            "category": None,
            "constraint_type": ctype,
            "constraint_boost": boost,
        }
    else:
        if boost > rec["constraint_boost"]:
            rec["constraint_boost"] = boost
            rec["constraint_type"] = ctype


def extract_constraints(clause_text: str) -> List[Dict]:
    """Extract boundary / condition / consequence phrases from a clause.

    Returns a list of candidate records (see module docstring). Ordered so the
    most decision-critical types (consequences, comparisons/thresholds) come
    first, but the importance scorer ultimately decides ranking.
    """
    if not clause_text or not clause_text.strip():
        return []

    out: Dict[str, Dict] = {}

    # Highest priority: strong consequences (prohibitions/obligations) and
    # conditional triggers - these carry the "what happens" and "when".
    for m in _CONSEQUENCE_RE.finditer(clause_text):
        _add(out, m.group(0), "consequence", 0.95)
    for m in _CONSEQUENCE_WORDS.finditer(clause_text):
        _add(out, m.group(0), "consequence", 0.90)
    for m in _CONDITION_RE.finditer(clause_text):
        _add(out, m.group(0), "condition", 0.88)
    # Weak/descriptive leads: only a supporting boost, tighter span.
    for m in _WEAK_CONSEQUENCE_RE.finditer(clause_text):
        _add(out, m.group(0), "obligation", 0.72)

    # Required levels: "minimum of 50% of marks", "minimum of 75%".
    for m in _MIN_MAX_RE.finditer(clause_text):
        _add(out, m.group(0), "threshold", 0.92)

    # Boundaries: comparisons and ranges (e.g. "less than 80%", "60% to 80%").
    for m in _COMPARISON_RE.finditer(clause_text):
        _add(out, m.group(0), "comparison", 0.92)
    for m in _RANGE_RE.finditer(clause_text):
        _add(out, m.group(0), "range", 0.90)

    # Bare numeric limits: amounts, percentages, quantities, durations.
    for m in _AMOUNT_RE.finditer(clause_text):
        _add(out, m.group(0), "amount", 0.85)
    for m in _PERCENT_RE.finditer(clause_text):
        _add(out, m.group(0), "threshold", 0.85)
    for m in _QUANTITY_RE.finditer(clause_text):
        _add(out, m.group(0), "quantity", 0.82)
    for m in _DURATION_RE.finditer(clause_text):
        _add(out, m.group(0), "duration", 0.80)

    # Sort by boost desc so callers that don't re-rank still get the best first.
    items = list(out.values())
    items.sort(key=lambda r: r["constraint_boost"], reverse=True)
    return items
