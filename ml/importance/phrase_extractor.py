"""
Phrase Extractor (Phase 7) - CORE INTELLIGENCE
==============================================
Extracts candidate important phrases from a single clause by combining:

  1. YAKE unsupervised keyword extraction (finds statistically salient phrases)
  2. An education-policy domain lexicon (guarantees domain terms are not missed)

Candidates from both sources are merged, de-duplicated, cleaned (stopword and
noise removal) and returned with provenance so later scoring (Phase 8) can use
the source signals.

Public API:
    class PhraseExtractor:
        __init__(education_lexicon_path=None, ...)
        extract(clause_text) -> list[dict]

    Module helpers (usable standalone):
        extract_yake_keywords(text, ...) -> list[(phrase, yake_score)]
        extract_domain_terms(text, lexicon) -> list[(phrase, category)]
        merge_candidates(yake_kw, domain_terms) -> list[dict]
        clean_candidates(candidates) -> list[dict]

Candidate record shape:
    {
        "phrase": "minimum attendance of 75%",
        "sources": ["yake", "domain"],     # which methods proposed it
        "yake_score": 0.0016,               # lower = more salient (or None)
        "category": "attendance",           # domain category (or None)
    }
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Dict, Tuple, Optional

import yake

from ml.importance.constraint_extractor import extract_constraints

# Project config paths (fall back gracefully if backend app not importable)
try:
    from app import config as _cfg
    _DEFAULT_LEXICON = str(_cfg.EDUCATION_LEXICON_PATH)
    _YAKE_MAX_NGRAM = _cfg.YAKE_MAX_NGRAM
    _YAKE_TOP_K = _cfg.YAKE_TOP_K
    _YAKE_DEDUP = _cfg.YAKE_DEDUP_LIM
except Exception:
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _DEFAULT_LEXICON = os.path.join(_PROJECT_ROOT, "configs", "education_lexicon.json")
    _YAKE_MAX_NGRAM = 3
    _YAKE_TOP_K = 15
    _YAKE_DEDUP = 0.9


# Common English stopwords that should never stand alone as an important phrase
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "for", "with", "as", "at", "by", "from", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "it", "its", "he", "she",
    "they", "them", "his", "her", "their", "which", "who", "whom", "will",
    "shall", "may", "can", "such", "any", "all", "each", "per", "into", "upon",
}

# Words that add no meaning at the START or END of a highlighted phrase. We trim
# these so a highlight lands on the important core ("evaluated by the DAC" ->
# "evaluated by the DAC" keeps DAC; "for internal evaluation of" -> "internal
# evaluation"). Prepositions/articles/auxiliaries only - never trim content
# words, numbers, or %.
_EDGE_TRIM = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "as",
    "at", "by", "from", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "his", "her", "their",
    "which", "who", "will", "shall", "may", "can", "such", "into", "upon",
    "also", "same", "etc", "per", "he", "she", "they", "them",
}


def trim_phrase(phrase: str) -> str:
    """Strip leading/trailing prepositions, articles and auxiliaries so the
    highlight covers the meaningful core, not filler like 'for the' / 'of that'.

    Content words, numbers and percentages are never trimmed. If trimming would
    empty the phrase, the original is returned.
    """
    if not phrase:
        return phrase
    tokens = phrase.split()
    if len(tokens) <= 1:
        return phrase

    def is_trim(tok: str) -> bool:
        core = re.sub(r"[^a-z%]", "", tok.lower())
        # keep anything with a digit or % (numbers/limits are always meaningful)
        if re.search(r"[0-9%]", tok):
            return False
        return core in _EDGE_TRIM

    lo, hi = 0, len(tokens)
    while hi - lo > 1 and is_trim(tokens[lo]):
        lo += 1
    while hi - lo > 1 and is_trim(tokens[hi - 1]):
        hi -= 1
    trimmed = " ".join(tokens[lo:hi]).strip(" ,;:.-")
    return trimmed or phrase

# Phrases that are pure noise (numbers alone, single punctuation, etc.)
_PURE_NUMBER = re.compile(r"^[\d\W]+$")


# --------------------------------------------------------------------------
# YAKE extraction
# --------------------------------------------------------------------------
def extract_yake_keywords(text: str, max_ngram: int = _YAKE_MAX_NGRAM,
                          top_k: int = _YAKE_TOP_K,
                          dedup_lim: float = _YAKE_DEDUP) -> List[Tuple[str, float]]:
    """Return YAKE (phrase, score) pairs. Lower score = more important."""
    text = (text or "").strip()
    if len(text) < 3:
        return []
    extractor = yake.KeywordExtractor(
        lan="en", n=max_ngram, dedupLim=dedup_lim, top=top_k,
    )
    try:
        return extractor.extract_keywords(text)
    except Exception:
        return []


# --------------------------------------------------------------------------
# Domain lexicon matching
# --------------------------------------------------------------------------
def load_education_lexicon(path: str = _DEFAULT_LEXICON) -> Dict:
    """Load the education lexicon JSON. Returns {} on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _flatten_lexicon(lexicon: Dict) -> List[Tuple[str, str]]:
    """Return [(term, category), ...] from the lexicon's categories block."""
    out: List[Tuple[str, str]] = []
    for category, block in lexicon.get("categories", {}).items():
        for term in block.get("terms", []):
            out.append((term.lower(), category))
    return out


def extract_domain_terms(text: str, lexicon: Dict) -> List[Tuple[str, str]]:
    """Find domain lexicon terms present in the clause text.

    Returns [(matched_phrase, category), ...]. Matching is case-insensitive
    and uses word boundaries so 'fee' does not match 'coffee'.
    """
    if not text:
        return []
    lower = text.lower()
    found: List[Tuple[str, str]] = []
    for term, category in _flatten_lexicon(lexicon):
        # word-boundary search; term may contain spaces or symbols like 75% / shall not
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        if re.search(pattern, lower):
            found.append((term, category))
    return found


# --------------------------------------------------------------------------
# Merge + clean
# --------------------------------------------------------------------------
def _normalize_phrase(p: str) -> str:
    return re.sub(r"\s+", " ", p.strip().lower())


def merge_candidates(yake_kw: List[Tuple[str, float]],
                     domain_terms: List[Tuple[str, str]],
                     constraint_cands: Optional[List[Dict]] = None) -> List[Dict]:
    """Merge YAKE, domain and constraint candidates into unified records.

    Constraint candidates (thresholds, comparisons, conditions, consequences)
    carry a constraint_type / constraint_boost that the importance scorer uses
    to strongly prioritize the clause's boundaries and consequences.
    """
    merged: Dict[str, Dict] = {}

    for phrase, score in yake_kw:
        key = _normalize_phrase(phrase)
        if not key:
            continue
        rec = merged.setdefault(key, {
            "phrase": phrase.strip(), "sources": [], "yake_score": None,
            "category": None, "constraint_type": None, "constraint_boost": 0.0,
        })
        if "yake" not in rec["sources"]:
            rec["sources"].append("yake")
        rec["yake_score"] = score if rec["yake_score"] is None else min(rec["yake_score"], score)

    for term, category in domain_terms:
        key = _normalize_phrase(term)
        if not key:
            continue
        rec = merged.setdefault(key, {
            "phrase": term, "sources": [], "yake_score": None,
            "category": None, "constraint_type": None, "constraint_boost": 0.0,
        })
        if "domain" not in rec["sources"]:
            rec["sources"].append("domain")
        if rec["category"] is None:
            rec["category"] = category

    for cand in (constraint_cands or []):
        key = _normalize_phrase(cand.get("phrase", ""))
        if not key:
            continue
        rec = merged.setdefault(key, {
            "phrase": cand["phrase"], "sources": [], "yake_score": None,
            "category": None, "constraint_type": None, "constraint_boost": 0.0,
        })
        if "constraint" not in rec["sources"]:
            rec["sources"].append("constraint")
        # keep the strongest boost / most specific type
        if cand.get("constraint_boost", 0.0) > rec.get("constraint_boost", 0.0):
            rec["constraint_boost"] = cand["constraint_boost"]
            rec["constraint_type"] = cand.get("constraint_type")
        if rec["category"] is None and cand.get("category"):
            rec["category"] = cand["category"]

    return list(merged.values())


def clean_candidates(candidates: List[Dict]) -> List[Dict]:
    """Remove noise and trim filler words from the edges of each phrase.

    - drops pure numbers/punctuation (unless a % like "75%")
    - drops standalone stopwords
    - trims leading/trailing prepositions/articles so the highlight covers the
      meaningful core instead of "for the ..." / "... of that"
    """
    cleaned: List[Dict] = []
    seen = set()
    for c in candidates:
        phrase = (c.get("phrase") or "").strip()
        # Trim filler from the edges (keeps numbers/%/content words).
        phrase = trim_phrase(phrase)
        c["phrase"] = phrase
        norm = _normalize_phrase(phrase)
        if not norm or norm in seen:
            continue
        # drop pure numbers / punctuation UNLESS it carries a % (e.g. "75%")
        if _PURE_NUMBER.match(norm) and "%" not in norm:
            continue
        tokens = norm.split()
        # drop single stopword tokens (but keep domain/constraint matches)
        if (len(tokens) == 1 and tokens[0] in _STOPWORDS
                and "domain" not in c.get("sources", [])
                and "constraint" not in c.get("sources", [])):
            continue
        if not tokens:
            continue
        seen.add(norm)
        cleaned.append(c)
    return cleaned


# --------------------------------------------------------------------------
# High-level extractor
# --------------------------------------------------------------------------
class PhraseExtractor:
    """Extracts important candidate phrases from clauses (YAKE + domain lexicon)."""

    def __init__(self, education_lexicon_path: str = _DEFAULT_LEXICON,
                 max_ngram: int = _YAKE_MAX_NGRAM, top_k: int = _YAKE_TOP_K):
        self.lexicon = load_education_lexicon(education_lexicon_path)
        self.max_ngram = max_ngram
        self.top_k = top_k

    def extract(self, clause_text: str) -> List[Dict]:
        """Return cleaned candidate phrases for a single clause.

        Sources merged:
          * YAKE keywords          - statistical salience
          * domain lexicon terms   - guaranteed education terms
          * constraint phrases     - thresholds, comparisons, ranges,
                                     conditions and consequences (the clause's
                                     boundaries and "what happens if" spans)
        """
        yake_kw = extract_yake_keywords(clause_text, self.max_ngram, self.top_k)
        domain_terms = extract_domain_terms(clause_text, self.lexicon)
        constraint_cands = extract_constraints(clause_text)
        merged = merge_candidates(yake_kw, domain_terms, constraint_cands)
        return clean_candidates(merged)
