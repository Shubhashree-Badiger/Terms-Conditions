"""
Importance Scorer (Phase 8)
===========================
Assigns an EXPLAINABLE, DETERMINISTIC importance score in [0, 1] to each
candidate phrase and maps it to a level: HIGH / MEDIUM / LOW.

The score is a weighted sum of documented signals. The dominant signal is
CLAUSE CENTRALITY: how well a phrase captures what the clause is actually about
(cosine between the phrase and the whole clause embedding). This directly
answers the user's need - "highlight the words the whole clause is about" -
instead of letting generic YAKE n-grams (e.g. title fragments) win.

  1. centrality          (0.40) - cosine(phrase, whole-clause) via MiniLM.
                                    High when the phrase is semantically central
                                    to the clause (its "aboutness").
  2. domain_match        (0.22) - phrase matches an education-lexicon category;
                                    scaled by that category's weight.
  3. policy_rule         (0.16) - presence of obligation/consequence language
                                    (must, shall, penalty, prohibited, ...).
  4. anchor_semantic     (0.12) - cosine similarity between the phrase and a set
                                    of "important policy concept" anchor sentences.
  5. yake_rank           (0.10) - RELATIVE YAKE salience within the clause. YAKE
                                    scores saturate on short clauses, so we use
                                    the phrase's rank among the clause candidates
                                    rather than the raw (inverted) score.

Weights come from configs/model_config.json (importance.weights) with a
hard-coded fallback. Nothing here is random; the same input always yields the
same score.

Public API:
    class ImportanceScorer:
        __init__(...)                      # loads MiniLM + lexicon once
        score_phrase(candidate, clause_text) -> dict
        score_candidates(candidates, clause_text) -> list[dict]
        assign_level(score) -> str
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Dict, List, Optional

# --- config / paths ------------------------------------------------------
try:
    from app import config as _cfg
    _MODEL_CFG_PATH = str(_cfg.MODEL_CONFIG_PATH)
    _EDU_LEXICON_PATH = str(_cfg.EDUCATION_LEXICON_PATH)
    _SBERT_MODEL = _cfg.SBERT_MODEL
    _HIGH = _cfg.IMPORTANCE_HIGH
    _MEDIUM = _cfg.IMPORTANCE_MEDIUM
except Exception:
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _MODEL_CFG_PATH = os.path.join(_ROOT, "configs", "model_config.json")
    _EDU_LEXICON_PATH = os.path.join(_ROOT, "configs", "education_lexicon.json")
    _SBERT_MODEL = "all-MiniLM-L6-v2"
    _HIGH, _MEDIUM = 0.66, 0.33

from ml.importance.phrase_extractor import load_education_lexicon
from ml.importance.risk_scorer import RiskScorer


# Obligation / consequence cue words for the policy-rule signal
_POLICY_CUES = [
    "must", "shall", "shall not", "required", "mandatory", "compulsory",
    "prohibited", "not permitted", "not allowed", "cannot", "will not",
    "penalty", "fine", "forfeit", "non-refundable", "debarred", "debar",
    "suspension", "suspend", "expulsion", "expel", "terminate", "termination",
    "liable", "disqualified", "cancellation", "deadline",
]

# Anchor concepts for semantic similarity (what "important" looks like)
_ANCHORS = [
    "Students must meet a mandatory requirement or face a penalty.",
    "There is a financial condition such as a non-refundable fee or fine.",
    "Disciplinary action such as suspension or expulsion may be taken.",
    "A deadline or eligibility condition must be satisfied.",
    "Personal data may be shared or consent is required.",
]


_DEFAULT_WEIGHTS = {
    "centrality": 0.32,
    "constraint": 0.28,
    "domain_match": 0.16,
    "policy_rule": 0.12,
    "anchor_semantic": 0.06,
    "yake_rank": 0.06,
}


def _load_weights() -> Dict[str, float]:
    """Load signal weights from model_config.json (importance.weights).

    Backward compatible: if the config still carries the OLD keys
    (yake_score / semantic_similarity), we ignore them and use the new
    centrality-first defaults, because the old mix let generic YAKE n-grams
    dominate the highlights.
    """
    try:
        with open(_MODEL_CFG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        w = cfg["importance"].get("weights", {})
        # Only trust the config if it uses the NEW signal names.
        if "centrality" in w:
            return {k: float(w.get(k, _DEFAULT_WEIGHTS[k])) for k in _DEFAULT_WEIGHTS}
        return dict(_DEFAULT_WEIGHTS)
    except Exception:
        return dict(_DEFAULT_WEIGHTS)


def _tokens(p: str) -> set:
    return set(re.sub(r"[^a-z0-9%\s]", " ", (p or "").lower()).split())


def _suppress_redundant(scored: List[Dict]) -> List[Dict]:
    """Collapse near-duplicate / overlapping phrases.

    When two selected phrases overlap heavily (one's words are a subset of the
    other, e.g. "Academic Regulations" vs "Academic Regulations Governing"),
    keep only the higher-scored one. This stops the highlights from stacking
    several fragments of the same span and frees slots for genuinely different
    important phrases.
    """
    ordered = sorted(scored, key=lambda x: x["score"], reverse=True)
    kept: List[Dict] = []
    kept_tokens: List[set] = []
    for s in ordered:
        toks = _tokens(s["phrase"])
        if not toks:
            continue
        redundant = False
        for kt in kept_tokens:
            inter = toks & kt
            if not inter:
                continue
            smaller = min(len(toks), len(kt))
            # >=80% of the smaller phrase's words overlap -> treat as duplicate
            if smaller and len(inter) / smaller >= 0.8:
                redundant = True
                break
        if not redundant:
            kept.append(s)
            kept_tokens.append(toks)
    return kept


class ImportanceScorer:
    def __init__(self, sbert_model: str = _SBERT_MODEL,
                 education_lexicon_path: str = _EDU_LEXICON_PATH,
                 use_semantic: bool = True):
        self.weights = _load_weights()
        self.lexicon = load_education_lexicon(education_lexicon_path)
        self._cat_weights = {
            cat: float(block.get("weight", 1.0))
            for cat, block in self.lexicon.get("categories", {}).items()
        }
        self._max_cat_weight = max(self._cat_weights.values(), default=1.0)
        self.risk_scorer = RiskScorer()

        # Load MiniLM once (optional; can be disabled for fast/offline use)
        self._model = None
        self._anchor_emb = None
        self.use_semantic = use_semantic
        if use_semantic:
            self._load_model(sbert_model)

    # -- model ------------------------------------------------------------
    def _load_model(self, name: str):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(name)
            self._anchor_emb = self._model.encode(
                _ANCHORS, convert_to_tensor=True, normalize_embeddings=True
            )
        except Exception:
            self._model = None
            self.use_semantic = False

    # -- individual signals ----------------------------------------------
    def _domain_signal(self, category: Optional[str]) -> float:
        if not category:
            return 0.0
        w = self._cat_weights.get(category, 1.0)
        return min(1.0, w / self._max_cat_weight)

    def _policy_signal(self, phrase: str, clause_text: str) -> float:
        text = f"{phrase} {clause_text}".lower()
        hits = 0
        for cue in _POLICY_CUES:
            if re.search(r"(?<!\w)" + re.escape(cue) + r"(?!\w)", text):
                hits += 1
        # saturate: 1 cue -> 0.5, 2 -> 0.75, 3+ -> ~1.0
        return 1.0 - math.pow(0.5, hits) if hits else 0.0

    def _encode(self, text):
        """Encode a single text to a normalized tensor (or None)."""
        if not (self.use_semantic and self._model is not None):
            return None
        try:
            return self._model.encode(text, convert_to_tensor=True,
                                      normalize_embeddings=True)
        except Exception:
            return None

    def _anchor_signal(self, phrase_emb) -> float:
        """Similarity of the phrase to generic 'important policy' anchors."""
        if phrase_emb is None or self._anchor_emb is None:
            return 0.0
        try:
            from sentence_transformers import util
            sims = util.cos_sim(phrase_emb, self._anchor_emb)
            return max(0.0, float(sims.max().item()))
        except Exception:
            return 0.0

    def _centrality_signal(self, phrase_emb, clause_emb) -> float:
        """How central the phrase is to THIS clause: cosine(phrase, clause).

        This is the key 'aboutness' signal. A phrase that captures what the
        clause is about scores high; a generic fragment scores low. Cosine
        for related short-vs-long texts typically lands ~0.2-0.6, so we
        rescale to spread that band across 0..1.
        """
        if phrase_emb is None or clause_emb is None:
            return 0.0
        try:
            from sentence_transformers import util
            cos = float(util.cos_sim(phrase_emb, clause_emb).item())
            cos = max(0.0, cos)
            # rescale: 0.20 -> ~0.13, 0.40 -> ~0.5, 0.60 -> ~0.87
            scaled = (cos - 0.15) / 0.35
            return max(0.0, min(1.0, scaled))
        except Exception:
            return 0.0

    # -- public -----------------------------------------------------------
    def assign_level(self, score: float) -> str:
        if score >= _HIGH:
            return "HIGH"
        if score >= _MEDIUM:
            return "MEDIUM"
        return "LOW"

    def score_phrase(self, candidate: Dict, clause_text: str,
                     clause_emb=None, yake_rank: float = 0.0) -> Dict:
        """Score one candidate phrase. Returns an enriched record.

        Args:
            clause_emb: precomputed embedding of the whole clause (reused across
                        all candidates of the clause; computed here if None).
            yake_rank:  relative YAKE salience within the clause in [0,1]
                        (1.0 = most salient candidate). Passed in by
                        score_candidates; defaults to 0 for standalone use.
        """
        phrase = candidate.get("phrase", "")
        category = candidate.get("category")

        if clause_emb is None:
            clause_emb = self._encode(clause_text)
        phrase_emb = self._encode(phrase)

        s_centrality = self._centrality_signal(phrase_emb, clause_emb)
        s_domain = self._domain_signal(category)
        s_policy = self._policy_signal(phrase, clause_text)
        s_anchor = self._anchor_signal(phrase_emb)
        s_yake = float(max(0.0, min(1.0, yake_rank)))
        # constraint signal: thresholds, comparisons, conditions, consequences
        s_constraint = float(max(0.0, min(1.0, candidate.get("constraint_boost", 0.0))))
        constraint_type = candidate.get("constraint_type")

        w = self.weights
        score = (
            w["centrality"] * s_centrality
            + w.get("constraint", 0.0) * s_constraint
            + w["domain_match"] * s_domain
            + w["policy_rule"] * s_policy
            + w["anchor_semantic"] * s_anchor
            + w["yake_rank"] * s_yake
        )
        # Score FLOOR for real constraints: a clause's boundary/consequence
        # (e.g. "less than 80%", "shall not be permitted to appear") must never
        # be filtered out below the highlight threshold, even if it is not
        # semantically central on its own. This guarantees the limits and
        # "what happens if" spans are always highlighted.
        if s_constraint >= 0.85:
            score = max(score, 0.62)
        elif s_constraint >= 0.80:
            score = max(score, 0.45)

        score = round(min(1.0, max(0.0, score)), 4)

        risk = self.risk_scorer.score(phrase, clause_text)

        return {
            "phrase": phrase,
            "score": score,
            "level": self.assign_level(score),
            "category": category,
            "constraint_type": constraint_type,
            "sources": candidate.get("sources", []),
            "signals": {
                "centrality": round(s_centrality, 4),
                "constraint": round(s_constraint, 4),
                "domain": round(s_domain, 4),
                "policy_rule": round(s_policy, 4),
                "anchor": round(s_anchor, 4),
                "yake_rank": round(s_yake, 4),
            },
            "risk_type": risk["risk_type"],
            "risk_level": risk["risk_level"],
            "risk_evidence": risk["evidence"],
        }

    def score_candidates(self, candidates: List[Dict], clause_text: str) -> List[Dict]:
        """Score all candidates for a clause, sorted by score descending.

        The whole-clause embedding is computed ONCE and reused for every
        candidate (the centrality signal). YAKE scores are converted to a
        relative RANK within the clause, because raw YAKE saturates on short
        clauses and stops discriminating.
        """
        if not candidates:
            return []

        clause_emb = self._encode(clause_text)

        # relative YAKE rank: best (lowest raw yake_score) -> 1.0
        yaked = [c for c in candidates if c.get("yake_score") is not None]
        yake_rank_map = {}
        if yaked:
            order = sorted(yaked, key=lambda c: c["yake_score"])  # lower=better
            n = len(order)
            for i, c in enumerate(order):
                # linear from 1.0 (best) down to ~0.1 (worst)
                yake_rank_map[id(c)] = 1.0 - (i / n) if n > 1 else 1.0

        scored = [
            self.score_phrase(c, clause_text, clause_emb=clause_emb,
                              yake_rank=yake_rank_map.get(id(c), 0.0))
            for c in candidates
        ]
        scored = _suppress_redundant(scored)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored
