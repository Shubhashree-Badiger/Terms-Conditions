"""
Phase 8 tests: importance scoring + risk scoring.
Most tests run with use_semantic=False for speed (semantic signal is optional
and additive); one test exercises the full extractor->scorer chain.
Run from the backend directory with the venv pytest.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.importance.importance_scorer import ImportanceScorer
from ml.importance.risk_scorer import RiskScorer
from ml.importance.phrase_extractor import PhraseExtractor


def _scorer():
    # semantic disabled -> fast, deterministic, no model download
    return ImportanceScorer(use_semantic=False)


def test_assign_level_thresholds():
    s = _scorer()
    assert s.assign_level(0.9) == "HIGH"
    assert s.assign_level(0.5) == "MEDIUM"
    assert s.assign_level(0.1) == "LOW"


def test_domain_phrase_scores_higher_than_plain():
    s = _scorer()
    clause = "Fees once paid are non-refundable and a penalty applies."
    domain_c = {"phrase": "non-refundable", "sources": ["yake", "domain"],
                "yake_score": 0.05, "category": "fees"}
    plain_c = {"phrase": "once paid", "sources": ["yake"],
               "yake_score": 0.05, "category": None}
    ds = s.score_phrase(domain_c, clause)
    ps = s.score_phrase(plain_c, clause)
    assert ds["score"] > ps["score"]


def test_policy_signal_present():
    s = _scorer()
    c = {"phrase": "attendance", "sources": ["domain"],
         "yake_score": None, "category": "attendance"}
    with_penalty = s.score_phrase(c, "Students shall be debarred and must pay a penalty.")
    without = s.score_phrase(c, "Attendance is recorded daily for reference.")
    assert with_penalty["signals"]["policy_rule"] > without["signals"]["policy_rule"]


def test_score_record_shape():
    s = _scorer()
    c = {"phrase": "suspension", "sources": ["domain"],
         "yake_score": 0.02, "category": "discipline"}
    rec = s.score_phrase(c, "The student faces suspension.")
    for key in ("phrase", "score", "level", "category", "signals",
                "risk_type", "risk_level", "risk_evidence"):
        assert key in rec
    assert 0.0 <= rec["score"] <= 1.0


def test_determinism():
    s = _scorer()
    c = {"phrase": "non-refundable", "sources": ["yake", "domain"],
         "yake_score": 0.03, "category": "fees"}
    clause = "The fee is non-refundable."
    r1 = s.score_phrase(c, clause)
    r2 = s.score_phrase(c, clause)
    assert r1["score"] == r2["score"]


def test_risk_scorer_disciplinary_high():
    rs = RiskScorer()
    r = rs.score("expulsion", "The student will face expulsion for ragging.")
    assert r["risk_type"] in ("disciplinary", "anti_ragging")
    assert r["risk_level"] in ("HIGH", "MEDIUM")
    assert len(r["evidence"]) > 0


def test_risk_scorer_none_for_neutral():
    rs = RiskScorer()
    r = rs.score("library", "The library opens at eight in the morning.")
    assert r["risk_type"] is None
    assert r["risk_level"] == "NONE"


def test_full_chain_ranks_penalty_high():
    ex = PhraseExtractor()
    s = _scorer()
    clause = ("Students failing to maintain 75% attendance shall not be permitted "
              "to appear for the examination and the fee is non-refundable.")
    cands = ex.extract(clause)
    scored = s.score_candidates(cands, clause)
    assert scored[0]["score"] >= scored[-1]["score"]  # sorted desc
    # a domain phrase should appear near the top
    top_phrases = {r["phrase"].lower() for r in scored[:5]}
    assert any("attendance" in p or "refundable" in p for p in top_phrases)
