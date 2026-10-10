"""
Phase 17 tests: evaluation metrics (evaluation/*).
Run from the backend directory with the venv pytest.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation import phrase_metrics, summary_metrics, mapping_metrics


# ---- phrase P/R/F1 ----
def test_prf_perfect():
    r = phrase_metrics.prf(tp=5, fp=0, fn=0)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["f1"] == 1.0


def test_prf_half():
    r = phrase_metrics.prf(tp=1, fp=1, fn=1)
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and r["f1"] == 0.5


def test_evaluate_clause_matching():
    # tolerant containment matching: "attendance" matches gold "75% attendance"
    c = phrase_metrics.evaluate_clause(
        pred_phrases=["attendance", "unrelated"],
        gold_phrases=["75% attendance", "debarred"])
    assert c["tp"] == 1
    assert c["fp"] == 1     # "unrelated" is a false positive
    assert c["fn"] == 1     # "debarred" missed


def test_evaluate_document_none_without_gt():
    assert phrase_metrics.evaluate_document({"C1": ["x"]}, {}) is None


def test_evaluate_document_aggregate():
    pred = {"C1": ["ragging"], "C2": ["fee"]}
    gold = {"C1": ["ragging"], "C2": ["fee", "penalty"]}
    r = phrase_metrics.evaluate_document(pred, gold)
    assert r["tp"] == 2
    assert r["fn"] == 1      # 'penalty' missed
    assert r["clauses_evaluated"] == 2


# ---- ROUGE ----
def test_rouge_identical_is_one():
    s = summary_metrics.score_pair("students need 75% attendance",
                                   "students need 75% attendance")
    assert s["rouge1"] == 1.0
    assert s["rouge2"] == 1.0
    assert s["rougeL"] == 1.0


def test_rouge_no_overlap_is_zero():
    s = summary_metrics.score_pair("apple orange banana", "car train plane")
    assert s["rouge1"] == 0.0
    assert s["rouge2"] == 0.0
    assert s["rougeL"] == 0.0


def test_rouge_partial_overlap():
    s = summary_metrics.score_pair("students need attendance to sit exam",
                                   "students need attendance")
    assert 0.0 < s["rouge1"] <= 1.0
    assert s["rougeL"] > 0.0


def test_evaluate_summaries_none_without_refs():
    assert summary_metrics.evaluate_summaries({"C1": "x"}, {}) is None


def test_evaluate_summaries_average():
    pred = {"C1": "fees are non-refundable", "C2": "need 75% attendance"}
    refs = {"C1": "fees are non-refundable", "C2": "need 75% attendance"}
    r = summary_metrics.evaluate_summaries(pred, refs)
    assert r["rouge1"] == 1.0
    assert r["pairs_evaluated"] == 2


# ---- mapping ----
def test_mapping_metrics():
    result = {"clauses": [
        {"clause_id": "C1", "page_start": 1, "important_phrases": [
            {"text": "a", "locations": [{"page": 1, "bbox": [1, 2, 3, 4]}]},
            {"text": "b", "locations": []},
        ]},
        {"clause_id": "C2", "page_start": 2, "important_phrases": [
            {"text": "c", "locations": [{"page": 2, "bbox": [5, 6, 7, 8]}]},
        ]},
    ]}
    m = mapping_metrics.evaluate_mapping(result)
    assert m["total_phrases"] == 3
    assert m["mapped_phrases"] == 2
    assert m["failed_phrases"] == 1
    assert m["mapping_success_rate"] == round(2 / 3 * 100, 2)
    assert len(m["failures"]) == 1
    assert m["failures"][0]["phrase"] == "b"
