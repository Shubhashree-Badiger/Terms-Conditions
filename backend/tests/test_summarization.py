"""
Phase 11 tests: clause summarization (ml/summarization/summarizer).
Uses use_model=False (extractive path) for fast, deterministic tests.
Run from the backend directory with the venv pytest.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.summarization.summarizer import (
    ClauseSummarizer, _most_informative_sentence, _shorten, _critical_tokens,
)


def _summ():
    return ClauseSummarizer(use_model=False)


def test_empty_returns_notice():
    s = _summ()
    out = s.summarize("")
    assert "refer to the original" in out.lower()


def test_short_clause_returned():
    s = _summ()
    out = s.summarize("Students must carry their ID card.")
    assert "id card" in out.lower()


def test_preserves_percentage():
    s = _summ()
    clause = ("This section is about attendance. Students failing to maintain the "
              "minimum attendance requirement of 75% shall be debarred from the exam.")
    out = s.summarize(clause)
    assert "75%" in out


def test_extracts_most_informative_sentence():
    text = ("This regulation governs conduct. Students must pay a penalty of Rs. 500 "
            "for late fees. Records are kept.")
    best = _most_informative_sentence(text)
    assert "500" in best


def test_shorten_limits_length():
    long = " ".join(["word"] * 40)
    out = _shorten(long, max_words=22)
    assert out.endswith("...")
    assert len(out.split()) <= 23  # 22 words + ellipsis token


def test_critical_tokens_detects_numbers_and_consequences():
    toks = _critical_tokens("A penalty of 75% applies and students may be expelled.")
    assert "75%" in toks
    assert "penalty" in toks
    assert "expelled" in toks


def test_summary_is_faithful_prefix_or_sentence():
    # extractive summary must be a substring-derived piece of the source
    s = _summ()
    clause = ("Students who withdraw after the deadline must submit an application "
              "with a fee of Rs. 200 within seven working days.")
    out = s.summarize(clause).rstrip(" .")
    # first several words of the summary should appear in the source
    head = " ".join(out.split()[:6]).rstrip(".")
    assert head.lower() in clause.lower()


def test_multi_sentence_picks_key_clause():
    s = _summ()
    clause = ("This regulation governs student conduct. Students failing to maintain "
              "75% attendance shall not appear for the examination. Attendance is "
              "recorded daily by faculty.")
    out = s.summarize(clause)
    assert "75%" in out
    assert "governs student conduct" not in out.lower()  # filler dropped
