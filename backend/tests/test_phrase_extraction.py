"""
Phase 7 tests: important phrase extraction (ml/importance/phrase_extractor).
Run from the backend directory with the venv pytest.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.importance.phrase_extractor import (
    extract_yake_keywords, extract_domain_terms, merge_candidates,
    clean_candidates, load_education_lexicon, PhraseExtractor,
)

SAMPLE = ("Students failing to maintain the minimum attendance requirement of 75% "
          "shall not be permitted to appear for the end semester examination and "
          "the fee paid is non-refundable.")


def test_yake_returns_phrases():
    kw = extract_yake_keywords(SAMPLE)
    assert len(kw) > 0
    phrases = [p.lower() for p, _ in kw]
    assert any("attendance" in p for p in phrases)


def test_domain_terms_matched():
    lex = load_education_lexicon()
    terms = extract_domain_terms(SAMPLE, lex)
    matched = {t for t, _ in terms}
    assert "non-refundable" in matched
    assert "shall not" in matched
    assert "75%" in matched
    # word-boundary: 'fee' should match but not spuriously
    assert "fee" in matched


def test_domain_word_boundary():
    lex = load_education_lexicon()
    # 'fee' must NOT match inside 'coffee'
    terms = extract_domain_terms("The coffee machine is broken.", lex)
    assert "fee" not in {t for t, _ in terms}


def test_merge_and_sources():
    kw = extract_yake_keywords(SAMPLE)
    lex = load_education_lexicon()
    terms = extract_domain_terms(SAMPLE, lex)
    merged = merge_candidates(kw, terms)
    # at least one candidate should be supported by both sources
    both = [c for c in merged if set(c["sources"]) == {"yake", "domain"}]
    assert len(both) >= 1


def test_clean_removes_stopwords_and_numbers():
    raw = [
        {"phrase": "the", "sources": ["yake"], "yake_score": 0.5, "category": None},
        {"phrase": "123", "sources": ["yake"], "yake_score": 0.5, "category": None},
        {"phrase": "75%", "sources": ["domain"], "yake_score": None, "category": "attendance"},
        {"phrase": "minimum attendance", "sources": ["yake"], "yake_score": 0.1, "category": None},
    ]
    cleaned = clean_candidates(raw)
    phrases = {c["phrase"] for c in cleaned}
    assert "the" not in phrases       # standalone stopword removed
    assert "123" not in phrases       # pure number removed
    assert "75%" in phrases           # percent kept
    assert "minimum attendance" in phrases


def test_extractor_end_to_end():
    ex = PhraseExtractor()
    cands = ex.extract(SAMPLE)
    assert len(cands) > 0
    phrases = {c["phrase"].lower() for c in cands}
    assert any("attendance" in p for p in phrases)
    # domain-only catch present
    assert any(c["category"] == "restrictions" for c in cands)


def test_empty_clause():
    ex = PhraseExtractor()
    assert ex.extract("") == []
    assert ex.extract("  ") == []
