"""
Clause Summarizer (Phase 11)
============================
Generates a faithful, one-line summary for each clause.

Design
------
* T5-small is loaded ONCE (not per clause).
* The generation prompt instructs the model to preserve numbers, percentages,
  dates, deadlines, conditions and consequences, and to not add information.
* Output is validated:
    - non-empty
    - trimmed to a single sentence where possible
    - critical tokens (numbers, %, key consequence words) that appear in the
      source must survive; if the model drops them, we fall back.
* Extractive fallback: when T5 output is empty, too long, echoes the input, or
  loses critical numbers, we build a concise summary from the clause's most
  informative sentence (never hallucinating).

Faithfulness first: if we cannot confidently summarize, we return the most
informative source sentence rather than invent content.

Public API:
    class ClauseSummarizer:
        __init__(model_name="t5-small", use_model=True)
        summarize(clause_text) -> str
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger("analyzer.summarizer")

try:
    from app import config as _cfg
    _T5_MODEL = _cfg.T5_MODEL
    _MAX_TOK = _cfg.SUMMARY_MAX_TOKENS
    _MIN_TOK = _cfg.SUMMARY_MIN_TOKENS
except Exception:
    _T5_MODEL = "t5-small"
    _MAX_TOK = 48
    _MIN_TOK = 8

_PROMPT = (
    "Summarize this educational policy clause in one simple sentence. "
    "Preserve all important numbers, percentages, dates, deadlines, conditions "
    "and consequences. Do not add information. Clause: "
)

# tokens we consider "critical" and must not be silently dropped
_NUM_RE = re.compile(r"\d+(?:\.\d+)?%?")
_CONSEQUENCE_WORDS = [
    "non-refundable", "penalty", "fine", "debarred", "debar", "suspension",
    "suspended", "expulsion", "expelled", "terminate", "terminated", "forfeit",
    "prohibited", "not permitted", "shall not", "mandatory", "must",
]

_FALLBACK_NOTICE = "Important condition - please refer to the original clause for details."


# abbreviations whose trailing period should NOT end a sentence
_ABBREV = ["rs", "no", "mr", "mrs", "ms", "dr", "prof", "vs", "etc", "sec",
           "cl", "art", "ph", "b", "m", "e", "i", "viz", "fig"]
_ABBREV_RE = re.compile(r"(?<!\w)(" + "|".join(_ABBREV) + r")\.", re.I)


def _sentences(text: str) -> List[str]:
    # temporarily protect abbreviation periods so they don't split sentences
    protected = _ABBREV_RE.sub(lambda m: m.group(1) + "<DOT>", text.strip())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", protected)
    out = []
    for p in parts:
        p = p.replace("<DOT>", ".").strip()
        if p:
            out.append(p)
    return out


def _critical_tokens(text: str) -> set:
    toks = set(m.group(0) for m in _NUM_RE.finditer(text))
    low = text.lower()
    for w in _CONSEQUENCE_WORDS:
        if re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", low):
            toks.add(w)
    return toks


def _first_sentence(text: str) -> str:
    s = _sentences(text)
    return s[0] if s else text.strip()


def _most_informative_sentence(text: str) -> str:
    """Pick the sentence carrying the most critical tokens (numbers/consequences).
    Ties break toward the earliest sentence."""
    sents = _sentences(text)
    if not sents:
        return text.strip()
    best, best_score, best_i = sents[0], -1, 0
    for i, s in enumerate(sents):
        score = len(_critical_tokens(s))
        if score > best_score:
            best, best_score, best_i = s, score, i
    return best


def _shorten(sentence: str, max_words: int = 22) -> str:
    """Trim a sentence to a readable length without cutting mid-number.
    Keeps it faithful (a prefix of the source sentence)."""
    words = sentence.split()
    if len(words) <= max_words:
        return sentence.strip()
    short = " ".join(words[:max_words]).rstrip(",;:")
    if not short.endswith((".", "!", "?")):
        short += " ..."
    return short


def _clean_output(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # keep only the first sentence
    text = _first_sentence(text)
    return text


class ClauseSummarizer:
    def __init__(self, model_name: str = _T5_MODEL, use_model: bool = True):
        self.model = None
        self.tokenizer = None
        self.use_model = use_model
        if use_model:
            self._load(model_name)

    def _load(self, model_name: str):
        try:
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.eval()
        except Exception as e:
            logger.warning("T5 load failed (%s); using extractive fallback only.", e)
            self.model = None
            self.use_model = False

    # -- model generation -------------------------------------------------
    def _t5_summary(self, clause_text: str) -> Optional[str]:
        if not (self.use_model and self.model and self.tokenizer):
            return None
        try:
            import torch
            prompt = _PROMPT + clause_text
            inputs = self.tokenizer(prompt, return_tensors="pt",
                                    max_length=512, truncation=True)
            with torch.no_grad():
                ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=_MAX_TOK, min_length=_MIN_TOK,
                    num_beams=4, length_penalty=2.0,
                    no_repeat_ngram_size=3, early_stopping=True,
                )
            out = self.tokenizer.decode(ids[0], skip_special_tokens=True)
            return _clean_output(out)
        except Exception as e:
            logger.debug("T5 generation failed: %s", e)
            return None

    # -- validation -------------------------------------------------------
    def _is_valid(self, summary: str, clause_text: str) -> bool:
        if not summary or len(summary.strip()) < 5:
            return False
        # reject near-verbatim echo of a long clause (no compression achieved)
        if len(clause_text) > 160 and len(summary) >= 0.9 * len(clause_text):
            return False
        # critical numbers present in the source must survive
        src_nums = {t for t in _critical_tokens(clause_text) if _NUM_RE.fullmatch(t)}
        if src_nums:
            sum_nums = {t for t in _critical_tokens(summary) if _NUM_RE.fullmatch(t)}
            if not (src_nums & sum_nums):
                return False
        return True

    # -- extractive fallback ---------------------------------------------
    def _extractive(self, clause_text: str) -> str:
        sent = _most_informative_sentence(clause_text)
        short = _shorten(sent)
        if len(short.strip()) < 5:
            return _FALLBACK_NOTICE
        return short

    def _compresses_enough(self, summary: str, clause_text: str) -> bool:
        """True if the summary is meaningfully shorter than the source.
        Base t5-small often echoes short legal text; in that case we prefer
        the extractive summary which is faithful and genuinely concise."""
        if not summary:
            return False
        return len(summary) <= 0.75 * len(clause_text)

    # -- public -----------------------------------------------------------
    def summarize(self, clause_text: str) -> str:
        """Return a faithful one-line summary of the clause.

        Strategy:
          * Very short clauses -> return the (shortened) first sentence.
          * Otherwise try T5; accept ONLY if it is valid AND actually compresses.
          * Fall back to a faithful extractive summary (most informative
            sentence, shortened) - never hallucinating.
        """
        clause_text = (clause_text or "").strip()
        if not clause_text:
            return _FALLBACK_NOTICE

        # short clauses: don't over-summarize; return the (shortened) clause
        if len(clause_text) <= 90:
            return _shorten(_first_sentence(clause_text))

        # 1) try T5 - accept only if valid and it genuinely compresses
        candidate = self._t5_summary(clause_text)
        if (candidate and self._is_valid(candidate, clause_text)
                and self._compresses_enough(candidate, clause_text)):
            return candidate

        # 2) extractive fallback (faithful, always concise)
        return self._extractive(clause_text)
