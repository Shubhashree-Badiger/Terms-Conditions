"""
Threshold sweep (Phase 17/18): find min_score + top_k that maximize F1
against the real answer key at data/annotations/EDU_DOC_EVAL.json.

Runs the pipeline ONCE per document (extract/segment/phrase/score are cached
via a single full-scored analysis), then re-applies different (min_score,
top_k) filters to the already-scored phrases and evaluates each combination.

Prints a table and the best-F1 setting.
"""
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml.preprocessing.pdf_extractor import extract_pdf
from ml.preprocessing.text_preprocessor import preprocess_document
from ml.clause_segmentation.clause_segmenter import segment_document
from ml.importance.phrase_extractor import PhraseExtractor
from ml.importance.importance_scorer import ImportanceScorer
from evaluation import phrase_metrics

RAW = os.path.join(PROJECT_ROOT, "data", "raw")
ANN = os.path.join(PROJECT_ROOT, "data", "annotations")
DOC_ID = "EDU_DOC_EVAL"
PDF = "Christ_University_Anti_Ragging_Regulations_Cleaned.pdf"


def main():
    gt_path = os.path.join(ANN, f"{DOC_ID}.json")
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)["clauses"]

    print("Analyzing + scoring all clauses once...")
    pp = preprocess_document(extract_pdf(os.path.join(RAW, PDF)))
    clauses = segment_document(pp)
    extractor = PhraseExtractor()
    scorer = ImportanceScorer(use_semantic=False)

    # score every candidate for every labeled clause, keep full sorted lists
    scored_by_clause = {}
    for c in clauses:
        if c["clause_id"] in gt:
            cands = extractor.extract(c["clause_text"])
            scored = scorer.score_candidates(cands, c["clause_text"])
            scored_by_clause[c["clause_id"]] = scored

    def evaluate(min_score, top_k):
        predicted = {}
        for cid, scored in scored_by_clause.items():
            kept = [s["phrase"] for s in scored if s["score"] >= min_score][:top_k]
            predicted[cid] = kept
        return phrase_metrics.evaluate_document(predicted, gt)

    print("\nSweep results (against your real answer key):")
    print(f"{'min_score':>9} {'top_k':>6} {'precision':>10} {'recall':>8} {'f1':>8}")
    print("-" * 46)

    combos = []
    for min_score in [0.45, 0.40, 0.35, 0.30, 0.25, 0.20]:
        for top_k in [6, 8, 10, 15]:
            r = evaluate(min_score, top_k)
            combos.append((min_score, top_k, r))
            print(f"{min_score:>9.2f} {top_k:>6} {r['precision']:>10.3f} "
                  f"{r['recall']:>8.3f} {r['f1']:>8.3f}")

    best = max(combos, key=lambda x: x[2]["f1"])
    ms, tk, r = best
    print("-" * 46)
    print(f"BEST F1: min_score={ms}, top_k={tk} -> "
          f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
          f"(tp={r['tp']} fp={r['fp']} fn={r['fn']})")

    # also show the current baseline for reference
    base = evaluate(0.45, 6)
    print(f"CURRENT (0.45, 6):  P={base['precision']:.3f} "
          f"R={base['recall']:.3f} F1={base['f1']:.3f}")


if __name__ == "__main__":
    main()
