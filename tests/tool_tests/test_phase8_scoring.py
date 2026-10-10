"""
Phase 8 verification: importance scoring with explainable signal breakdown.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml.importance.phrase_extractor import PhraseExtractor
from ml.importance.importance_scorer import ImportanceScorer


def run():
    print("=" * 70)
    print("PHASE 8: Importance Scoring (explainable, deterministic)")
    print("=" * 70)

    extractor = PhraseExtractor()
    print("Loading MiniLM for semantic signal (first run may download)...")
    scorer = ImportanceScorer(use_semantic=True)
    print(f"Weights: {scorer.weights}")
    print(f"Semantic model active: {scorer.use_semantic}")

    samples = [
        ("Attendance/penalty clause",
         "Students failing to maintain the minimum attendance requirement of 75% "
         "shall not be permitted to appear for the end semester examination and "
         "the fee paid is non-refundable."),
        ("Low-risk informational clause",
         "Students are encouraged to carry their identity card while on campus."),
        ("Discipline clause",
         "Any student found indulging in ragging shall be liable for suspension "
         "or expulsion from the institution."),
    ]

    for title, clause in samples:
        print(f"\n### {title}")
        print(f"  {clause}")
        cands = extractor.extract(clause)
        scored = scorer.score_candidates(cands, clause)
        print(f"  Top scored phrases:")
        print(f"  {'PHRASE':<32} {'SCORE':>6} {'LEVEL':<7} {'RISK':<12} SIGNALS(y/d/s/p)")
        for s in scored[:7]:
            sig = s["signals"]
            sigstr = f"{sig['yake']:.2f}/{sig['domain']:.2f}/{sig['semantic']:.2f}/{sig['policy_rule']:.2f}"
            risk = f"{s['risk_type'] or '-'}({s['risk_level'][0]})"
            print(f"  {s['phrase'][:32]:<32} {s['score']:>6.3f} {s['level']:<7} {risk:<12} {sigstr}")

    # Determinism check
    print("\n--- Determinism check (same input twice) ---")
    c = extractor.extract(samples[0][1])
    r1 = scorer.score_candidates(c, samples[0][1])
    r2 = scorer.score_candidates(c, samples[0][1])
    same = all(a["score"] == b["score"] for a, b in zip(r1, r2))
    print(f"  Identical scores on repeat: {same}")

    print("\n" + "=" * 70)
    print("[PHASE 8 CHECK COMPLETE]")


if __name__ == "__main__":
    run()
