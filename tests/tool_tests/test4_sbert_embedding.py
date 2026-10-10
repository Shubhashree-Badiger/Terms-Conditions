"""
TEST 4: Sentence Transformer (all-MiniLM-L6-v2)
- Generate an embedding for a clause
- Verify embedding dimension and compute a similarity as a sanity check
NOTE: First run downloads the model (~80MB).
"""
from sentence_transformers import SentenceTransformer, util

CLAUSE_A = "Students must maintain at least 75% attendance to sit the exam."
CLAUSE_B = "A minimum of 75 percent attendance is required to take the examination."
CLAUSE_C = "The hostel gate closes at 9 PM every day."


def run():
    print("=" * 60)
    print("TEST 4: Sentence-BERT (all-MiniLM-L6-v2) Embeddings")
    print("=" * 60)
    print("Loading model (first run downloads ~80MB)...")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode([CLAUSE_A, CLAUSE_B, CLAUSE_C], convert_to_tensor=True)

    print(f"Embedding dimension: {emb.shape[1]}")
    assert emb.shape[1] == 384, "Expected 384-dim MiniLM embeddings"

    sim_ab = util.cos_sim(emb[0], emb[1]).item()
    sim_ac = util.cos_sim(emb[0], emb[2]).item()

    print(f"\nSimilarity (A vs B, both about attendance): {sim_ab:.3f}")
    print(f"Similarity (A vs C, attendance vs hostel):   {sim_ac:.3f}")

    assert sim_ab > sim_ac, "Similar clauses should score higher than unrelated ones"
    print("\n[TEST 4 PASSED] Embeddings capture semantic similarity correctly.")


if __name__ == "__main__":
    run()
