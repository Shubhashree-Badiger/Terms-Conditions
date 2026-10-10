"""
TEST 2: spaCy
- Load the English model
- Process a sample educational clause
- Detect sentence boundaries
"""
import spacy

SAMPLE = (
    "Students must maintain a minimum attendance of 75% in each course. "
    "Failure to meet this requirement will result in the student being debarred "
    "from appearing in the end-semester examination. A student may apply for "
    "condonation of shortage only on valid medical grounds."
)


def run():
    print("=" * 60)
    print("TEST 2: spaCy - Sentence Boundary Detection")
    print("=" * 60)

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(SAMPLE)

    sentences = list(doc.sents)
    print(f"Input length: {len(SAMPLE)} chars")
    print(f"Detected sentences: {len(sentences)}\n")
    for i, sent in enumerate(sentences, 1):
        print(f"  [{i}] {sent.text.strip()}")

    assert len(sentences) >= 3, "Expected at least 3 sentences"
    print("\n[TEST 2 PASSED] Sentence segmentation works.")


if __name__ == "__main__":
    run()
