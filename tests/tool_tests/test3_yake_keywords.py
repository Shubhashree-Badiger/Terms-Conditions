"""
TEST 3: YAKE
- Extract keywords / key phrases from a sample educational clause
"""
import yake

SAMPLE = (
    "Students failing to maintain the minimum attendance requirement of 75% "
    "shall not be permitted to appear for the end semester examination and "
    "the fee paid is non-refundable under any circumstances."
)


def run():
    print("=" * 60)
    print("TEST 3: YAKE - Keyword / Phrase Extraction")
    print("=" * 60)

    # n=3 => allow up to 3-word phrases; top=10 keywords
    extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=10)
    keywords = extractor.extract_keywords(SAMPLE)

    print("Extracted key phrases (lower YAKE score = more important):\n")
    # YAKE returns (phrase, score); sort ascending by score
    for phrase, score in sorted(keywords, key=lambda x: x[1]):
        print(f"  {score:.4f}   {phrase}")

    assert len(keywords) > 0, "YAKE returned no keywords"
    print("\n[TEST 3 PASSED] YAKE keyword extraction works.")


if __name__ == "__main__":
    run()
