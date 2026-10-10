"""
TEST 5: T5-small (Hugging Face Transformers)
- Generate a one-line summary of an educational clause
NOTE: First run downloads t5-small (~240MB).
"""
from transformers import T5Tokenizer, T5ForConditionalGeneration

CLAUSE = (
    "Students failing to maintain the minimum attendance requirement of 75% "
    "shall not be permitted to appear for the end semester examination."
)


def run():
    print("=" * 60)
    print("TEST 5: T5-small - One-Line Summarization")
    print("=" * 60)
    print("Loading t5-small (first run downloads ~240MB)...")

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    prompt = "summarize: " + CLAUSE
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    output_ids = model.generate(
        inputs["input_ids"],
        max_length=40,
        min_length=8,
        num_beams=4,
        early_stopping=True,
    )
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"\nOriginal clause:\n  {CLAUSE}")
    print(f"\nGenerated summary:\n  {summary}")

    assert summary and len(summary.strip()) > 0, "Summary is empty"
    print("\n[TEST 5 PASSED] T5-small produced a summary.")


if __name__ == "__main__":
    run()
