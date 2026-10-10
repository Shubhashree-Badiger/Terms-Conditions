"""
Phase 5 tests: text preprocessing (ml/preprocessing/text_preprocessor).
Run from the backend directory with the venv pytest.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.preprocessing.text_preprocessor import (
    normalize_unicode, dehyphenate, join_soft_linebreaks, collapse_whitespace,
    is_page_number_line, is_boilerplate_line, detect_repeated_lines,
    preprocess_page,
)


def test_normalize_unicode():
    assert normalize_unicode("\u201cquoted\u201d") == '"quoted"'
    assert normalize_unicode("a\u2013b") == "a-b"       # en dash
    assert normalize_unicode("nbsp\u00a0here") == "nbsp here"


def test_dehyphenate():
    assert dehyphenate("attend-\nance") == "attendance"
    assert dehyphenate("exam-\nination rules") == "examination rules"


def test_join_soft_linebreaks():
    # lowercase -> lowercase across a break should be joined by a space
    out = join_soft_linebreaks("the student\nmust attend")
    assert out == "the student must attend"


def test_collapse_whitespace():
    assert collapse_whitespace("a    b\t\tc") == "a b c"
    assert collapse_whitespace("x\n\n\n\ny") == "x\n\ny"


def test_page_number_detection():
    assert is_page_number_line("12")
    assert is_page_number_line("Page 3 of 40")
    assert is_page_number_line("1 | P a g e")
    assert not is_page_number_line("Rule 12 applies")


def test_boilerplate_detection():
    assert is_boilerplate_line("All rights reserved.")
    assert is_boilerplate_line("Printed on 01/01/2024")
    assert not is_boilerplate_line("Students must attend classes.")


def test_detect_repeated_lines():
    pages = [
        "College Header\nContent about fees on page one.\nFooter Line",
        "College Header\nContent about hostel on page two.\nFooter Line",
        "College Header\nContent about exams on page three.\nFooter Line",
        "College Header\nContent about rules on page four.\nFooter Line",
    ]
    repeated = detect_repeated_lines(pages)
    assert "college header" in repeated
    assert "footer line" in repeated
    # unique content lines should NOT be flagged
    assert "content about fees on page one." not in repeated


def test_preprocess_page_removes_headers():
    repeated = {"college header"}
    text = "College Header\n12\nStudents must maintain 75% attendance."
    out = preprocess_page(text, repeated)
    assert "College Header" not in out    # header removed
    assert "12" not in out.split("\n")    # page number removed
    assert "attendance" in out.lower()    # content kept


def test_original_preserved_in_document():
    # Ensure preprocessing keeps original text separate
    from ml.preprocessing.pdf_extractor import extract_pdf
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Fees are non-refundable.")
    data = doc.tobytes()
    doc.close()

    from ml.preprocessing.text_preprocessor import preprocess_document
    extracted = extract_pdf(data)
    pp = preprocess_document(extracted)
    assert "non-refundable" in pp.original_text.lower()
    assert "non-refundable" in pp.normalized_text.lower()
