"""
Upload validation utilities.

Validates an uploaded file before it is accepted:
  - file extension (.pdf)
  - declared MIME type (application/pdf)
  - file size (<= MAX_UPLOAD_SIZE_MB)
  - real PDF structure (magic header + PyMuPDF can open it)

Raises ValueError with a clear message on failure so the route can convert
it into an HTTP 400 response.
"""
import os

import pymupdf  # PyMuPDF

from app import config


class ValidationError(ValueError):
    """Raised when an uploaded file fails validation."""


def validate_extension(filename: str) -> None:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file extension '{ext}'. Allowed: "
            f"{', '.join(sorted(config.ALLOWED_EXTENSIONS))}"
        )


def validate_mime(content_type: str) -> None:
    # content_type may include parameters, e.g. 'application/pdf; charset=...'
    base = (content_type or "").split(";")[0].strip().lower()
    if base not in config.ALLOWED_MIME_TYPES:
        raise ValidationError(
            f"Unsupported MIME type '{content_type}'. Expected application/pdf."
        )


def validate_size(size_bytes: int) -> None:
    max_bytes = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if size_bytes <= 0:
        raise ValidationError("Uploaded file is empty.")
    if size_bytes > max_bytes:
        mb = size_bytes / (1024 * 1024)
        raise ValidationError(
            f"File too large ({mb:.1f} MB). Maximum allowed is "
            f"{config.MAX_UPLOAD_SIZE_MB} MB."
        )


def validate_pdf_bytes(data: bytes) -> int:
    """Validate that the bytes are a real, openable PDF.

    Returns the page count on success.
    Raises ValidationError if the structure is invalid or the PDF is encrypted.
    """
    # Quick magic-number check
    if not data[:5].startswith(b"%PDF-"):
        raise ValidationError("File does not appear to be a valid PDF "
                              "(missing %PDF- header).")
    # Deep check: can PyMuPDF actually open it?
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise ValidationError(f"PDF could not be opened: {e}")

    try:
        if doc.needs_pass:
            raise ValidationError("PDF is password-protected and cannot be processed.")
        page_count = doc.page_count
        if page_count < 1:
            raise ValidationError("PDF has no pages.")
    finally:
        doc.close()

    return page_count


def validate_upload(filename: str, content_type: str, data: bytes) -> int:
    """Run all validation checks. Returns page count on success."""
    validate_extension(filename)
    validate_mime(content_type)
    validate_size(len(data))
    return validate_pdf_bytes(data)
