"""
Document ID generation.

Produces sequential, human-readable document IDs like EDU_DOC_001.
The next number is derived from the files already present in the uploads
directory so IDs remain stable across restarts (no DB dependency yet;
the database layer arrives in Phase 15).
"""
import re
from pathlib import Path

from app import config


_ID_RE = re.compile(rf"^{re.escape(config.DOC_ID_PREFIX)}(\d+)")


def _highest_existing_index(uploads_dir: Path) -> int:
    """Scan the uploads directory for existing EDU_DOC_<n>__ files and return
    the highest index found (0 if none)."""
    highest = 0
    if not uploads_dir.exists():
        return highest
    for p in uploads_dir.iterdir():
        m = _ID_RE.match(p.name)
        if m:
            try:
                idx = int(m.group(1))
                highest = max(highest, idx)
            except ValueError:
                continue
    return highest


def generate_document_id(uploads_dir: Path = None) -> str:
    """Return the next sequential document id, e.g. 'EDU_DOC_001'."""
    uploads_dir = uploads_dir or config.UPLOADS_DIR
    next_index = _highest_existing_index(uploads_dir) + 1
    return f"{config.DOC_ID_PREFIX}{next_index:03d}"


def stored_filename(document_id: str, original_filename: str) -> str:
    """Build the on-disk filename that embeds the document id.

    Example: EDU_DOC_001__hostel_rules.pdf
    This lets us recover the id from the filename and keeps the original name.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", original_filename).strip("_")
    return f"{document_id}__{safe}"
