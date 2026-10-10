"""
In-memory analysis progress tracker (Phase 23)
==============================================
Tracks the live stage of a background analysis so the frontend can poll it.

This is intentionally simple (a thread-safe dict). For a single-process
deployment it is sufficient; a multi-worker deployment would use Redis or the
database instead.

Stages (index -> label) mirror the frontend ProgressSteps:
    0 uploaded / queued
    1 extracting
    2 segmenting clauses
    3 finding phrases
    4 mapping highlights
    5 summarizing
    6 complete
   -1 error
"""
from __future__ import annotations

import threading
from typing import Dict, Optional

STAGE_LABELS = {
    0: "Queued",
    1: "Extracting text",
    2: "Detecting clauses",
    3: "Finding important phrases",
    4: "Mapping highlights",
    5: "Generating summaries",
    6: "Analysis complete",
    -1: "Error",
}

_lock = threading.Lock()
_progress: Dict[str, Dict] = {}


def set_stage(document_id: str, stage: int, message: str = "") -> None:
    with _lock:
        _progress[document_id] = {
            "document_id": document_id,
            "stage": stage,
            "label": STAGE_LABELS.get(stage, ""),
            "message": message,
            "done": stage == 6,
            "error": stage == -1,
        }


def get_progress(document_id: str) -> Optional[Dict]:
    with _lock:
        p = _progress.get(document_id)
        return dict(p) if p else None


def clear(document_id: str) -> None:
    with _lock:
        _progress.pop(document_id, None)
