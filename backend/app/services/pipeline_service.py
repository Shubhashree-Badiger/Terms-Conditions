"""
Pipeline service (Phase 14)
===========================
Holds the singleton AnalysisPipeline (models loaded once) and provides
helpers to locate uploaded PDFs by document_id and run analysis.

The pipeline is created lazily on first use (so importing the app is fast and
tests that never analyze don't pay the model-load cost).
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

from app import config

logger = logging.getLogger("analyzer.service")

# Ensure the project root is importable so `ml` package resolves when the
# backend is launched from the backend/ directory.
_PROJECT_ROOT = str(config.PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_pipeline = None


def get_pipeline():
    """Return the shared AnalysisPipeline, building it once on first call."""
    global _pipeline
    if _pipeline is None:
        from ml.analysis_pipeline import AnalysisPipeline
        # Semantic ON: clause-centrality (SBERT MiniLM) is the primary signal
        # for choosing which words are the important ones in a clause. The
        # model is small and loaded once. T5 stays off (extractive summaries).
        _pipeline = AnalysisPipeline(use_semantic=True, use_t5=False)
        logger.info("AnalysisPipeline initialized.")
    return _pipeline


def find_uploaded_pdf(document_id: str) -> Optional[str]:
    """Locate the stored PDF for a document_id in the uploads directory.

    Files are stored as <document_id>__<original_name>.pdf (see Phase 3).
    """
    uploads = config.UPLOADS_DIR
    if not uploads.exists():
        return None
    for p in uploads.iterdir():
        if p.is_file() and p.name.startswith(document_id + "__") and p.suffix.lower() == ".pdf":
            return str(p)
    return None


def annotated_pdf_path(document_id: str) -> str:
    """Path where the annotated PDF for a document is written."""
    return str(config.ANNOTATED_PDFS_DIR / f"{document_id}_annotated.pdf")
