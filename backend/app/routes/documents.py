"""
Document analysis + retrieval routes (Phase 14).

Endpoints:
    POST /api/analyze/{document_id}             run the pipeline
    GET  /api/documents/{document_id}           metadata / status
    GET  /api/documents/{document_id}/results   full analysis JSON
    GET  /api/documents/{document_id}/clauses   clauses only
    GET  /api/documents/{document_id}/annotated serve the annotated PDF
"""
import logging
import os
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.services import pipeline_service as svc
from app.services import progress_store
from app.services.results_store import (
    format_result, save_result, load_result, result_json_path,
)

logger = logging.getLogger("analyzer.documents")

router = APIRouter(prefix="/api", tags=["documents"])


def _run_analysis(document_id: str):
    """Run the full pipeline for a document, persisting results + progress.
    Returns the formatted result dict (or a scanned/error dict)."""
    pdf_path = svc.find_uploaded_pdf(document_id)
    if not pdf_path:
        progress_store.set_stage(document_id, -1, "PDF not found")
        return {"document_id": document_id, "status": "error",
                "error": "No uploaded PDF found."}

    pipeline = svc.get_pipeline()
    annotated_out = svc.annotated_pdf_path(document_id)

    def cb(stage):
        progress_store.set_stage(document_id, stage)

    raw = pipeline.analyze_document(pdf_path, document_id=document_id,
                                    annotated_out=annotated_out, progress_cb=cb)

    if raw.get("status") == "error":
        progress_store.set_stage(document_id, -1, raw.get("error", ""))
        return raw
    if raw.get("status") == "scanned":
        progress_store.set_stage(document_id, 6, raw.get("message", ""))
        return raw

    formatted = format_result(raw)
    save_result(formatted)

    try:
        from app.models.database import SessionLocal
        from app.services import db_service
        with SessionLocal() as db:
            db_service.save_analysis(db, formatted, annotated_pdf_path=annotated_out)
    except Exception as e:
        logger.warning("DB save_analysis failed for %s: %s", document_id, e)

    progress_store.set_stage(document_id, 6)
    return formatted


@router.post("/analyze/{document_id}")
def analyze(document_id: str, force: bool = False):
    """Run (or return cached) analysis synchronously and return the result."""
    if not force:
        cached = load_result(document_id)
        if cached is not None:
            return cached

    if not svc.find_uploaded_pdf(document_id):
        raise HTTPException(status_code=404,
                            detail=f"No uploaded PDF found for document_id '{document_id}'.")

    result = _run_analysis(document_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=422, detail=result.get("error", "Analysis failed."))
    return result


@router.post("/analyze-async/{document_id}")
def analyze_async(document_id: str, force: bool = False):
    """Start analysis in a background thread and return immediately.
    Poll GET /api/documents/{id}/progress to track progress, then fetch
    /results when complete."""
    if not force:
        cached = load_result(document_id)
        if cached is not None:
            progress_store.set_stage(document_id, 6)
            return {"document_id": document_id, "status": "cached", "stage": 6}

    if not svc.find_uploaded_pdf(document_id):
        raise HTTPException(status_code=404,
                            detail=f"No uploaded PDF found for document_id '{document_id}'.")

    progress_store.set_stage(document_id, 0, "Queued")

    def worker():
        try:
            _run_analysis(document_id)
        except Exception as e:  # never leave progress hanging
            logger.error("Background analysis failed for %s: %s", document_id, e)
            progress_store.set_stage(document_id, -1, str(e))

    threading.Thread(target=worker, daemon=True).start()
    return {"document_id": document_id, "status": "started", "stage": 0}


@router.get("/documents/{document_id}/progress")
def get_progress(document_id: str):
    """Return the live analysis progress for a document."""
    p = progress_store.get_progress(document_id)
    if p is None:
        # if a result already exists, report complete
        if load_result(document_id) is not None:
            return {"document_id": document_id, "stage": 6,
                    "label": "Analysis complete", "done": True, "error": False}
        raise HTTPException(status_code=404,
                            detail=f"No analysis in progress for '{document_id}'.")
    return p


@router.get("/documents/{document_id}")
def get_document(document_id: str):
    """Return lightweight metadata / status for a document."""
    result = load_result(document_id)
    pdf_path = svc.find_uploaded_pdf(document_id)

    if result is None and pdf_path is None:
        raise HTTPException(status_code=404, detail=f"Unknown document_id '{document_id}'.")

    if result is None:
        return {"document_id": document_id, "status": "uploaded",
                "analyzed": False,
                "filename": os.path.basename(pdf_path) if pdf_path else None}

    return {
        "document_id": document_id,
        "status": result.get("status"),
        "analyzed": result.get("status") == "analyzed",
        "filename": result.get("filename"),
        "total_pages": result.get("total_pages"),
        "total_clauses": result.get("total_clauses"),
        "processing_seconds": result.get("processing_seconds"),
        "stats": result.get("stats"),
        "annotated_pdf": result.get("annotated_pdf"),
    }


@router.get("/documents/{document_id}/results")
def get_results(document_id: str):
    """Return the full formatted analysis JSON."""
    result = load_result(document_id)
    if result is None:
        raise HTTPException(status_code=404,
                            detail=f"No analysis results for '{document_id}'. "
                                   f"Run POST /api/analyze/{document_id} first.")
    return result


@router.get("/documents/{document_id}/clauses")
def get_clauses(document_id: str):
    """Return just the clauses array for a document."""
    result = load_result(document_id)
    if result is None:
        raise HTTPException(status_code=404,
                            detail=f"No analysis results for '{document_id}'.")
    return {"document_id": document_id,
            "total_clauses": result.get("total_clauses", 0),
            "clauses": result.get("clauses", [])}


@router.get("/documents/{document_id}/annotated")
def get_annotated_pdf(document_id: str):
    """Serve the annotated PDF for a document."""
    path = svc.annotated_pdf_path(document_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404,
                            detail=f"No annotated PDF for '{document_id}'. "
                                   f"Run analysis first.")
    return FileResponse(path, media_type="application/pdf",
                        filename=f"{document_id}_annotated.pdf")


@router.get("/documents/{document_id}/original")
def get_original_pdf(document_id: str):
    """Serve the ORIGINAL uploaded PDF (used by the interactive PDF.js viewer,
    which draws its own overlays on top of the clean document)."""
    path = svc.find_uploaded_pdf(document_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404,
                            detail=f"No uploaded PDF for '{document_id}'.")
    return FileResponse(path, media_type="application/pdf",
                        filename=f"{document_id}.pdf")
