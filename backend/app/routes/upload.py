"""
Upload route (Phase 3).

POST /api/upload
  - accepts multipart/form-data with a single 'file' field (PDF)
  - validates extension, MIME type, size and real PDF structure
  - assigns a sequential document_id (EDU_DOC_001, ...)
  - stores the PDF in backend/uploads/ as <document_id>__<original_name>
  - returns { document_id, filename, status, total_pages, size_kb }
"""
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException

from app import config
from app.models.schemas import UploadResponse
from app.utils import validation
from app.utils.doc_id import generate_document_id, stored_filename

logger = logging.getLogger("analyzer.upload")

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Validate and store an uploaded educational-policy PDF."""
    # Read the file bytes once
    data = await file.read()

    # Validate (raises ValidationError -> HTTP 400)
    try:
        page_count = validation.validate_upload(
            filename=file.filename or "",
            content_type=file.content_type or "",
            data=data,
        )
    except validation.ValidationError as e:
        logger.warning("Upload rejected: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    # Assign id and persist to disk
    config.ensure_directories()
    document_id = generate_document_id()
    out_name = stored_filename(document_id, file.filename or "document.pdf")
    out_path = config.UPLOADS_DIR / out_name

    try:
        with open(out_path, "wb") as f:
            f.write(data)
    except OSError as e:
        logger.error("Failed to save upload: %s", e)
        raise HTTPException(status_code=500, detail="Failed to store the uploaded file.")

    size_kb = round(len(data) / 1024, 1)
    logger.info("Stored %s (%s, %d pages, %.1f KB)",
                document_id, out_name, page_count, size_kb)

    # register in the database (best-effort; never fail the upload on DB error)
    try:
        from app.models.database import SessionLocal
        from app.services import db_service
        with SessionLocal() as db:
            db_service.register_upload(db, document_id,
                                       file.filename or out_name,
                                       page_count, str(out_path))
    except Exception as e:
        logger.warning("DB register_upload failed for %s: %s", document_id, e)

    return UploadResponse(
        document_id=document_id,
        filename=file.filename or out_name,
        status="uploaded",
        total_pages=page_count,
        size_kb=size_kb,
    )
