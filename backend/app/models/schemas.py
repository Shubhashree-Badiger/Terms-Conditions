"""
Pydantic schemas for API request/response bodies.
Extended in later phases as new endpoints are added.
"""
from pydantic import BaseModel


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    total_pages: int
    size_kb: float


class ErrorResponse(BaseModel):
    detail: str
