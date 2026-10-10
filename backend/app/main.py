"""
FastAPI application entry point.

At this scaffolding stage it only exposes a health check. The upload,
analyze, and results endpoints are added in later phases once the ML
pipeline modules are implemented and tested.

Run (from the backend directory):
    uvicorn app.main:app --reload
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.routes import upload, documents

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure runtime directories exist + create DB tables
    config.ensure_directories()
    from app.models.database import init_db
    init_db()
    yield
    # Shutdown: nothing to clean up yet


app = FastAPI(
    title="AI-Based Education Policy & T&C Analyzer",
    description="Analyzes educational institution policy PDFs clause by clause: "
                "extracts important phrases, scores importance, highlights them "
                "in the PDF, and generates one-line summaries.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS - allow the React dev server (added properly in the frontend phase)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routers
app.include_router(upload.router)
app.include_router(documents.router)


@app.get("/api/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "education-policy-analyzer", "version": "0.1.0"}


@app.get("/")
def root():
    return {"message": "Education Policy & T&C Analyzer API. See /docs for the API."}
