"""
Central configuration for the Education Policy & T&C Analyzer backend.

All paths are derived from the project root so the app works regardless of
where it is launched from. Model names and tunable thresholds live here so
they can be changed in one place.
"""
import os
from pathlib import Path

# --- Project paths -------------------------------------------------------
# config.py is at: <root>/backend/app/config.py  -> root is 3 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]

BACKEND_DIR = PROJECT_ROOT / "backend"
UPLOADS_DIR = BACKEND_DIR / "uploads"
BACKEND_OUTPUTS_DIR = BACKEND_DIR / "outputs"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = DATA_DIR / "extracted"
CLEANED_DIR = DATA_DIR / "cleaned"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
METADATA_DIR = DATA_DIR / "metadata"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ANNOTATED_PDFS_DIR = OUTPUTS_DIR / "annotated_pdfs"
OUTPUT_JSON_DIR = OUTPUTS_DIR / "json"
OUTPUT_SUMMARIES_DIR = OUTPUTS_DIR / "summaries"

CONFIGS_DIR = PROJECT_ROOT / "configs"
RISK_LEXICON_PATH = CONFIGS_DIR / "risk_lexicon.json"
EDUCATION_LEXICON_PATH = CONFIGS_DIR / "education_lexicon.json"
MODEL_CONFIG_PATH = CONFIGS_DIR / "model_config.json"

# --- Database ------------------------------------------------------------
DATABASE_PATH = BACKEND_DIR / "analyzer.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# --- Models --------------------------------------------------------------
SPACY_MODEL = "en_core_web_sm"
SBERT_MODEL = "all-MiniLM-L6-v2"
T5_MODEL = "t5-small"

# --- Upload validation ---------------------------------------------------
MAX_UPLOAD_SIZE_MB = 25
ALLOWED_EXTENSIONS = {".pdf"}
ALLOWED_MIME_TYPES = {"application/pdf"}

# --- Importance scoring thresholds --------------------------------------
# Normalized score in [0, 1]
IMPORTANCE_HIGH = 0.66
IMPORTANCE_MEDIUM = 0.33

# --- YAKE settings -------------------------------------------------------
YAKE_MAX_NGRAM = 3
YAKE_TOP_K = 15
YAKE_DEDUP_LIM = 0.9

# --- Summarization -------------------------------------------------------
SUMMARY_MAX_TOKENS = 48
SUMMARY_MIN_TOKENS = 8

# --- Phrase selection (tuned in Phase 17 against a real answer key) ------
# min_score 0.30 + top_k 10 balances recall vs precision (F1 ~0.39 vs the
# original 0.45/6 which gave F1 ~0.29 on the anti-ragging evaluation doc).
ANALYZE_MIN_SCORE = 0.30
ANALYZE_TOP_PHRASES = 10

# --- Document id prefix --------------------------------------------------
DOC_ID_PREFIX = "EDU_DOC_"


def ensure_directories():
    """Create all runtime directories if they do not already exist."""
    for d in [
        UPLOADS_DIR, BACKEND_OUTPUTS_DIR,
        EXTRACTED_DIR, CLEANED_DIR, PROCESSED_DIR, ANNOTATIONS_DIR, METADATA_DIR,
        ANNOTATED_PDFS_DIR, OUTPUT_JSON_DIR, OUTPUT_SUMMARIES_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print("Project root:", PROJECT_ROOT)
    print("Directories ensured.")
