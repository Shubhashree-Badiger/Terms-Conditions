# AI-Based Intelligent Education Policy & Terms and Conditions Analyzer

Analyzes educational institution policy / regulation / terms-and-conditions PDFs
**clause by clause** and helps students understand the important parts without
reading the whole document.

For every clause the system:
1. Identifies important words / phrases
2. Assigns an importance score and level (HIGH / MEDIUM / LOW)
3. Locates those phrases in the original PDF (coordinates)
4. Highlights them in an annotated copy of the PDF
5. Generates a simple one-line summary
6. Returns structured JSON

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13 |
| PDF processing | PyMuPDF (fitz) |
| NLP | spaCy, YAKE |
| Semantic NLP | Sentence-Transformers (all-MiniLM-L6-v2) |
| Summarization | Hugging Face Transformers, PyTorch, T5-small |
| Backend | FastAPI, Uvicorn, Pydantic |
| Database | SQLite + SQLAlchemy |
| Frontend (later) | React + Vite + PDF.js |
| Testing | pytest, httpx |

## Project Structure

```
tc-analyzer/
├── backend/            FastAPI app, uploads, outputs, tests
│   └── app/            config.py, main.py, models/ routes/ services/ utils/
├── ml/                 preprocessing, clause_segmentation, importance,
│                       coordinate_mapping, annotation, summarization, ...
├── data/               raw/ extracted/ cleaned/ processed/ annotations/ metadata/
├── outputs/            annotated_pdfs/ json/ summaries/
├── evaluation/         phrase / summary / mapping metrics + report
├── configs/            risk_lexicon.json, education_lexicon.json, model_config.json
├── frontend/           React + Vite (built after backend works)
├── scripts/            batch / utility scripts
├── notebooks/          experiments
├── verify_setup.py     environment checker
└── README.md
```

## Setup

### 1. Backend (Python)

```powershell
# from the tc-analyzer folder
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python -m spacy download en_core_web_sm
python verify_setup.py            # confirms all tools + spaCy model
```

### 2. Frontend (Node)

```powershell
cd frontend
npm install
cd ..
```

## Running the App

Open **two terminals**.

**Terminal 1 - backend API** (runs on port 8090):
```powershell
cd tc-analyzer\backend
..\venv\Scripts\python.exe -m uvicorn app.main:app --port 8090
```
> Note: port 8000 is reserved/blocked on some Windows machines
> (winerror 10013). This project uses **8090**; the Vite dev server proxies
> `/api` to it. If you change the port, update `frontend/vite.config.js`.

**Terminal 2 - frontend** (runs on port 5173):
```powershell
cd tc-analyzer\frontend
npm run dev
```

Then open **http://localhost:5173** and upload an educational policy PDF.

Swagger API docs: **http://localhost:8090/docs**

## API Reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET  | `/api/health` | Health check |
| POST | `/api/upload` | Upload a PDF (multipart), returns `document_id` |
| POST | `/api/analyze/{id}` | Analyze synchronously, returns full JSON |
| POST | `/api/analyze-async/{id}` | Start analysis in background |
| GET  | `/api/documents/{id}/progress` | Live analysis stage (for polling) |
| GET  | `/api/documents/{id}` | Metadata / status |
| GET  | `/api/documents/{id}/results` | Full analysis JSON |
| GET  | `/api/documents/{id}/clauses` | Clauses only |
| GET  | `/api/documents/{id}/original` | Original PDF (for the viewer) |
| GET  | `/api/documents/{id}/annotated` | Annotated (highlighted) PDF |

Add `?force=true` to the analyze endpoints to bypass the cached result.

## Batch tools & evaluation (optional)

```powershell
# analyze every PDF in data/raw and write a report
python scripts\batch_analyze.py

# label a ground-truth answer key, then measure P/R/F1 + ROUGE
python scripts\label_ground_truth.py
python scripts\run_evaluation.py

# categorize failures into an error report
python scripts\error_analysis.py
```

## Tests

```powershell
cd backend
..\venv\Scripts\python.exe -m pytest tests -q
```

## Development Phases

The project is built and verified phase by phase (backend/NLP first, frontend last):

- **Phase 0** Environment setup ✅
- **Phase 1** Individual tool testing ✅
- **Phase 2** Directory structure ✅
- **Phase 3** PDF upload API ✅
- **Phase 4** PDF text extraction with coordinates ✅
- **Phase 5** Preprocessing ✅
- **Phase 6** Clause segmentation ✅
- **Phase 7** Important phrase extraction (YAKE + domain lexicon) ✅
- **Phase 8** Importance scoring ✅
- **Phase 9** Coordinate mapping ✅
- **Phase 10** PDF highlighting ✅
- **Phase 11** One-line summarization ✅
- **Phase 12** Complete analysis pipeline ✅
- **Phase 13** JSON output ✅
- **Phase 14** FastAPI backend ✅
- **Phase 15** Database (SQLite) ✅
- **Phase 16** Test with 30-50 PDFs ✅ (48 PDFs, 98.6% mapping, 0 crashes)
- **Phase 17** Evaluation ✅ (real P/R/F1 + ROUGE vs human labels)
- **Phase 18** Error analysis ✅ (see evaluation/ERROR_REPORT.md)
- **Phase 19** React + Vite frontend (upload, progress, results) ✅
- **Phase 20** PDF.js viewer with highlight overlays ✅
- **Phase 21** Interactive highlights + clause detail side panel ✅
- **Phase 22** Clause navigation, page jump, keyboard shortcuts ✅
- **Phase 23** Real-time polling-based progress ✅

Backend test suite: **104 tests passing**.

## Frontend Features

- Drag-and-drop / click PDF upload with live upload + analysis progress
  (real backend stages via polling)
- **List view** - searchable, filterable (HIGH/MEDIUM/LOW) clause cards
- **PDF view** - the original document rendered with color-coded highlight
  overlays; hover for a summary, click to open details
- **Clause detail side panel** - clause number, summary, important phrases
  (with category + risk), original text; prev/next navigation and keyboard
  arrows (Esc / left / right)
- Page-jump bar (pages containing highlights are emphasized)
- Download the annotated PDF
- All data comes from the live API - nothing is hard-coded

## System Rules

- Never modify the original PDF; always work on a copy.
- Always preserve PDF coordinates.
- Highlight meaningful phrases, not every keyword.
- Preserve numbers, dates, percentages and conditions in summaries.
- Do not hallucinate summaries or provide legal advice.
- Load ML models once, not per clause.
- If one clause fails, continue processing the rest.

## Disclaimer

This tool analyzes educational policies to aid understanding. It does **not**
provide legal advice or determine whether a policy is legally valid.
