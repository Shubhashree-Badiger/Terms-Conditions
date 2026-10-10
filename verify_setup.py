"""
Phase 0 Verification Script
Checks all required packages per the updated project prompt are installed.
Run with the venv python interpreter.
"""
import sys
import importlib.util

print("=" * 58)
print(" EDUCATION POLICY & T&C ANALYZER - Phase 0 Verification")
print("=" * 58)
print(f"Python: {sys.version.split()[0]}")
print(f"Executable: {sys.executable}")
print("-" * 58)

# (import_name, friendly_name, purpose)
checks = [
    ("fitz", "PyMuPDF", "PDF extraction + highlighting + coordinates"),
    ("spacy", "spaCy", "sentence segmentation"),
    ("yake", "YAKE", "keyword/phrase extraction"),
    ("sentence_transformers", "sentence-transformers", "semantic scoring (MiniLM)"),
    ("transformers", "transformers", "T5-small summarization"),
    ("torch", "PyTorch", "DL backend"),
    ("fastapi", "FastAPI", "REST backend"),
    ("uvicorn", "uvicorn", "ASGI server"),
    ("pydantic", "pydantic", "data validation"),
    ("sqlalchemy", "SQLAlchemy", "SQLite ORM"),
    ("multipart", "python-multipart", "file upload support"),
    ("pytest", "pytest", "testing"),
    ("httpx", "httpx", "API test client"),
    ("numpy", "NumPy", "numerical arrays"),
    ("sklearn", "scikit-learn", "utility ML"),
]

missing = []
for module, name, purpose in checks:
    spec = importlib.util.find_spec(module)
    if spec is not None:
        try:
            m = importlib.import_module(module)
            ver = getattr(m, "__version__", "n/a")
        except Exception:
            ver = "installed"
        print(f"[OK]  {name:<22} {ver:<12} ({purpose})")
    else:
        missing.append(name)
        print(f"[X]   {name:<22} {'MISSING':<12} ({purpose})")

print("-" * 58)

# spaCy model
if importlib.util.find_spec("spacy") is not None:
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("[OK]  spaCy model 'en_core_web_sm' loaded")
    except Exception:
        missing.append("en_core_web_sm (spaCy model)")
        print("[X]   spaCy model 'en_core_web_sm' NOT found")

print("=" * 58)
if not missing:
    print(" ALL PHASE 0 CHECKS PASSED - ready for Phase 1")
else:
    print(f" MISSING ({len(missing)}): {', '.join(missing)}")
print("=" * 58)
