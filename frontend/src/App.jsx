import React, { useState, useRef, useEffect } from "react";
import { health, uploadPdf, analyzeAsync, getProgress, getResults } from "./api";
import ProgressSteps from "./components/ProgressSteps.jsx";
import ResultsPanel from "./components/ResultsPanel.jsx";

export default function App() {
  const [file, setFile] = useState(null);
  const [stage, setStage] = useState(-1); // -1 idle; 0..6 progress
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [backendOk, setBackendOk] = useState(null);
  const fileInput = useRef(null);

  useEffect(() => {
    health()
      .then(() => setBackendOk(true))
      .catch(() => setBackendOk(false));
  }, []);

  function pickFile(e) {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setResult(null);
      setError("");
      setStage(-1);
    }
  }

  function onDrop(e) {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f && f.type === "application/pdf") {
      setFile(f);
      setResult(null);
      setError("");
      setStage(-1);
    }
  }

  async function run() {
    if (!file) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      // Step 0: upload
      setStage(0);
      const up = await uploadPdf(file);

      // Start background analysis, then poll the REAL progress endpoint so the
      // UI reflects the backend's actual stage (not a simulated timer).
      await analyzeAsync(up.document_id);

      const finalProgress = await pollUntilDone(up.document_id);

      if (finalProgress.error) {
        throw new Error(finalProgress.message || "Analysis failed");
      }

      // fetch the completed results, with a short retry in case the JSON is
      // still being written the instant progress flips to done.
      const res = await fetchResultsWithRetry(up.document_id, 6);
      if (res.status === "scanned") {
        setStage(-1);
        setResult(res);
      } else {
        setStage(6);
        setResult(res);
      }
    } catch (err) {
      setError(err.message || "Something went wrong");
      setStage(-1);
    } finally {
      setBusy(false);
    }
  }

  // Fetch results, retrying briefly if the file isn't written yet.
  async function fetchResultsWithRetry(documentId, attempts = 6) {
    let lastErr;
    for (let i = 0; i < attempts; i++) {
      try {
        return await getResults(documentId);
      } catch (e) {
        lastErr = e;
        await new Promise((r) => setTimeout(r, 800));
      }
    }
    throw lastErr || new Error("Results not found");
  }

  // Poll the backend progress every ~700ms until done or error.
  // Transient poll failures (e.g. progress row not created yet) are tolerated
  // for a few attempts before giving up.
  function pollUntilDone(documentId) {
    return new Promise((resolve, reject) => {
      let misses = 0;
      const tick = async () => {
        try {
          const p = await getProgress(documentId);
          misses = 0;
          if (p.stage >= 0) setStage(p.stage);
          if (p.done) return resolve(p);
          if (p.error) return resolve(p);
          setTimeout(tick, 700);
        } catch (e) {
          misses += 1;
          if (misses > 8) return reject(e);
          setTimeout(tick, 700);
        }
      };
      tick();
    });
  }

  return (
    <div className="app">
      <header className="header">
        <h1>AI-Based Education Policy &amp; T&amp;C Analyzer</h1>
        <p className="subtitle">
          Upload an institutional policy PDF to see the important clauses
          highlighted, with a simple one-line summary for each.
        </p>
        {backendOk === false && (
          <div className="banner banner-warn">
            Backend not reachable. Start it with:{" "}
            <code>uvicorn app.main:app --port 8090</code> (from the backend
            folder).
          </div>
        )}
      </header>

      <section
        className="uploader"
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        onClick={() => fileInput.current?.click()}
      >
        <input
          ref={fileInput}
          type="file"
          accept="application/pdf"
          onChange={pickFile}
          hidden
        />
        <div className="uploader-inner">
          <div className="uploader-icon">PDF</div>
          {file ? (
            <p className="file-name">{file.name}</p>
          ) : (
            <p>Click to choose a PDF, or drag &amp; drop it here</p>
          )}
        </div>
      </section>

      <div className="actions">
        <button
          className="btn btn-primary"
          disabled={!file || busy}
          onClick={run}
        >
          {busy ? "Analyzing..." : "Analyze document"}
        </button>
        {file && !busy && (
          <button
            className="btn btn-ghost"
            onClick={() => {
              setFile(null);
              setResult(null);
              setStage(-1);
              setError("");
            }}
          >
            Clear
          </button>
        )}
      </div>

      {error && <div className="banner banner-error">{error}</div>}

      <ProgressSteps stage={stage} />

      <ResultsPanel result={result} />

      <footer className="footer">
        This tool aids understanding of educational policies. It does not
        provide legal advice.
      </footer>
    </div>
  );
}
