// API client for the Education Policy & T&C Analyzer backend.
// All calls go through /api which Vite proxies to the FastAPI server.

const BASE = "/api";

export async function health() {
  const r = await fetch(`${BASE}/health`);
  if (!r.ok) throw new Error("Backend not reachable");
  return r.json();
}

export async function uploadPdf(file) {
  const form = new FormData();
  form.append("file", file);
  const r = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(err.detail || "Upload failed");
  }
  return r.json();
}

export async function analyze(documentId) {
  const r = await fetch(`${BASE}/analyze/${documentId}`, { method: "POST" });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: "Analysis failed" }));
    throw new Error(err.detail || "Analysis failed");
  }
  return r.json();
}

// Start analysis in the background; returns immediately.
export async function analyzeAsync(documentId) {
  const r = await fetch(`${BASE}/analyze-async/${documentId}`, { method: "POST" });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: "Analysis failed" }));
    throw new Error(err.detail || "Analysis failed");
  }
  return r.json();
}

// Poll the live analysis progress.
export async function getProgress(documentId) {
  const r = await fetch(`${BASE}/documents/${documentId}/progress`);
  if (!r.ok) throw new Error("Progress unavailable");
  return r.json();
}

export async function getResults(documentId) {
  const r = await fetch(`${BASE}/documents/${documentId}/results`);
  if (!r.ok) throw new Error("Results not found");
  return r.json();
}

export function annotatedPdfUrl(documentId) {
  return `${BASE}/documents/${documentId}/annotated`;
}

export function originalPdfUrl(documentId) {
  return `${BASE}/documents/${documentId}/original`;
}
