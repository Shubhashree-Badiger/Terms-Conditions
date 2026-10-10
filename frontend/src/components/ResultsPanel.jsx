import React, { useState, useMemo } from "react";
import { annotatedPdfUrl, originalPdfUrl } from "../api";
import PdfViewer from "./PdfViewer.jsx";
import ClausePanel from "./ClausePanel.jsx";

function levelClass(level) {
  const l = (level || "").toUpperCase();
  if (l === "HIGH") return "badge badge-high";
  if (l === "MEDIUM") return "badge badge-medium";
  return "badge badge-low";
}

export default function ResultsPanel({ result }) {
  const [filter, setFilter] = useState("ALL");
  const [query, setQuery] = useState("");
  const [view, setView] = useState("list"); // "list" | "pdf"
  const [activeClauseId, setActiveClauseId] = useState(null);

  // clause lookup by id (for the side panel)
  const clauseById = useMemo(() => {
    const m = {};
    for (const c of result?.clauses || []) m[c.clause_id] = c;
    return m;
  }, [result]);

  // flat highlight list for the PDF viewer
  const highlights = useMemo(() => {
    if (!result || !result.clauses) return [];
    const out = [];
    for (const c of result.clauses) {
      for (const p of c.important_phrases || []) {
        if (!p.locations || !p.locations.length) continue;
        out.push({
          clauseId: c.clause_id,
          phrase: p.text,
          level: p.level,
          summary: c.summary,
          locations: p.locations,
        });
      }
    }
    return out;
  }, [result]);

  if (!result) return null;

  if (result.status === "scanned") {
    return (
      <div className="card notice">
        <h3>Scanned document</h3>
        <p>{result.message}</p>
      </div>
    );
  }

  const clauses = result.clauses || [];
  const withPhrases = clauses.filter((c) => (c.important_phrases || []).length);

  const visible = withPhrases.filter((c) => {
    const matchesQuery =
      !query ||
      c.clause_text.toLowerCase().includes(query.toLowerCase()) ||
      (c.summary || "").toLowerCase().includes(query.toLowerCase());
    const matchesFilter =
      filter === "ALL" ||
      (c.important_phrases || []).some(
        (p) => (p.level || "").toUpperCase() === filter
      );
    return matchesQuery && matchesFilter;
  });

  const activeClause = activeClauseId ? clauseById[activeClauseId] : null;

  // Ordered list of clauses that can be navigated (those with phrases).
  const navIds = withPhrases.map((c) => c.clause_id);
  const activeIndex = activeClauseId ? navIds.indexOf(activeClauseId) : -1;
  const gotoPrev =
    activeIndex > 0 ? () => setActiveClauseId(navIds[activeIndex - 1]) : null;
  const gotoNext =
    activeIndex >= 0 && activeIndex < navIds.length - 1
      ? () => setActiveClauseId(navIds[activeIndex + 1])
      : null;
  const position =
    activeIndex >= 0 ? { index: activeIndex, total: navIds.length } : null;

  return (
    <div className="results">
      <div className="card stats">
        <div>
          <strong>{result.total_clauses}</strong>
          <span>clauses</span>
        </div>
        <div>
          <strong>{withPhrases.length}</strong>
          <span>with highlights</span>
        </div>
        <div>
          <strong>{result.total_pages}</strong>
          <span>pages</span>
        </div>
        <div>
          <strong>{result.stats?.mapped_phrases ?? 0}</strong>
          <span>mapped phrases</span>
        </div>
        <div className="view-toggle">
          <button
            className={`chip ${view === "list" ? "chip-active" : ""}`}
            onClick={() => setView("list")}
          >
            List view
          </button>
          <button
            className={`chip ${view === "pdf" ? "chip-active" : ""}`}
            onClick={() => setView("pdf")}
          >
            PDF view
          </button>
        </div>
        <a
          className="btn btn-secondary"
          href={annotatedPdfUrl(result.document_id)}
          target="_blank"
          rel="noreferrer"
        >
          Download annotated PDF
        </a>
      </div>

      {view === "list" && (
        <>
          <div className="toolbar">
            <input
              className="search"
              placeholder="Search clauses / summaries..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <div className="filters">
              {["ALL", "HIGH", "MEDIUM", "LOW"].map((f) => (
                <button
                  key={f}
                  className={`chip ${filter === f ? "chip-active" : ""}`}
                  onClick={() => setFilter(f)}
                >
                  {f}
                </button>
              ))}
            </div>
          </div>

          <div className="clause-list">
            {visible.map((c) => (
              <div
                className={`card clause clause-clickable ${
                  activeClauseId === c.clause_id ? "clause-active" : ""
                }`}
                key={c.clause_id}
                onClick={() => setActiveClauseId(c.clause_id)}
              >
                <div className="clause-head">
                  <span className="clause-id">
                    {c.clause_id}
                    {c.clause_number ? ` \u00b7 ${c.clause_number}` : ""}
                  </span>
                  <span className="clause-page">page {c.page_start}</span>
                </div>

                {c.summary && (
                  <p className="clause-summary">
                    <strong>Summary:</strong> {c.summary}
                  </p>
                )}

                <div className="phrases">
                  {(c.important_phrases || []).map((p, idx) => (
                    <span key={idx} className="phrase-row">
                      <span className={levelClass(p.level)}>{p.level}</span>
                      <span className="phrase-text">{p.text}</span>
                      {p.category && (
                        <span className="phrase-cat">{p.category}</span>
                      )}
                      <span className="phrase-score">{p.score}</span>
                    </span>
                  ))}
                </div>
                <div className="clause-cta">Click for details &rarr;</div>
              </div>
            ))}
            {visible.length === 0 && (
              <div className="card empty">No clauses match your filter.</div>
            )}
          </div>
        </>
      )}

      {view === "pdf" && (
        <div className="pdf-wrap card">
          <p className="pdf-hint">
            Highlights are drawn over the original document. Hover to see the
            clause summary; click a highlight to open its full details.
          </p>
          <PdfViewer
            fileUrl={originalPdfUrl(result.document_id)}
            highlights={highlights}
            activeClauseId={activeClauseId}
            onHighlightClick={(h) => setActiveClauseId(h.clauseId)}
          />
        </div>
      )}

      <ClausePanel
        clause={activeClause}
        onClose={() => setActiveClauseId(null)}
        onPrev={gotoPrev}
        onNext={gotoNext}
        position={position}
      />
    </div>
  );
}
