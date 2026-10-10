import React from "react";

function levelClass(level) {
  const l = (level || "").toUpperCase();
  if (l === "HIGH") return "badge badge-high";
  if (l === "MEDIUM") return "badge badge-medium";
  return "badge badge-low";
}

/**
 * ClausePanel
 * -----------
 * Slide-in side panel showing full details of a selected clause, matching the
 * spec's example layout, with previous / next navigation.
 *
 * Props:
 *   clause   : the clause object (or null -> panel hidden)
 *   onClose  : callback to close the panel
 *   onPrev   : go to previous clause (optional)
 *   onNext   : go to next clause (optional)
 *   position : { index, total } for the "3 of 12" indicator (optional)
 */
import { useEffect } from "react";

export default function ClausePanel({ clause, onClose, onPrev, onNext, position }) {
  const open = !!clause;

  // keyboard navigation: Esc closes, arrows move between clauses
  useEffect(() => {
    if (!open) return;
    function onKey(e) {
      if (e.key === "Escape") onClose && onClose();
      else if (e.key === "ArrowLeft") onPrev && onPrev();
      else if (e.key === "ArrowRight") onNext && onNext();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose, onPrev, onNext]);

  return (
    <>
      <div
        className={`panel-scrim ${open ? "open" : ""}`}
        onClick={onClose}
        aria-hidden={!open}
      />
      <aside className={`clause-panel ${open ? "open" : ""}`} aria-hidden={!open}>
        {clause && (
          <div className="panel-body">
            <div className="panel-head">
              <div>
                <div className="panel-kicker">CLAUSE</div>
                <h2 className="panel-title">
                  {clause.clause_number
                    ? clause.clause_number
                    : clause.clause_id}
                </h2>
                <div className="panel-sub">
                  {clause.clause_id} &middot; page {clause.page_start}
                </div>
              </div>
              <button className="panel-close" onClick={onClose} aria-label="Close">
                &times;
              </button>
            </div>

            <div className="panel-nav">
              <button className="btn btn-ghost" onClick={onPrev} disabled={!onPrev}>
                &larr; Prev
              </button>
              {position && (
                <span className="panel-pos">
                  {position.index + 1} of {position.total}
                </span>
              )}
              <button className="btn btn-ghost" onClick={onNext} disabled={!onNext}>
                Next &rarr;
              </button>
            </div>

            {clause.summary && (
              <section className="panel-section">
                <h4>Summary</h4>
                <p className="panel-summary">{clause.summary}</p>
              </section>
            )}

            <section className="panel-section">
              <h4>Important phrases</h4>
              {(clause.important_phrases || []).length === 0 && (
                <p className="panel-muted">None flagged.</p>
              )}
              <ul className="panel-phrases">
                {(clause.important_phrases || []).map((p, i) => (
                  <li key={i}>
                    <span className={levelClass(p.level)}>{p.level}</span>
                    <span className="pp-text">{p.text}</span>
                    {p.category && <span className="pp-cat">{p.category}</span>}
                    {p.risk_type && p.risk_level && p.risk_level !== "NONE" && (
                      <span className="pp-risk">
                        risk: {p.risk_type} ({p.risk_level})
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            </section>

            <section className="panel-section">
              <h4>Original clause</h4>
              <p className="panel-original">{clause.clause_text}</p>
            </section>

            <p className="panel-disclaimer">
              This is an aid to understanding, not legal advice.
            </p>
          </div>
        )}
      </aside>
    </>
  );
}
