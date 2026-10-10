import React, { useEffect, useRef, useState } from "react";
import { pdfjsLib } from "../pdfWorker.js";

/**
 * PdfViewer
 * ---------
 * Renders a PDF (from a URL) using PDF.js and draws highlight overlays on top
 * of each page from the backend's phrase locations.
 *
 * Props:
 *   fileUrl   : string  - URL to fetch the PDF bytes from (the ORIGINAL pdf).
 *   highlights: array    - flat list of highlight items:
 *       { page, bbox:[x0,y0,x1,y1], level, phrase, clauseId, summary }
 *   activeClauseId : string | null - clause to emphasize / scroll to.
 *   onHighlightClick(item) : callback when a highlight is clicked.
 *   scale     : number  - render scale (default 1.3).
 *
 * Coordinate conversion:
 *   The backend bbox is in PDF user-space (origin bottom-left, points). PDF.js
 *   viewport.convertToViewportRectangle() maps a [x0,y0,x1,y1] PDF rect to
 *   viewport (screen) space at the current scale, which we then use to place an
 *   absolutely-positioned <div> over the canvas.
 */
export default function PdfViewer({
  fileUrl,
  highlights = [],
  activeClauseId = null,
  onHighlightClick,
  scale = 1.3,
}) {
  const [doc, setDoc] = useState(null);
  const [numPages, setNumPages] = useState(0);
  const [loadError, setLoadError] = useState("");

  // group highlights by page number (1-based)
  const byPage = React.useMemo(() => {
    const m = {};
    for (const h of highlights) {
      for (const loc of h.locations || []) {
        (m[loc.page] ||= []).push({ ...h, bbox: loc.bbox });
      }
    }
    return m;
  }, [highlights]);

  useEffect(() => {
    let cancelled = false;
    setLoadError("");
    setDoc(null);
    if (!fileUrl) return;

    (async () => {
      try {
        const task = pdfjsLib.getDocument(fileUrl);
        const pdf = await task.promise;
        if (cancelled) return;
        setDoc(pdf);
        setNumPages(pdf.numPages);
      } catch (e) {
        if (!cancelled) setLoadError(e?.message || "Failed to load PDF");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [fileUrl]);

  function scrollToPage(n) {
    const el = document.getElementById(`pdf-page-${n}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  if (loadError) {
    return <div className="pdf-error">Could not render PDF: {loadError}</div>;
  }
  if (!doc) {
    return <div className="pdf-loading">Loading PDF...</div>;
  }

  // pages that actually contain highlights (for quick jump)
  const highlightedPages = Object.keys(byPage)
    .map(Number)
    .sort((a, b) => a - b);

  return (
    <div>
      <div className="pdf-nav">
        <span className="pdf-nav-label">Jump to page:</span>
        {Array.from({ length: numPages }, (_, i) => i + 1).map((n) => (
          <button
            key={n}
            className={`pagebtn ${
              highlightedPages.includes(n) ? "pagebtn-hl" : ""
            }`}
            onClick={() => scrollToPage(n)}
            title={highlightedPages.includes(n) ? "has highlights" : ""}
          >
            {n}
          </button>
        ))}
      </div>
      <div className="pdf-viewer">
        {Array.from({ length: numPages }, (_, i) => i + 1).map((pageNum) => (
          <PdfPage
            key={pageNum}
            doc={doc}
            pageNum={pageNum}
            scale={scale}
            highlights={byPage[pageNum] || []}
            activeClauseId={activeClauseId}
            onHighlightClick={onHighlightClick}
          />
        ))}
      </div>
    </div>
  );
}

function levelColor(level) {
  const l = (level || "").toUpperCase();
  if (l === "HIGH") return "rgba(229,72,77,0.30)";
  if (l === "MEDIUM") return "rgba(242,160,61,0.32)";
  return "rgba(232,201,74,0.38)";
}
function levelBorder(level) {
  const l = (level || "").toUpperCase();
  if (l === "HIGH") return "#e5484d";
  if (l === "MEDIUM") return "#f2a03d";
  return "#c9b41f";
}

function PdfPage({ doc, pageNum, scale, highlights, activeClauseId, onHighlightClick }) {
  const canvasRef = useRef(null);
  const [dims, setDims] = useState({ width: 0, height: 0 });
  const [boxes, setBoxes] = useState([]);
  const renderTaskRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      const page = await doc.getPage(pageNum);
      if (cancelled) return;
      const viewport = page.getViewport({ scale });
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      setDims({ width: viewport.width, height: viewport.height });

      // cancel any in-flight render before starting a new one
      if (renderTaskRef.current) {
        try { renderTaskRef.current.cancel(); } catch { /* ignore */ }
      }
      const task = page.render({ canvasContext: ctx, viewport });
      renderTaskRef.current = task;
      try {
        await task.promise;
      } catch {
        /* render cancelled */
      }

      // Convert each highlight's PDF bbox -> viewport (screen) rect.
      //
      // IMPORTANT: the backend bbox comes from PyMuPDF's page.search_for(),
      // which returns rectangles in a TOP-LEFT origin coordinate system (y
      // grows downward) measured in unscaled PDF points - the SAME orientation
      // as the rendered canvas. We therefore must NOT use
      // viewport.convertToViewportRectangle(), which expects PDF user-space
      // (BOTTOM-LEFT origin) and flips the y-axis internally. Passing already
      // top-left coordinates through it flips y a second time and drops every
      // highlight at a mirrored/incorrect vertical position (the bug the user
      // reported). Instead we scale the top-left points directly.
      const converted = highlights.map((h) => {
        const [x0, y0, x1, y1] = h.bbox;
        const left = Math.min(x0, x1) * scale;
        const top = Math.min(y0, y1) * scale;
        const width = Math.abs(x1 - x0) * scale;
        const height = Math.abs(y1 - y0) * scale;
        return { ...h, rect: { left, top, width, height } };
      });
      if (!cancelled) setBoxes(converted);
    })();

    return () => {
      cancelled = true;
      if (renderTaskRef.current) {
        try { renderTaskRef.current.cancel(); } catch { /* ignore */ }
      }
    };
  }, [doc, pageNum, scale, highlights]);

  // When this page contains the active clause, scroll it into view.
  useEffect(() => {
    if (!activeClauseId || !boxes.length) return;
    if (boxes.some((b) => b.clauseId === activeClauseId)) {
      const el = document.getElementById(`pdf-page-${pageNum}`);
      if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [activeClauseId, boxes, pageNum]);

  return (
    <div
      id={`pdf-page-${pageNum}`}
      className="pdf-page"
      style={{ width: dims.width, height: dims.height }}
    >
      <canvas ref={canvasRef} className="pdf-canvas" />
      <div className="pdf-overlay">
        {boxes.map((b, i) => {
          const active = activeClauseId && b.clauseId === activeClauseId;
          return (
            <div
              key={i}
              className={`hl ${active ? "hl-active" : ""}`}
              title={b.summary || b.phrase}
              onClick={() => onHighlightClick && onHighlightClick(b)}
              style={{
                left: b.rect.left,
                top: b.rect.top,
                width: b.rect.width,
                height: b.rect.height,
                background: levelColor(b.level),
                outline: active ? `2px solid ${levelBorder(b.level)}` : "none",
              }}
            />
          );
        })}
      </div>
      <div className="pdf-page-num">Page {pageNum}</div>
    </div>
  );
}
