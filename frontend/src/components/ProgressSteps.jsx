import React from "react";

// Visual step tracker for the analysis lifecycle.
// stage is an index into STEPS; -1 = idle.
const STEPS = [
  "PDF uploaded",
  "Extracting text",
  "Detecting clauses",
  "Finding important phrases",
  "Mapping highlights",
  "Generating summaries",
  "Analysis complete",
];

export default function ProgressSteps({ stage }) {
  if (stage < 0) return null;
  return (
    <ol className="steps">
      {STEPS.map((label, i) => {
        const state =
          i < stage ? "done" : i === stage ? "active" : "todo";
        return (
          <li key={label} className={`step step-${state}`}>
            <span className="step-dot">
              {state === "done" ? "\u2713" : i + 1}
            </span>
            <span className="step-label">{label}</span>
          </li>
        );
      })}
    </ol>
  );
}

export { STEPS };
