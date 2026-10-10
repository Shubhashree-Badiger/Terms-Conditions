// Configure the PDF.js worker. Vite can resolve the worker file from the
// installed pdfjs-dist package via a URL import.
import * as pdfjsLib from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";

pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;

export { pdfjsLib };
