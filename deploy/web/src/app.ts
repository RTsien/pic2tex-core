/**
 * TeXer Web Application - main UI logic.
 *
 * Features:
 * - Drag & drop / paste / upload formula images
 * - Real-time LaTeX recognition via ONNX Runtime Web
 * - Live KaTeX preview of recognized formula
 * - One-click copy to clipboard
 */

import { TeXerEngine } from "./inference";
import katex from "katex";

const engine = new TeXerEngine();

const $ = (sel: string) => document.querySelector(sel)!;

const dropZone = $("#drop-zone") as HTMLDivElement;
const fileInput = $("#file-input") as HTMLInputElement;
const previewImg = $("#preview-img") as HTMLImageElement;
const resultArea = $("#result-area") as HTMLDivElement;
const latexOutput = $("#latex-output") as HTMLTextAreaElement;
const katexPreview = $("#katex-preview") as HTMLDivElement;
const copyBtn = $("#copy-btn") as HTMLButtonElement;
const statusText = $("#status") as HTMLSpanElement;
const loadingOverlay = $("#loading-overlay") as HTMLDivElement;
const modelSelect = $("#model-select") as HTMLSelectElement;

type ModelKey = "cnn" | "swin" | "swin_small";

const MODEL_CONFIG: Record<ModelKey, { label: string; dir: string }> = {
  cnn: { label: "CNN", dir: "./model_cnn" },
  swin: { label: "Swin", dir: "./model" },
  swin_small: { label: "Swin Small", dir: "./model_swin_small" },
};

let activeModel: ModelKey = "cnn";

function setStatus(text: string, type: "info" | "success" | "error" = "info") {
  statusText.textContent = text;
  statusText.className = `status-${type}`;
}

function showLoading(show: boolean) {
  loadingOverlay.style.display = show ? "flex" : "none";
}

function getModelFromQuery(): ModelKey {
  const value = new URLSearchParams(window.location.search).get("model");
  if (value === "cnn" || value === "swin" || value === "swin_small") return value;
  return "cnn";
}

function updateModelQuery(model: ModelKey) {
  const url = new URL(window.location.href);
  url.searchParams.set("model", model);
  window.history.replaceState({}, "", url.toString());
}

async function loadModel(model: ModelKey, syncQuery: boolean): Promise<void> {
  const cfg = MODEL_CONFIG[model];
  activeModel = model;
  modelSelect.value = model;
  modelSelect.disabled = true;
  setStatus(`Loading ${cfg.label} model...`, "info");
  showLoading(true);

  try {
    await engine.load(cfg.dir);
    if (syncQuery) updateModelQuery(model);
    setStatus(`Ready (${cfg.label}) - drop or paste a formula image`, "success");
  } catch (err) {
    setStatus(
      `Model not found. Place ONNX files in public/${cfg.dir.replace("./", "")}/ directory.`,
      "error"
    );
    console.error("Failed to load model:", err);
  } finally {
    modelSelect.disabled = false;
    showLoading(false);
  }
}

function renderKatex(latex: string) {
  try {
    katex.render(latex, katexPreview, {
      throwOnError: false,
      displayMode: true,
      trust: true,
    });
  } catch {
    katexPreview.textContent = "(Preview unavailable)";
  }
}

async function processImage(file: File | Blob) {
  const img = new Image();
  const url = URL.createObjectURL(file);

  img.onload = async () => {
    previewImg.src = url;
    previewImg.style.display = "block";
    resultArea.style.display = "block";

    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    setStatus("Recognizing formula...", "info");
    showLoading(true);

    try {
      const startTime = performance.now();
      const latex = await engine.recognize(imageData);
      const elapsed = (performance.now() - startTime).toFixed(0);

      latexOutput.value = latex;
      renderKatex(latex);
      setStatus(`Done in ${elapsed}ms (${MODEL_CONFIG[activeModel].label})`, "success");
    } catch (err) {
      setStatus(`Error: ${err}`, "error");
      latexOutput.value = "";
      katexPreview.textContent = "";
    } finally {
      showLoading(false);
    }
  };

  img.src = url;
}

// --- Event handlers ---

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const files = (e as DragEvent).dataTransfer?.files;
  if (files && files.length > 0) {
    processImage(files[0]);
  }
});

dropZone.addEventListener("click", () => {
  fileInput.click();
});

modelSelect.addEventListener("change", async () => {
  const next = modelSelect.value as ModelKey;
  await loadModel(next, true);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files.length > 0) {
    processImage(fileInput.files[0]);
  }
});

document.addEventListener("paste", (e) => {
  const items = (e as ClipboardEvent).clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith("image/")) {
      const blob = item.getAsFile();
      if (blob) processImage(blob);
      break;
    }
  }
});

copyBtn.addEventListener("click", async () => {
  const text = latexOutput.value;
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    const originalText = copyBtn.textContent;
    copyBtn.textContent = "Copied!";
    setTimeout(() => {
      copyBtn.textContent = originalText;
    }, 1500);
  } catch {
    latexOutput.select();
    document.execCommand("copy");
  }
});

latexOutput.addEventListener("input", () => {
  renderKatex(latexOutput.value);
});

// --- Init ---

async function autoTest(imageUrl: string) {
  const resp = await fetch(imageUrl);
  const blob = await resp.blob();
  const img = new Image();
  img.onload = async () => {
    previewImg.src = imageUrl;
    previewImg.style.display = "block";
    resultArea.style.display = "block";

    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    setStatus("Recognizing formula...", "info");
    showLoading(true);
    try {
      const startTime = performance.now();
      const latex = await engine.recognize(imageData);
      const elapsed = (performance.now() - startTime).toFixed(0);
      latexOutput.value = latex;
      renderKatex(latex);
      setStatus(`Done in ${elapsed}ms`, "success");
    } catch (err) {
      setStatus(`Error: ${err}`, "error");
    } finally {
      showLoading(false);
    }
  };
  img.src = URL.createObjectURL(blob);
}

async function init() {
  const initialModel = getModelFromQuery();
  await loadModel(initialModel, true);

  const params = new URLSearchParams(window.location.search);
  const testImg = params.get("test");
  if (testImg && engine.isLoaded) {
    await autoTest(testImg);
  }
}

init();
