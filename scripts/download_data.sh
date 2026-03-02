#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ "${1:-}" != "" ]; then
    if [[ "$1" = /* ]]; then
        DATA_DIR="$1"
    else
        DATA_DIR="$PROJECT_ROOT/$1"
    fi
else
    DATA_DIR="$PROJECT_ROOT/data/external"
fi

export DATA_DIR
mkdir -p "$DATA_DIR"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "Error: neither python3 nor python found in PATH."
    exit 1
fi

echo "=== Downloading im2latex-230k dataset ==="
ZENODO_URL="https://zenodo.org/records/56198/files"

download_file() {
    local dst="$1"
    local src="$2"
    if [ -f "$dst" ] && [ -s "$dst" ]; then
        echo "$(basename "$dst") already exists, skipping."
        return 0
    fi
    echo "Downloading $(basename "$dst") ..."
    curl -L --fail -o "$dst" "$src"
}

download_file "$DATA_DIR/im2latex_formulas.norm.lst" "$ZENODO_URL/im2latex_formulas.lst?download=1"
download_file "$DATA_DIR/im2latex_train_filter.lst" "$ZENODO_URL/im2latex_train.lst?download=1"
download_file "$DATA_DIR/im2latex_validate_filter.lst" "$ZENODO_URL/im2latex_validate.lst?download=1"
download_file "$DATA_DIR/im2latex_test_filter.lst" "$ZENODO_URL/im2latex_test.lst?download=1"

echo "=== Downloading formula images (this may take a while) ==="
if [ ! -f "$DATA_DIR/formula_images.tar.gz" ]; then
    curl -L --fail -o "$DATA_DIR/formula_images.tar.gz" "$ZENODO_URL/formula_images.tar.gz?download=1"
else
    echo "formula_images.tar.gz already exists, skipping."
fi

mkdir -p "$DATA_DIR/images"
if [ -d "$DATA_DIR/images" ] && [ -z "$(ls -A "$DATA_DIR/images" 2>/dev/null)" ]; then
    echo "Extracting images..."
    tar -xzf "$DATA_DIR/formula_images.tar.gz" -C "$DATA_DIR"
fi

if [ -d "$DATA_DIR/formula_images" ]; then
    echo "Normalizing image layout to $DATA_DIR/images ..."
    mv "$DATA_DIR/formula_images"/*.png "$DATA_DIR/images"/ 2>/dev/null || true
    rmdir "$DATA_DIR/formula_images" 2>/dev/null || true
fi

echo "=== Preparing external labels ==="
"$PYTHON_BIN" - <<'PY'
import os
import json
from pathlib import Path

ext = Path(os.environ["DATA_DIR"]).resolve()
images = ext / "images"
formulas_path = ext / "im2latex_formulas.norm.lst"
splits = [
    ("im2latex_train_filter.lst", "train"),
    ("im2latex_validate_filter.lst", "validate"),
    ("im2latex_test_filter.lst", "test"),
]

if not formulas_path.exists():
    raise SystemExit(f"Missing formulas file: {formulas_path}")
if not images.exists():
    raise SystemExit(f"Missing images dir: {images}")

raw = formulas_path.read_bytes()
try:
    formulas = raw.decode("utf-8").splitlines()
except UnicodeDecodeError:
    formulas = raw.decode("latin-1").splitlines()

items = []
missing_images = 0
for split_file, split_name in splits:
    p = ext / split_file
    if not p.exists():
        continue
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        idx = int(parts[0])
        image_name = parts[1] + ".png"
        if idx < 0 or idx >= len(formulas):
            continue
        if not (images / image_name).exists():
            missing_images += 1
            continue
        latex = formulas[idx].strip()
        if not latex:
            continue
        items.append({
            "id": f"im2latex_{parts[1]}",
            "latex": latex,
            "category": "external",
            "split": split_name,
            "image": image_name,
        })

out = ext / "labels.jsonl"
with out.open("w", encoding="utf-8") as f:
    for item in items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"External labels prepared: {len(items)} samples, missing_images={missing_images}")
PY

echo "=== Download complete ==="
echo "Data saved to: $DATA_DIR"
