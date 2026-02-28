#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data/external}"
mkdir -p "$DATA_DIR"

echo "=== Downloading im2latex-230k dataset ==="
ZENODO_URL="https://zenodo.org/records/7738969/files"

for f in im2latex_formulas.norm.lst im2latex_train_filter.lst im2latex_validate_filter.lst im2latex_test_filter.lst; do
    if [ ! -f "$DATA_DIR/$f" ]; then
        echo "Downloading $f ..."
        curl -L -o "$DATA_DIR/$f" "$ZENODO_URL/$f" || echo "Warning: failed to download $f"
    else
        echo "$f already exists, skipping."
    fi
done

echo "=== Downloading formula images (this may take a while) ==="
if [ ! -f "$DATA_DIR/formula_images.tar.gz" ]; then
    curl -L -o "$DATA_DIR/formula_images.tar.gz" "$ZENODO_URL/formula_images.tar.gz" || echo "Warning: failed to download images"
    if [ -f "$DATA_DIR/formula_images.tar.gz" ]; then
        echo "Extracting images..."
        tar -xzf "$DATA_DIR/formula_images.tar.gz" -C "$DATA_DIR"
    fi
else
    echo "formula_images.tar.gz already exists, skipping."
fi

echo "=== Download complete ==="
echo "Data saved to: $DATA_DIR"
