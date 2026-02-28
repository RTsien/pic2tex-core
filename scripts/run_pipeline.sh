#!/usr/bin/env bash
set -euo pipefail

echo "=== TeXer Full Pipeline ==="

echo "[1/6] Downloading external datasets..."
bash scripts/download_data.sh data/external

echo "[2/6] Generating synthetic formula images..."
python -m data_gen.formula_renderer \
    --output data/synthetic \
    --count 50000

echo "[3/6] Building combined dataset..."
python -m data_gen.build_dataset \
    --synthetic-dir data/synthetic \
    --external-dir data/external \
    --output data/processed

echo "[4/6] Training model (pretrain)..."
python -m model.train \
    --config model/configs/pretrain.yaml

echo "[5/6] Evaluating model..."
python -m model.evaluate \
    --checkpoint checkpoints/best.pt \
    --data data/processed/test

echo "[6/6] Exporting ONNX model..."
python -m model.export_onnx \
    --checkpoint checkpoints/best.pt \
    --output deploy/web/public/model

echo "=== Pipeline complete ==="
