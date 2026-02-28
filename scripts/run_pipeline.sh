#!/usr/bin/env bash
set -euo pipefail

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "Error: neither python3 nor python found in PATH."
    exit 1
fi

echo "=== TeXer Full Pipeline ==="

echo "[1/6] Downloading external datasets..."
bash scripts/download_data.sh data/external

echo "[2/6] Generating synthetic formula images..."
"$PYTHON_BIN" -m data_gen.formula_renderer \
    --output data/synthetic \
    --count 50000

echo "[3/6] Building combined dataset..."
"$PYTHON_BIN" -m data_gen.build_dataset \
    --synthetic-dir data/synthetic \
    --external-dir data/external \
    --output data/processed

# Auto-select training config by available accelerator (CUDA > MPS > CPU).
DEVICE_TYPE="$("$PYTHON_BIN" - <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        print("cuda")
    elif torch.backends.mps.is_available():
        print("mps")
    else:
        print("cpu")
except Exception:
    print("cpu")
PY
)"

case "$DEVICE_TYPE" in
    cuda)
        TRAIN_CONFIG="model/configs/pretrain.yaml"
        BEST_CHECKPOINT="checkpoints/pretrain/best.pt"
        ;;
    mps)
        TRAIN_CONFIG="model/configs/mps_train.yaml"
        BEST_CHECKPOINT="checkpoints/best.pt"
        ;;
    *)
        TRAIN_CONFIG="model/configs/cpu_train.yaml"
        BEST_CHECKPOINT="checkpoints/best.pt"
        ;;
esac

echo "[4/6] Training model (device: $DEVICE_TYPE, config: $TRAIN_CONFIG)..."
"$PYTHON_BIN" -m model.train \
    --config "$TRAIN_CONFIG" \
    --device "$DEVICE_TYPE"

echo "[5/6] Evaluating model..."
"$PYTHON_BIN" -m model.evaluate \
    --checkpoint "$BEST_CHECKPOINT" \
    --data data/processed/test

echo "[6/6] Exporting ONNX model..."
"$PYTHON_BIN" -m model.export_onnx \
    --checkpoint "$BEST_CHECKPOINT" \
    --output deploy/web/public/model

echo "=== Pipeline complete ==="
