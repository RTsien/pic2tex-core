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

echo "[3/6] Building and preprocessing dataset..."
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
        TRAIN_CONFIG="model/configs/swin_small_cuda_train.yaml"
        BEST_CHECKPOINT="checkpoints/swin_small_cuda/best.pt"
        ;;
    mps)
        TRAIN_CONFIG="model/configs/swin_small_mps_train.yaml"
        BEST_CHECKPOINT="checkpoints/swin_small_mps/best.pt"
        ;;
    *)
        TRAIN_CONFIG="model/configs/cpu_train.yaml"
        BEST_CHECKPOINT="checkpoints/best.pt"
        ;;
esac

read -r PREPROCESS_HEIGHT PREPROCESS_WIDTH <<< "$("$PYTHON_BIN" - "$TRAIN_CONFIG" <<'PY'
import sys
import yaml

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

data_cfg = cfg.get("data", {}) or {}
height = data_cfg.get("image_height", data_cfg.get("image_size"))
width = data_cfg.get("image_width", data_cfg.get("image_size"))
if height is None or width is None:
    raise SystemExit(f"Missing image size in config: {config_path}")

print(f"{int(height)} {int(width)}")
PY
)"
"$PYTHON_BIN" scripts/preprocess_images.py \
    --dataset-root data/processed \
    --splits train,val,test \
    --image-height "$PREPROCESS_HEIGHT" \
    --image-width "$PREPROCESS_WIDTH" \
    --keep-aspect-ratio \
    --workers 12 \
    --autocontrast-cutoff 0 \
    --unsharp-percent 0

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
