# TeXer

公式截图转 LaTeX 轻量级识别工具。

## 概述

TeXer 是一个端到端的公式图片识别系统，能将数学公式、化学方程式等截图转换为 LaTeX 代码。核心模型仅约 20M 参数，可在浏览器和移动设备上直接运行。

### 三大模块

1. **数据生成** (`data_gen/`) — 使用 Qwen3-VL-4B-Instruct 作为教师模型生成高质量训练数据
2. **模型训练** (`model/`) — 训练 Swin Encoder + Transformer Decoder 轻量级模型
3. **部署** (`deploy/`) — ONNX 格式导出，支持浏览器 (ONNX Runtime Web) 和移动端

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

### 数据生成

```bash
# 1. 下载公开数据集
bash scripts/download_data.sh

# 2. 生成合成公式图片
python -m data_gen.formula_renderer --output data/train --count 50000

# 3. 使用 Qwen3-VL 标注真实图片
python -m data_gen.qwen_annotator --images path/to/real_images --output data/train

# 4. 构建最终数据集
python -m data_gen.build_dataset --data-dir data --output data/processed
```

### 模型训练

训练脚本自动检测最佳设备（CUDA > MPS > CPU），也可通过 `--device` 手动指定。

```bash
# Mac M 芯片加速训练（自动检测 MPS，batch_size=32）
python -m model.train --config model/configs/mps_train.yaml

# CUDA GPU 训练（支持 FP16 混合精度，batch_size=64）
python -m model.train --config model/configs/pretrain.yaml

# CPU 训练（batch_size=16，适合无 GPU 环境）
python -m model.train --config model/configs/cpu_train.yaml

# 从 checkpoint 恢复训练（config 需与原训练一致）
python -m model.train --config model/configs/mps_train.yaml --resume checkpoints/best.pt

# 手动指定设备（覆盖自动检测）
python -m model.train --config model/configs/cpu_train.yaml --device cpu

# 评估
python -m model.evaluate --checkpoint checkpoints/best.pt --data data/test
```

> 每个配置文件的 `batch_size` 和 `num_workers` 已针对对应设备优化，请选择匹配的配置。

#### 设备加速对比

| 设备 | 预计速度 | 说明 |
|------|---------|------|
| NVIDIA GPU (CUDA) | ~10-20x | 支持 FP16 混合精度 |
| Apple M1/M2/M3/M4 (MPS) | ~3-5x | Metal Performance Shaders 加速 |
| CPU | 1x (baseline) | 适合小规模实验 |

### 导出与部署

```bash
# 导出 ONNX
python -m model.export_onnx --checkpoint checkpoints/best.pt --output deploy/web/public/model

# 启动 Web 应用
cd deploy/web && npm install && npm run dev
```

## 项目结构

```
texer/
├── data_gen/                # 数据生成模块
│   ├── formula_collector.py # 公式收集（arXiv/模板生成）
│   ├── formula_renderer.py  # 公式渲染为图片
│   ├── qwen_annotator.py    # Qwen3-VL 标注与验证
│   ├── augmentation.py      # 数据增强
│   └── build_dataset.py     # 构建最终数据集
├── model/                   # 模型训练模块
│   ├── config.py            # 模型/训练配置
│   ├── tokenizer.py         # LaTeX tokenizer
│   ├── architecture.py      # 模型架构定义
│   ├── dataset.py           # PyTorch Dataset
│   ├── train.py             # 训练主脚本
│   ├── evaluate.py          # 评估脚本
│   └── export_onnx.py       # ONNX 导出
├── deploy/                  # 部署模块
│   ├── web/                 # 浏览器端应用
│   └── mobile/              # 移动端（后续扩展）
├── scripts/                 # 实用脚本
├── data/                    # 数据目录
└── requirements.txt
```

## 技术栈

| 模块 | 技术 |
|------|------|
| 数据生成 | Python, matplotlib, LaTeX, Qwen3-VL-4B-Instruct (vLLM) |
| 模型训练 | PyTorch, HuggingFace Transformers, Swin Transformer |
| 模型导出 | ONNX, ONNX Runtime, INT8 量化 |
| 浏览器部署 | TypeScript, ONNX Runtime Web, React |
| 移动端部署 | ONNX Runtime Mobile / CoreML |

## License

MIT
