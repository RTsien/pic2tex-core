# 长公式优化工作记录（项目内）

更新时间：2026-03-01

## 1. 背景与目标

- 现象：CNN 在较长公式场景下更容易出现漏字、错位、提前结束等问题。
- 目标：在不大改主干结构的前提下，优先提升长公式识别稳定性，并保留可回退能力。
- 范围：训练数据预处理、采样策略、解码策略、对比实验流程（CNN vs Swin Small）。

## 2. 已落地的优化（代码与配置）

### 2.1 图像预处理：保长宽比 + 白底 padding（核心）

- 文件：`model/dataset.py`
- 变更：
  - 新增 `ResizeWithPad` 可序列化变换类。
  - 替换原先强制 `Resize((image_size, image_size))` 的做法。
  - 训练增强里 `RandomAffine` 增加 `fill=255`，避免仿射后黑边噪声。
- 目的：降低长公式横向压缩带来的细节损失（上下标、括号、分式结构）。

### 2.2 长公式重采样（训练侧）

- 文件：
  - `model/config.py`
  - `model/dataset.py`
  - `model/train.py`
  - `model/configs/cnn_mps_train.yaml`
  - `model/configs/swin_small_mps_long.yaml`
- 变更：
  - 新增配置项：
    - `train.long_formula_min_tokens`
    - `train.long_formula_oversample_factor`
  - `create_dataloader(...)` 在训练阶段可启用 `WeightedRandomSampler`。
  - 训练日志打印当前长公式重采样策略，便于追踪实验条件。
- 当前策略：
  - CNN：`min_tokens=120`、`factor=1.8`
  - Swin Small 公平微调：`min_tokens=40`、`factor=1.8`

### 2.3 评估/推理：可选 Beam Search（默认兼容贪心）

- 文件：
  - `model/architecture.py`
  - `model/evaluate.py`
- 变更：
  - `TeXerModel.generate(...)` 新增参数：
    - `beam_size`
    - `length_penalty`
  - `evaluate.py` 增加 CLI 参数：
    - `--beam-size`
    - `--length-penalty`
- 说明：默认 `beam_size=1` 与旧逻辑一致，不影响既有流程。

### 2.4 新增公平对比训练配置（Swin Small）

- 文件：`model/configs/swin_small_mps_long.yaml`
- 用途：在 Swin Small 上对齐 CNN 的长公式策略，减少 A/B 对比偏差。
- 要点：
  - 使用 `keep_aspect_ratio: true`
  - 使用长样本重采样
  - 独立 checkpoint 目录：`checkpoints/swin_small_mps_long`

## 3. 对比实验过程记录

### 3.1 数据分布核查结论（重要）

- 核查后发现当前数据集并不存在真正“超长”样本：
  - train 最大长度约 70 token
  - val 最大长度约 66 token
  - test 最大长度约 64 token
- 结论：原先 `min_tokens=120` 对当前数据不敏感，后续对比中将阈值下调到 40/50/60 做分析更合理。

### 3.2 已完成的一轮长样本子集对比（贪心解码）

- 条件：`beam_size=1`，在 `val+test` 的长样本子集比较 CNN 与 Swin Small。
- 结果（示例）：
  - `len>=40, n=47`：CNN 平均相似度高于 Swin Small
  - `len>=50, n=15`：CNN 平均相似度高于 Swin Small
  - `len>=60, n=6`：CNN 平均相似度高于 Swin Small
- 备注：样本量偏小（尤其 >=60），统计波动较大，需持续复核。

## 4. 踩坑与修复记录

### 4.1 多进程 DataLoader 启动失败（已修复）

- 现象：
  - 报错：`AttributeError: Can't pickle local object '...<lambda>'`
- 原因：
  - 在 `transforms.Lambda(lambda ...)` 中使用局部 lambda，`num_workers>0` 时无法 pickle。
- 修复：
  - 用顶层可序列化类 `ResizeWithPad` 替代 lambda。
- 状态：已验证可正常多进程加载并恢复训练。

### 4.2 训练排队/串行任务冲突风险（已处理）

- 现象：
  - 历史遗留的队列任务可能在旧 PID 结束后触发错误训练命令，造成资源争用或串错配置。
- 处理：
  - 清理旧队列并重建“CNN -> Swin Small”串行链路，确保同一时间单任务占用 MPS。

### 4.3 评估阈值与数据现实不匹配（已修正策略）

- 现象：
  - 使用 `>=120 token` 做长公式评估时样本数为 0。
- 处理：
  - 调整评估阈值到 40/50/60，并明确记录样本量后再解读指标。

## 5. 当前状态

- 长公式相关优化已接入主训练/评估链路。
- `swin_small` 已完成一版训练并导出 Web 模型。
- `swin_small_mps_long` 公平微调已启动，用于下一轮更公平的 CNN vs Swin Small 对比。

## 6. 下一步建议（可执行）

1. 等 `swin_small_mps_long` 跑完后，固定同一 long 子集做最终 A/B 报告（EM/BLEU/相似度/耗时）。
2. 对长样本子集补充人工误差标签（漏 token、括号不平衡、上下标错位）做错误分型。
3. 若 Web 端更关注长公式稳定性，可再评估 `beam_size=3~4` 的延迟/效果折中。
4. 若后续补充更长公式数据（>80/100 token），建议同步提升 `max_seq_len` 与长样本采样权重。

