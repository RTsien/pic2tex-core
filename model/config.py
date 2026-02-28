"""
Model and training configuration.
"""

from dataclasses import dataclass, field
from typing import Optional
import yaml
from pathlib import Path


@dataclass
class EncoderConfig:
    encoder_type: str = "swin"  # "swin" or "cnn"
    image_size: int = 224
    patch_size: int = 4
    in_channels: int = 1  # grayscale
    embed_dim: int = 64
    depths: list[int] = field(default_factory=lambda: [2, 2, 6, 2])
    num_heads: list[int] = field(default_factory=lambda: [2, 4, 8, 16])
    window_size: int = 7
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    cnn_channels: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    cnn_dropout: float = 0.1


@dataclass
class DecoderConfig:
    vocab_size: int = 600
    max_seq_len: int = 512
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    fp16: bool = True
    num_workers: int = 4
    save_every: int = 5
    eval_every: int = 1
    patience: int = 10  # early stopping
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    wandb_project: Optional[str] = None


@dataclass
class DataConfig:
    train_dir: str = "data/processed/train"
    val_dir: str = "data/processed/val"
    test_dir: str = "data/processed/test"
    vocab_path: str = "model/vocab.json"
    image_size: int = 224
    max_seq_len: int = 512


@dataclass
class TexerConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        d = {
            "encoder": self.encoder.__dict__,
            "decoder": self.decoder.__dict__,
            "train": self.train.__dict__,
            "data": self.data.__dict__,
        }
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> "TexerConfig":
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        config = cls()
        if "encoder" in d:
            config.encoder = EncoderConfig(**d["encoder"])
        if "decoder" in d:
            config.decoder = DecoderConfig(**d["decoder"])
        if "train" in d:
            config.train = TrainConfig(**d["train"])
        if "data" in d:
            config.data = DataConfig(**d["data"])
        return config
