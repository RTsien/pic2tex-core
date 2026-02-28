"""
Training script for TeXer model.

Supports:
- Multi-stage training (pretrain on synthetic, finetune on real)
- Mixed precision (FP16/BF16 on CUDA, BF16 on MPS where available)
- Cosine LR schedule with warmup
- Early stopping
- Checkpoint save/resume
- Optional W&B logging
- Accelerated training on CUDA, Apple MPS (M-series chips), or CPU
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from model.config import TexerConfig
from model.architecture import build_model
from model.tokenizer import LaTeXTokenizer
from model.dataset import create_dataloader
from model.evaluate import evaluate_model


def select_device(force: str = None) -> torch.device:
    """Auto-detect the best available device: CUDA > MPS > CPU.

    Args:
        force: Override auto-detection with "cuda", "mps", or "cpu".
    """
    if force:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_amp_config(device: torch.device, fp16_requested: bool) -> dict:
    """Return autocast / GradScaler settings appropriate for the device.

    Returns dict with keys: use_amp, amp_device_type, amp_dtype, use_scaler.
    """
    if device.type == "cuda" and fp16_requested:
        return {
            "use_amp": True,
            "amp_device_type": "cuda",
            "amp_dtype": torch.float16,
            "use_scaler": True,
        }
    if device.type == "mps":
        return {
            "use_amp": False,
            "amp_device_type": "cpu",
            "amp_dtype": torch.float32,
            "use_scaler": False,
        }
    return {
        "use_amp": False,
        "amp_device_type": "cpu",
        "amp_dtype": torch.float32,
        "use_scaler": False,
    }


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    criterion: nn.Module,
    device: torch.device,
    amp_cfg: dict,
    grad_clip: float = 1.0,
) -> dict:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        optimizer.zero_grad()

        with autocast(
            device_type=amp_cfg["amp_device_type"],
            dtype=amp_cfg["amp_dtype"],
            enabled=amp_cfg["use_amp"],
        ):
            logits = model(images, input_ids, tgt_key_padding_mask=padding_mask)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

        if amp_cfg["use_scaler"]:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()

        non_pad = ~padding_mask
        num_tokens = non_pad.sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        preds = logits.argmax(dim=-1)
        correct_tokens += ((preds == target_ids) & non_pad).sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    accuracy = correct_tokens / max(total_tokens, 1)
    return {"loss": avg_loss, "accuracy": accuracy}


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_val_loss: float,
    path: str,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
    }, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    scaler=None,
) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint


def train(config: TexerConfig, resume: str = None, force_device: str = None):
    device = select_device(force_device)
    amp_cfg = get_amp_config(device, config.train.fp16)
    print(f"Using device: {device}")
    if device.type == "mps":
        print("  Apple MPS acceleration enabled (Metal Performance Shaders)")
    if amp_cfg["use_amp"]:
        print(f"  Mixed precision: {amp_cfg['amp_dtype']}")

    tokenizer = LaTeXTokenizer()
    if Path(config.data.vocab_path).exists():
        tokenizer = LaTeXTokenizer.load(config.data.vocab_path)
    config.decoder.vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    model = build_model(config)
    model = model.to(device)
    params = model.count_parameters()
    print(f"Model parameters: {params['total']:,} ({params['total_mb']:.1f} MB)")

    pin = device.type == "cuda"
    train_loader = create_dataloader(
        config.data.train_dir, tokenizer,
        batch_size=config.train.batch_size,
        image_size=config.data.image_size,
        max_seq_len=config.data.max_seq_len,
        augment=True, shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=pin,
    )
    val_loader = create_dataloader(
        config.data.val_dir, tokenizer,
        batch_size=config.train.batch_size,
        image_size=config.data.image_size,
        max_seq_len=config.data.max_seq_len,
        augment=False, shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=pin,
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id,
        label_smoothing=config.train.label_smoothing,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    total_steps = len(train_loader) * config.train.num_epochs
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=config.train.warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - config.train.warmup_steps,
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.train.warmup_steps],
    )

    scaler = GradScaler(enabled=amp_cfg["use_scaler"])

    start_epoch = 0
    best_val_loss = float("inf")
    best_val_bleu = -1.0
    patience_counter = 0

    if resume:
        print(f"Resuming from {resume}")
        ckpt = load_checkpoint(resume, model, optimizer, scheduler, scaler)
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

    wandb_run = None
    if config.train.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(project=config.train.wandb_project, config={
                "encoder": config.encoder.__dict__,
                "decoder": config.decoder.__dict__,
                "train": config.train.__dict__,
            })
        except ImportError:
            print("wandb not installed, skipping logging")

    ckpt_dir = Path(config.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training from epoch {start_epoch}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Total epochs: {config.train.num_epochs}")
    print(f"Batch size: {config.train.batch_size}")
    print(f"Learning rate: {config.train.learning_rate}")
    print()

    for epoch in range(start_epoch, config.train.num_epochs):
        t0 = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion, device,
            amp_cfg=amp_cfg,
            grad_clip=config.train.grad_clip,
        )

        epoch_time = time.time() - t0
        print(
            f"Epoch {epoch+1}/{config.train.num_epochs} "
            f"| train_loss: {train_metrics['loss']:.4f} "
            f"| train_acc: {train_metrics['accuracy']:.4f} "
            f"| time: {epoch_time:.1f}s"
        )

        if (epoch + 1) % config.train.eval_every == 0:
            val_metrics = evaluate_model(
                model, val_loader, criterion, device,
                tokenizer=tokenizer,
                amp_cfg=amp_cfg,
            )
            print(
                f"  val_loss: {val_metrics['loss']:.4f} "
                f"| val_acc: {val_metrics['accuracy']:.4f} "
                f"| val_bleu: {val_metrics.get('bleu', 0):.4f} "
                f"| val_unique: {val_metrics.get('unique_prediction_ratio', 0):.4f}"
            )

            if wandb_run:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"],
                    "val_bleu": val_metrics.get("bleu", 0),
                    "val_unique_prediction_ratio": val_metrics.get("unique_prediction_ratio", 0),
                    "val_most_common_prediction_fraction": val_metrics.get("most_common_prediction_fraction", 0),
                    "lr": optimizer.param_groups[0]["lr"],
                })

            # Detect output collapse where many inputs decode to the same formula.
            # This often does not show up in token-level loss, so we warn explicitly.
            num_eval = val_metrics.get("num_evaluated", 0)
            collapse_fraction = val_metrics.get("most_common_prediction_fraction", 0.0)
            if num_eval >= 20 and collapse_fraction >= 0.8:
                print(
                    "  WARNING: output collapse detected "
                    f"(most_common_prediction_fraction={collapse_fraction:.2%}, "
                    f"num_evaluated={num_eval})."
                )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch, best_val_loss,
                    str(ckpt_dir / "best.pt"),
                )
                print(f"  -> New best model saved (val_loss={best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= config.train.patience:
                    print(f"  Early stopping after {patience_counter} epochs without improvement")
                    break

            val_bleu = val_metrics.get("bleu", 0.0)
            if val_bleu > best_val_bleu:
                best_val_bleu = val_bleu
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch, best_val_loss,
                    str(ckpt_dir / "best_bleu.pt"),
                )
                print(f"  -> New BLEU-best model saved (val_bleu={best_val_bleu:.4f})")

        if (epoch + 1) % config.train.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, best_val_loss,
                str(ckpt_dir / f"epoch_{epoch+1}.pt"),
            )

    save_checkpoint(
        model, optimizer, scheduler, scaler,
        epoch, best_val_loss,
        str(ckpt_dir / "last.pt"),
    )
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")

    if wandb_run:
        wandb_run.finish()


def main():
    parser = argparse.ArgumentParser(description="Train TeXer model")
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument("--resume", type=str, help="Checkpoint to resume from")
    parser.add_argument("--device", type=str, choices=["cuda", "mps", "cpu"],
                        help="Force device (default: auto-detect)")
    args = parser.parse_args()

    if args.config:
        config = TexerConfig.load(args.config)
    else:
        config = TexerConfig()

    train(config, resume=args.resume, force_device=args.device)


if __name__ == "__main__":
    main()
