"""
Evaluation script for TeXer model.

Metrics:
- Cross-entropy loss
- Token accuracy
- BLEU score
- Exact match rate
- LaTeX compile success rate (optional)
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm

from model.config import TexerConfig
from model.architecture import build_model
from model.tokenizer import LaTeXTokenizer
from model.dataset import create_dataloader


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Compute corpus-level BLEU score using nltk."""
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        refs = [[ref.split()] for ref in references]
        hyps = [pred.split() for pred in predictions]
        smooth = SmoothingFunction().method1
        return corpus_bleu(refs, hyps, smoothing_function=smooth)
    except ImportError:
        return 0.0


def compute_exact_match(predictions: list[str], references: list[str]) -> float:
    """Compute exact match rate after normalization."""
    import re
    matches = 0
    for pred, ref in zip(predictions, references):
        pred_norm = re.sub(r"\s+", " ", pred.strip())
        ref_norm = re.sub(r"\s+", " ", ref.strip())
        if pred_norm == ref_norm:
            matches += 1
    return matches / max(len(predictions), 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    tokenizer: Optional[LaTeXTokenizer] = None,
    fp16: bool = True,
    max_generate: int = 200,
) -> dict:
    """
    Evaluate model on a dataset.

    Returns dict with loss, accuracy, bleu, exact_match.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    all_predictions = []
    all_references = []
    gen_count = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        with autocast(enabled=fp16):
            logits = model(images, input_ids, tgt_key_padding_mask=padding_mask)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

        non_pad = ~padding_mask
        num_tokens = non_pad.sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        preds = logits.argmax(dim=-1)
        correct_tokens += ((preds == target_ids) & non_pad).sum().item()

        if tokenizer and gen_count < max_generate:
            generated = model.generate(
                images,
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                max_len=512,
            )
            for i, gen_ids in enumerate(generated):
                if gen_count >= max_generate:
                    break
                pred_latex = tokenizer.decode(gen_ids, skip_special=True)
                all_predictions.append(pred_latex)

                ref_ids = target_ids[i].tolist()
                ref_ids = [t for t, m in zip(ref_ids, padding_mask[i].tolist()) if not m]
                ref_latex = tokenizer.decode(ref_ids, skip_special=True)
                all_references.append(ref_latex)
                gen_count += 1

    metrics = {
        "loss": total_loss / max(total_tokens, 1),
        "accuracy": correct_tokens / max(total_tokens, 1),
    }

    if all_predictions:
        metrics["bleu"] = compute_bleu(all_predictions, all_references)
        metrics["exact_match"] = compute_exact_match(all_predictions, all_references)
        metrics["num_evaluated"] = len(all_predictions)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate TeXer model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True, help="Test data directory")
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument("--output", type=str, help="Output results JSON path")
    parser.add_argument("--max-generate", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.config:
        config = TexerConfig.load(args.config)
    else:
        config = TexerConfig()

    tokenizer = LaTeXTokenizer()
    if Path(config.data.vocab_path).exists():
        tokenizer = LaTeXTokenizer.load(config.data.vocab_path)
    config.decoder.vocab_size = tokenizer.vocab_size

    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    test_loader = create_dataloader(
        args.data, tokenizer,
        batch_size=args.batch_size,
        image_size=config.data.image_size,
        max_seq_len=config.data.max_seq_len,
        augment=False, shuffle=False,
        num_workers=config.train.num_workers,
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id,
        label_smoothing=0.0,
    )

    metrics = evaluate_model(
        model, test_loader, criterion, device,
        tokenizer=tokenizer,
        max_generate=args.max_generate,
    )

    print("\nEvaluation Results:")
    print(f"  Loss:        {metrics['loss']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  BLEU:        {metrics.get('bleu', 0):.4f}")
    print(f"  Exact Match: {metrics.get('exact_match', 0):.4f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
