"""
Dataset builder: merges synthetic, external, and Qwen-annotated data
into a unified train/val/test split with consistent format.
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from collections import Counter

from tqdm import tqdm


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def filter_valid_samples(items: list[dict], images_dir: str) -> list[dict]:
    """Keep only samples whose image file exists and LaTeX is non-empty."""
    valid = []
    images_path = Path(images_dir)
    for item in items:
        latex = item.get("latex", "").strip()
        if not latex:
            continue
        if len(latex) > 1024:
            continue
        img_file = item.get("image", "")
        if img_file and (images_path / img_file).exists():
            valid.append(item)
        elif not img_file:
            valid.append(item)
    return valid


def deduplicate(items: list[dict]) -> list[dict]:
    """Remove duplicate LaTeX formulas, keeping the first occurrence."""
    seen = set()
    unique = []
    for item in items:
        key = item["latex"].strip()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def compute_stats(items: list[dict]) -> dict:
    cat_counts = Counter(item.get("category", "unknown") for item in items)
    latex_lengths = [len(item["latex"]) for item in items]
    return {
        "total": len(items),
        "categories": dict(cat_counts),
        "avg_latex_length": sum(latex_lengths) / len(latex_lengths) if latex_lengths else 0,
        "max_latex_length": max(latex_lengths) if latex_lengths else 0,
        "min_latex_length": min(latex_lengths) if latex_lengths else 0,
    }


def split_dataset(
    items: list[dict],
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list, list, list]:
    random.seed(seed)
    random.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]

    return train, val, test


def copy_images(items: list[dict], src_dir: str, dst_dir: str) -> None:
    """Copy image files from source to destination directory."""
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    for item in items:
        img_name = item.get("image", "")
        if not img_name:
            continue
        src_path = src / img_name
        if src_path.exists():
            shutil.copy2(src_path, dst / img_name)


def build(
    synthetic_dir: str = None,
    external_dir: str = None,
    annotated_dir: str = None,
    output_dir: str = "data/processed",
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> None:
    all_items = []
    image_sources = {}

    if synthetic_dir:
        synth_path = Path(synthetic_dir)
        labels_file = synth_path / "labels.jsonl"
        if labels_file.exists():
            items = load_jsonl(str(labels_file))
            items = filter_valid_samples(items, str(synth_path / "images"))
            print(f"Synthetic: {len(items)} valid samples")
            all_items.extend(items)
            image_sources.update({item["image"]: str(synth_path / "images") for item in items if item.get("image")})

    if external_dir:
        ext_path = Path(external_dir)
        labels_file = ext_path / "labels.jsonl"
        if labels_file.exists():
            items = load_jsonl(str(labels_file))
            items = filter_valid_samples(items, str(ext_path / "images"))
            print(f"External: {len(items)} valid samples")
            all_items.extend(items)
            image_sources.update({item["image"]: str(ext_path / "images") for item in items if item.get("image")})

    if annotated_dir:
        ann_path = Path(annotated_dir)
        labels_file = ann_path / "labels.jsonl"
        if labels_file.exists():
            items = load_jsonl(str(labels_file))
            items = filter_valid_samples(items, str(ann_path / "images"))
            print(f"Annotated: {len(items)} valid samples")
            all_items.extend(items)
            image_sources.update({item["image"]: str(ann_path / "images") for item in items if item.get("image")})

    if not all_items:
        print("No data found. Please generate or download data first.")
        return

    print(f"\nTotal before dedup: {len(all_items)}")
    all_items = deduplicate(all_items)
    print(f"Total after dedup: {len(all_items)}")

    train, val, test = split_dataset(all_items, train_ratio, val_ratio, seed)
    print(f"\nSplit: train={len(train)}, val={len(val)}, test={len(test)}")

    out = Path(output_dir)
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        split_dir = out / split_name
        save_jsonl(split_data, str(split_dir / "labels.jsonl"))

        images_dst = split_dir / "images"
        images_dst.mkdir(parents=True, exist_ok=True)
        for item in tqdm(split_data, desc=f"Copying {split_name} images"):
            img_name = item.get("image", "")
            if img_name and img_name in image_sources:
                src_path = Path(image_sources[img_name]) / img_name
                if src_path.exists():
                    shutil.copy2(src_path, images_dst / img_name)

    stats = compute_stats(all_items)
    stats_path = out / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nDataset stats: {json.dumps(stats, indent=2)}")
    print(f"Dataset saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build unified training dataset")
    parser.add_argument("--synthetic-dir", type=str, help="Synthetic data directory")
    parser.add_argument("--external-dir", type=str, help="External dataset directory")
    parser.add_argument("--annotated-dir", type=str, help="Qwen-annotated data directory")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build(
        synthetic_dir=args.synthetic_dir,
        external_dir=args.external_dir,
        annotated_dir=args.annotated_dir,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
