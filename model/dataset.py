"""
PyTorch Dataset for TeXer: loads formula images and LaTeX labels.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

from model.tokenizer import LaTeXTokenizer


class ResizeWithPad:
    """Resize with optional aspect-ratio preserving white padding."""

    def __init__(self, target_height: int, target_width: int, keep_aspect_ratio: bool = True):
        self.target_height = target_height
        self.target_width = target_width
        self.keep_aspect_ratio = keep_aspect_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        if not self.keep_aspect_ratio:
            return img.resize((self.target_width, self.target_height), Image.BILINEAR)

        w, h = img.size
        if w == 0 or h == 0:
            return img.resize((self.target_width, self.target_height), Image.BILINEAR)

        scale = min(self.target_width / w, self.target_height / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = img.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new("L", (self.target_width, self.target_height), color=255)
        offset = ((self.target_width - new_w) // 2, (self.target_height - new_h) // 2)
        canvas.paste(resized, offset)
        return canvas


class FormulaDataset(Dataset):
    """Dataset that loads formula images and tokenized LaTeX sequences."""

    def __init__(
        self,
        data_dir: str,
        tokenizer: LaTeXTokenizer,
        image_height: int = 224,
        image_width: int = 224,
        max_seq_len: int = 512,
        keep_aspect_ratio: bool = True,
        augment: bool = False,
        preprocessed_images: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.preprocessed_images = preprocessed_images

        self.samples = []
        self.sample_lengths: list[int] = []
        labels_path = self.data_dir / "labels.jsonl"
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    img_path = self.data_dir / "images" / item["image"]
                    if img_path.exists():
                        token_ids = self.tokenizer.encode(item["latex"], add_special=True)
                        if len(token_ids) > self.max_seq_len:
                            token_ids = token_ids[: self.max_seq_len - 1] + [self.tokenizer.eos_id]
                        self.samples.append({
                            "image_path": str(img_path),
                            "latex": item["latex"],
                            "token_ids": token_ids,
                        })
                        self.sample_lengths.append(max(len(token_ids) - 1, 1))

        if preprocessed_images:
            # Images are already grayscale and resized offline.
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            resize_op = ResizeWithPad(
                target_height=image_height,
                target_width=image_width,
                keep_aspect_ratio=keep_aspect_ratio,
            )
            if augment:
                self.transform = transforms.Compose([
                    resize_op,
                    transforms.RandomAffine(
                        degrees=2,
                        translate=(0.05, 0.05),
                        scale=(0.95, 1.05),
                        fill=255,
                    ),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ])
            else:
                self.transform = transforms.Compose([
                    resize_op,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        with Image.open(sample["image_path"]) as img:
            # Palette + transparency PNGs trigger warnings when converted directly.
            if img.mode == "P" and "transparency" in img.info:
                img = img.convert("RGBA")
            image = img.convert("L")
        image = self.transform(image)

        token_ids = sample["token_ids"]

        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        return {
            "image": image,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "length": len(input_ids),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad sequences to the same length within a batch."""
    images = torch.stack([item["image"] for item in batch])
    lengths = [item["length"] for item in batch]
    max_len = max(lengths)

    pad_id = 0  # <pad>
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    target_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    padding_mask = torch.ones((len(batch), max_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        L = item["length"]
        input_ids[i, :L] = item["input_ids"]
        target_ids[i, :L] = item["target_ids"]
        padding_mask[i, :L] = False

    return {
        "images": images,
        "input_ids": input_ids,
        "target_ids": target_ids,
        "padding_mask": padding_mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }


def create_dataloader(
    data_dir: str,
    tokenizer: LaTeXTokenizer,
    batch_size: int = 64,
    image_height: int = 224,
    image_width: int = 224,
    max_seq_len: int = 512,
    keep_aspect_ratio: bool = True,
    augment: bool = False,
    shuffle: bool = True,
    long_formula_min_tokens: int = 120,
    long_formula_oversample_factor: float = 1.0,
    num_workers: int = 4,
    pin_memory: bool = False,
    preprocessed_images: bool = False,
) -> DataLoader:
    dataset = FormulaDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        image_height=image_height,
        image_width=image_width,
        max_seq_len=max_seq_len,
        keep_aspect_ratio=keep_aspect_ratio,
        augment=augment,
        preprocessed_images=preprocessed_images,
    )
    sampler = None
    use_weighted_sampling = (
        shuffle
        and long_formula_oversample_factor > 1.0
        and len(dataset.sample_lengths) == len(dataset)
    )
    if use_weighted_sampling:
        weights = [
            long_formula_oversample_factor if L >= long_formula_min_tokens else 1.0
            for L in dataset.sample_lengths
        ]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )

    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
        "drop_last": shuffle or sampler is not None,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 2

    return DataLoader(**dataloader_kwargs)
