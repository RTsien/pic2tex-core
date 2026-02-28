"""
PyTorch Dataset for TeXer: loads formula images and LaTeX labels.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from model.tokenizer import LaTeXTokenizer


class FormulaDataset(Dataset):
    """Dataset that loads formula images and tokenized LaTeX sequences."""

    def __init__(
        self,
        data_dir: str,
        tokenizer: LaTeXTokenizer,
        image_size: int = 224,
        max_seq_len: int = 512,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.samples = []
        labels_path = self.data_dir / "labels.jsonl"
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    img_path = self.data_dir / "images" / item["image"]
                    if img_path.exists():
                        self.samples.append({
                            "image_path": str(img_path),
                            "latex": item["latex"],
                        })

        if augment:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.RandomAffine(degrees=2, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)

        token_ids = self.tokenizer.encode(sample["latex"], add_special=True)

        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len - 1] + [self.tokenizer.eos_id]

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
    image_size: int = 224,
    max_seq_len: int = 512,
    augment: bool = False,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    dataset = FormulaDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        image_size=image_size,
        max_seq_len=max_seq_len,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=shuffle,
    )
