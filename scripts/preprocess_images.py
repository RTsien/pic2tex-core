#!/usr/bin/env python3
"""
Offline image preprocessing for faster training.

- Converts to grayscale
- Resolves palette-transparency PNG edge cases
- Resizes with optional aspect-ratio preserving white padding
- Supports in-place rewrite or separate output directory
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps


def crop_to_content(gray_img: Image.Image, threshold: int = 250) -> Image.Image:
    """
    Crop near-white margins to keep formula content readable after resize.
    Pixels darker than `threshold` are treated as foreground.
    """
    if gray_img.mode != "L":
        gray_img = gray_img.convert("L")

    # Foreground mask: dark text becomes 255, white background becomes 0.
    mask = gray_img.point(lambda p: 255 if p < threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return gray_img

    x0, y0, x1, y1 = bbox
    w, h = gray_img.size
    # Small context margin so symbols near boundaries are not cut.
    pad_x = max(2, int((x1 - x0) * 0.03))
    pad_y = max(2, int((y1 - y0) * 0.03))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)
    return gray_img.crop((x0, y0, x1, y1))


def iter_image_paths(split_dir: Path, limit: int | None = None) -> Iterable[Path]:
    labels_path = split_dir / "labels.jsonl"
    images_dir = split_dir / "images"
    if not labels_path.exists() or not images_dir.exists():
        return []

    paths: list[Path] = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            img_name = item.get("image")
            if not img_name:
                continue
            p = images_dir / img_name
            if p.exists():
                paths.append(p)
            if limit is not None and len(paths) >= limit:
                break
    return paths


def process_one(
    src_path: Path,
    dst_path: Path,
    image_height: int,
    image_width: int,
    keep_aspect_ratio: bool,
    crop_content: bool,
    autocontrast_cutoff: float,
    unsharp_radius: float,
    unsharp_percent: int,
    unsharp_threshold: int,
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        if img.mode == "P" and "transparency" in img.info:
            img = img.convert("RGBA")
        gray = img.convert("L")
        if crop_content:
            gray = crop_to_content(gray)
        if keep_aspect_ratio:
            w, h = gray.size
            if w == 0 or h == 0:
                processed = gray.resize((image_width, image_height), Image.BILINEAR)
            else:
                scale = min(image_width / w, image_height / h)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                resized = gray.resize((new_w, new_h), Image.BILINEAR)
                processed = Image.new("L", (image_width, image_height), color=255)
                processed.paste(resized, ((image_width - new_w) // 2, (image_height - new_h) // 2))
        else:
            processed = gray.resize((image_width, image_height), Image.BILINEAR)

    if autocontrast_cutoff > 0:
        processed = ImageOps.autocontrast(processed, cutoff=autocontrast_cutoff)
    if unsharp_percent > 0:
        processed = processed.filter(
            ImageFilter.UnsharpMask(
                radius=unsharp_radius,
                percent=unsharp_percent,
                threshold=unsharp_threshold,
            )
        )

    tmp = dst_path.with_suffix(dst_path.suffix + ".tmp")
    processed.save(tmp, format="PNG", optimize=True)
    os.replace(tmp, dst_path)


def preprocess_split(
    split: str,
    split_dir: Path,
    image_height: int,
    image_width: int,
    keep_aspect_ratio: bool,
    workers: int,
    limit: int | None,
    output_root: Path | None,
    crop_content: bool,
    autocontrast_cutoff: float,
    unsharp_radius: float,
    unsharp_percent: int,
    unsharp_threshold: int,
) -> tuple[int, int]:
    src_paths = list(iter_image_paths(split_dir, limit=limit))
    if not src_paths:
        print(f"{split}: no images found, skip")
        return 0, 0

    total = len(src_paths)
    def print_progress_line(completed: int, failed_count: int) -> None:
        percent = (completed / total) * 100 if total else 100.0
        # Single-line overwrite output for cleaner terminal progress.
        print(
            f"\r{split}: progress {completed}/{total} ({percent:.1f}%), failed={failed_count}",
            end="",
            flush=True,
        )

    print_progress_line(0, 0)

    done = 0
    failed = 0
    completed = 0
    progress_interval = max(100, total // 100)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = []
        for src in src_paths:
            if output_root is None:
                dst = src
            else:
                rel = src.relative_to(split_dir.parent)
                dst = output_root / rel
            futures.append(
                ex.submit(
                    process_one,
                    src,
                    dst,
                    image_height,
                    image_width,
                    keep_aspect_ratio,
                    crop_content,
                    autocontrast_cutoff,
                    unsharp_radius,
                    unsharp_percent,
                    unsharp_threshold,
                )
            )

        for fut in as_completed(futures):
            try:
                fut.result()
                done += 1
            except Exception:
                failed += 1
            completed += 1

            if completed % progress_interval == 0 or completed == total:
                print_progress_line(completed, failed)

    # End the progress line for this split before summary logging.
    print(flush=True)

    return done, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess dataset images for faster training")
    parser.add_argument("--dataset-root", type=str, default="data/processed")
    parser.add_argument("--image-size", type=int, default=None, help="Legacy square size fallback")
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--keep-aspect-ratio", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="If unset, rewrite images in place",
    )
    parser.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated splits")
    parser.add_argument("--autocontrast-cutoff", type=float, default=0.0)
    parser.add_argument("--unsharp-radius", type=float, default=1.5)
    parser.add_argument("--unsharp-percent", type=int, default=0)
    parser.add_argument("--unsharp-threshold", type=int, default=2)
    parser.add_argument(
        "--no-crop-content",
        action="store_true",
        help="Disable auto-cropping white margins before resize",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else None
    if args.image_height is None and args.image_width is None:
        if args.image_size is None:
            raise SystemExit("Specify --image-height/--image-width or --image-size")
        image_height = image_width = int(args.image_size)
    else:
        if args.image_height is None or args.image_width is None:
            raise SystemExit("Both --image-height and --image-width are required together")
        image_height = int(args.image_height)
        image_width = int(args.image_width)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    total_done = 0
    total_failed = 0
    for split in splits:
        split_dir = dataset_root / split
        done, failed = preprocess_split(
            split=split,
            split_dir=split_dir,
            image_height=image_height,
            image_width=image_width,
            keep_aspect_ratio=args.keep_aspect_ratio,
            workers=args.workers,
            limit=args.limit_per_split,
            output_root=output_root,
            crop_content=not args.no_crop_content,
            autocontrast_cutoff=args.autocontrast_cutoff,
            unsharp_radius=args.unsharp_radius,
            unsharp_percent=args.unsharp_percent,
            unsharp_threshold=args.unsharp_threshold,
        )
        total_done += done
        total_failed += failed
        print(f"{split}: processed={done}, failed={failed}, size={image_height}x{image_width}")

    print(f"total: processed={total_done}, failed={total_failed}")
    if total_failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

