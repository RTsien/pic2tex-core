"""
Formula renderer: converts LaTeX strings to PNG images using matplotlib.

Supports multiple fonts, DPI settings, and integrates with the augmentation pipeline.
"""

import argparse
import json
import random
import io
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from data_gen.augmentation import augment_image
from data_gen.formula_collector import generate_formulas


FONT_CONFIGS = [
    {"family": "serif", "mathtext_fontset": "cm"},
    {"family": "serif", "mathtext_fontset": "stix"},
    {"family": "serif", "mathtext_fontset": "dejavuserif"},
]

DPI_RANGE = (100, 300)
TARGET_HEIGHT = 64
MAX_WIDTH = 512


def render_latex_to_image(
    latex: str,
    dpi: int = 200,
    font_config: Optional[dict] = None,
    fontsize: int = 16,
) -> Optional[Image.Image]:
    """Render a LaTeX formula string to a PIL Image."""
    if font_config is None:
        font_config = random.choice(FONT_CONFIGS)

    plt.rcParams.update({
        "font.family": font_config["family"],
        "mathtext.fontset": font_config["mathtext_fontset"],
    })

    fig = plt.figure(figsize=(10, 1))
    fig.patch.set_facecolor("white")

    try:
        fig.text(
            0.5, 0.5,
            f"${latex}$",
            fontsize=fontsize,
            ha="center", va="center",
            color="black",
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    pad_inches=0.05, facecolor="white")
        buf.seek(0)
        img = Image.open(buf).convert("L")

        img = _auto_crop(img)
        img = _normalize_height(img, TARGET_HEIGHT, MAX_WIDTH)

        return img
    except Exception:
        return None
    finally:
        plt.close(fig)


def _auto_crop(img: Image.Image, threshold: int = 240, margin: int = 4) -> Image.Image:
    """Crop whitespace around the formula."""
    import numpy as np
    arr = np.array(img)
    mask = arr < threshold

    if not mask.any():
        return img

    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    rmin, rmax = rows.argmax(), len(rows) - rows[::-1].argmax()
    cmin, cmax = cols.argmax(), len(cols) - cols[::-1].argmax()

    rmin = max(0, rmin - margin)
    rmax = min(arr.shape[0], rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(arr.shape[1], cmax + margin)

    return img.crop((cmin, rmin, cmax, rmax))


def _normalize_height(img: Image.Image, target_h: int, max_w: int) -> Image.Image:
    """Resize to target height while maintaining aspect ratio, cap width."""
    if img.height == 0:
        return img
    ratio = target_h / img.height
    new_w = min(int(img.width * ratio), max_w)
    return img.resize((new_w, target_h), Image.LANCZOS)


def _render_single(args: tuple) -> Optional[tuple]:
    """Worker function for parallel rendering."""
    item, output_dir, do_augment, aug_intensity = args
    formula_id = item["id"]
    latex = item["latex"]

    dpi = random.randint(*DPI_RANGE)
    fontsize = random.randint(14, 22)
    font_config = random.choice(FONT_CONFIGS)

    img = render_latex_to_image(latex, dpi=dpi, font_config=font_config, fontsize=fontsize)
    if img is None:
        return None

    if do_augment:
        img = augment_image(img, intensity=aug_intensity)

    img_filename = f"{formula_id}.png"
    img_path = Path(output_dir) / "images" / img_filename
    img.save(img_path, "PNG")

    return {
        "id": formula_id,
        "image": img_filename,
        "latex": latex,
        "category": item.get("category", "unknown"),
    }


def render_dataset(
    formulas: list,
    output_dir: str,
    augment: bool = True,
    aug_intensity: str = "medium",
    num_workers: int = 4,
) -> list:
    """Render a list of formula dicts to images, return metadata list."""
    out = Path(output_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    tasks = [(item, output_dir, augment, aug_intensity) for item in formulas]
    results = []

    if num_workers <= 1:
        for task in tqdm(tasks, desc="Rendering"):
            result = _render_single(task)
            if result:
                results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_render_single, t): t for t in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering"):
                result = future.result()
                if result:
                    results.append(result)

    labels_path = out / "labels.jsonl"
    with open(labels_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Rendered {len(results)}/{len(formulas)} formulas to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Render LaTeX formulas to images")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--count", type=int, default=10000, help="Number of formulas")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--aug-intensity", choices=["light", "medium", "heavy"], default="medium")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    random.seed(args.seed)
    formulas = list(generate_formulas(args.count, args.seed))

    render_dataset(
        formulas,
        output_dir=args.output,
        augment=not args.no_augment,
        aug_intensity=args.aug_intensity,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
