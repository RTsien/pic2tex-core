"""
Image augmentation pipeline for formula images.

Applies realistic distortions: noise, blur, rotation, scaling, contrast changes,
and simulated paper/screen backgrounds.
"""

import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw


def add_gaussian_noise(img: Image.Image, mean: float = 0, std: float = 10) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(mean, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_salt_pepper_noise(img: Image.Image, prob: float = 0.01) -> Image.Image:
    arr = np.array(img)
    mask = np.random.random(arr.shape[:2])
    arr[mask < prob / 2] = 0
    arr[mask > 1 - prob / 2] = 255
    return Image.fromarray(arr)


def random_blur(img: Image.Image, max_radius: float = 1.5) -> Image.Image:
    radius = random.uniform(0.3, max_radius)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def random_rotation(img: Image.Image, max_angle: float = 3.0) -> Image.Image:
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)


def random_scale(img: Image.Image, scale_range: tuple = (0.85, 1.15)) -> Image.Image:
    factor = random.uniform(*scale_range)
    new_w = max(1, int(img.width * factor))
    new_h = max(1, int(img.height * factor))
    return img.resize((new_w, new_h), Image.LANCZOS)


def random_brightness(img: Image.Image, range_: tuple = (0.7, 1.3)) -> Image.Image:
    factor = random.uniform(*range_)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def random_contrast(img: Image.Image, range_: tuple = (0.7, 1.3)) -> Image.Image:
    factor = random.uniform(*range_)
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def add_paper_texture(img: Image.Image) -> Image.Image:
    """Simulate slightly uneven paper background."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    texture = np.random.normal(0, 3, (h, w))
    if arr.ndim == 3:
        texture = texture[:, :, np.newaxis]
    arr = np.clip(arr + texture, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def random_padding(img: Image.Image, max_pad: int = 20) -> Image.Image:
    """Add random asymmetric padding around the formula."""
    top = random.randint(2, max_pad)
    bottom = random.randint(2, max_pad)
    left = random.randint(2, max_pad)
    right = random.randint(2, max_pad)

    if img.mode == "L":
        new_img = Image.new("L", (img.width + left + right, img.height + top + bottom), 255)
    else:
        new_img = Image.new("RGB", (img.width + left + right, img.height + top + bottom), (255, 255, 255))
    new_img.paste(img, (left, top))
    return new_img


def random_line_artifact(img: Image.Image, prob: float = 0.1) -> Image.Image:
    """Occasionally draw a faint line to simulate scan artifacts."""
    if random.random() > prob:
        return img
    draw = ImageDraw.Draw(img)
    w, h = img.size
    gray = random.randint(180, 230)
    if random.random() < 0.5:
        y = random.randint(0, h - 1)
        draw.line([(0, y), (w, y)], fill=gray, width=1)
    else:
        x = random.randint(0, w - 1)
        draw.line([(x, 0), (x, h)], fill=gray, width=1)
    return img


def random_background_color(img: Image.Image) -> Image.Image:
    """Change background from pure white to a slight tint."""
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 2:
        tint = random.uniform(-15, 0)
        mask = arr > 240
        arr[mask] += tint
    else:
        for c in range(3):
            tint = random.uniform(-15, 0)
            channel = arr[:, :, c]
            mask = channel > 240
            channel[mask] += tint
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def augment_image(
    img: Image.Image,
    intensity: str = "medium",
) -> Image.Image:
    """
    Apply a random subset of augmentations.

    intensity: "light", "medium", "heavy"
    """
    if intensity == "light":
        augmentations = [
            (0.3, random_brightness),
            (0.3, random_contrast),
            (0.2, lambda im: add_gaussian_noise(im, std=5)),
            (0.5, random_padding),
        ]
    elif intensity == "heavy":
        augmentations = [
            (0.7, random_brightness),
            (0.7, random_contrast),
            (0.5, lambda im: add_gaussian_noise(im, std=15)),
            (0.4, lambda im: random_blur(im, max_radius=2.0)),
            (0.5, lambda im: random_rotation(im, max_angle=5.0)),
            (0.4, random_scale),
            (0.3, add_salt_pepper_noise),
            (0.5, add_paper_texture),
            (0.8, random_padding),
            (0.2, random_line_artifact),
            (0.4, random_background_color),
        ]
    else:  # medium
        augmentations = [
            (0.5, random_brightness),
            (0.5, random_contrast),
            (0.3, lambda im: add_gaussian_noise(im, std=10)),
            (0.3, lambda im: random_blur(im, max_radius=1.5)),
            (0.3, lambda im: random_rotation(im, max_angle=3.0)),
            (0.2, random_scale),
            (0.6, random_padding),
            (0.3, add_paper_texture),
            (0.1, random_line_artifact),
            (0.3, random_background_color),
        ]

    for prob, aug_fn in augmentations:
        if random.random() < prob:
            try:
                img = aug_fn(img)
            except Exception:
                pass

    return img
