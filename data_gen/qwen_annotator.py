"""
Qwen3-VL annotator: uses Qwen3-VL-4B-Instruct as a teacher model for:
  1. Annotating real-world formula screenshots with LaTeX labels
  2. Validating synthetic data quality (round-trip verification)
  3. Generating alternative LaTeX representations for diversity

Supports both vLLM (fast batch inference) and HuggingFace transformers backends.
"""

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm


MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

ANNOTATE_PROMPT = (
    "You are a LaTeX OCR expert. Look at this image of a mathematical or scientific formula "
    "and output ONLY the LaTeX code that reproduces it. Do not include dollar signs or "
    "\\begin{equation}. Output only the raw LaTeX expression."
)

VALIDATE_PROMPT = (
    "You are a LaTeX OCR expert. Look at this image of a formula and output the LaTeX code. "
    "Output ONLY the raw LaTeX expression, nothing else."
)

ALTERNATIVE_PROMPT = (
    "Given this LaTeX formula: {latex}\n"
    "Generate 3 alternative but semantically equivalent LaTeX representations. "
    "Output each on a separate line, with no numbering or extra text."
)


def _encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_latex(text: str) -> str:
    """Extract LaTeX from model output, stripping common wrappers."""
    text = text.strip()
    for wrapper in [r"\[", r"\]", r"\(", r"\)", "$$", "$"]:
        text = text.replace(wrapper, "")
    match = re.search(r"```(?:latex)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    return text.strip()


class QwenAnnotatorVLLM:
    """Batch annotator using vLLM for high-throughput inference."""

    def __init__(self, model_id: str = MODEL_ID, tensor_parallel_size: int = 1):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=model_id,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=4096,
            dtype="bfloat16",
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=512,
            top_p=0.95,
        )

    def annotate_images(self, image_paths: list[str]) -> list[str]:
        """Annotate a batch of formula images, returning LaTeX strings."""
        from vllm import TextPrompt
        prompts = []
        for img_path in image_paths:
            b64 = _encode_image_base64(img_path)
            prompt = {
                "prompt": f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                          f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
                          f"{ANNOTATE_PROMPT}<|im_end|>\n"
                          f"<|im_start|>assistant\n",
                "multi_modal_data": {
                    "image": Image.open(img_path).convert("RGB"),
                },
            }
            prompts.append(prompt)

        outputs = self.llm.generate(prompts, self.sampling_params)
        results = []
        for output in outputs:
            text = output.outputs[0].text
            results.append(_extract_latex(text))
        return results

    def validate_batch(
        self, image_paths: list[str], expected_latex: list[str]
    ) -> list[dict]:
        """Validate synthetic data by comparing model output with expected LaTeX."""
        predicted = self.annotate_images(image_paths)
        results = []
        for img_path, pred, expected in zip(image_paths, predicted, expected_latex):
            is_match = _normalize_latex(pred) == _normalize_latex(expected)
            results.append({
                "image": img_path,
                "expected": expected,
                "predicted": pred,
                "valid": is_match,
            })
        return results


class QwenAnnotatorTransformers:
    """Annotator using HuggingFace transformers (lower throughput, easier setup)."""

    def __init__(self, model_id: str = MODEL_ID, device: str = "auto"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        # Use an auto model class so Qwen2.5-VL / Qwen3-VL configs map correctly.
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype="auto",
            device_map=device,
            trust_remote_code=True,
        )

    def annotate_single(self, image_path: str) -> str:
        """Annotate a single formula image."""
        import torch
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": ANNOTATE_PROMPT},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
            )

        generated = output_ids[0][inputs.input_ids.shape[1]:]
        result = self.processor.decode(generated, skip_special_tokens=True)
        return _extract_latex(result)

    def annotate_images(self, image_paths: list[str]) -> list[str]:
        """Annotate multiple images sequentially."""
        results = []
        for path in tqdm(image_paths, desc="Annotating"):
            results.append(self.annotate_single(path))
        return results

    def generate_alternatives(self, latex: str) -> list[str]:
        """Generate alternative LaTeX representations for training diversity."""
        import torch
        prompt = ALTERNATIVE_PROMPT.format(latex=latex)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
            )

        generated = output_ids[0][inputs.input_ids.shape[1]:]
        result = self.processor.decode(generated, skip_special_tokens=True)
        alternatives = [_extract_latex(line) for line in result.strip().split("\n") if line.strip()]
        return alternatives


def _normalize_latex(latex: str) -> str:
    """Normalize LaTeX for comparison: remove extra spaces, standardize commands."""
    latex = re.sub(r"\s+", " ", latex.strip())
    latex = latex.replace(r"\left(", "(").replace(r"\right)", ")")
    latex = latex.replace(r"\left[", "[").replace(r"\right]", "]")
    return latex


def annotate_directory(
    image_dir: str,
    output_path: str,
    backend: str = "transformers",
    model_id: str = MODEL_ID,
    batch_size: int = 16,
) -> None:
    """Annotate all images in a directory and save results as JSONL."""
    image_dir = Path(image_dir)
    image_paths = sorted(
        str(p) for p in image_dir.glob("*.png")
    ) + sorted(
        str(p) for p in image_dir.glob("*.jpg")
    )

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_paths)} images to annotate")

    if backend == "vllm":
        annotator = QwenAnnotatorVLLM(model_id)
        all_results = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            latex_results = annotator.annotate_images(batch)
            for path, latex in zip(batch, latex_results):
                all_results.append({
                    "image": Path(path).name,
                    "latex": latex,
                    "category": "real_annotated",
                })
    else:
        annotator = QwenAnnotatorTransformers(model_id)
        all_results = []
        for path in tqdm(image_paths, desc="Annotating"):
            latex = annotator.annotate_single(path)
            all_results.append({
                "image": Path(path).name,
                "latex": latex,
                "category": "real_annotated",
            })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Annotated {len(all_results)} images -> {output_path}")


def validate_synthetic_data(
    labels_path: str,
    images_dir: str,
    output_path: str,
    backend: str = "transformers",
    sample_size: Optional[int] = None,
) -> dict:
    """Validate synthetic data by round-trip: image -> Qwen prediction -> compare with label."""
    import random

    labels = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            labels.append(json.loads(line))

    if sample_size and sample_size < len(labels):
        labels = random.sample(labels, sample_size)

    image_paths = [str(Path(images_dir) / item["image"]) for item in labels]
    expected = [item["latex"] for item in labels]

    existing = [(p, e) for p, e in zip(image_paths, expected) if Path(p).exists()]
    if not existing:
        print("No valid image paths found")
        return {"total": 0, "valid": 0}

    image_paths, expected = zip(*existing)
    image_paths, expected = list(image_paths), list(expected)

    if backend == "vllm":
        annotator = QwenAnnotatorVLLM()
        predicted = annotator.annotate_images(image_paths)
    else:
        annotator = QwenAnnotatorTransformers()
        predicted = annotator.annotate_images(image_paths)

    valid_count = 0
    results = []
    for img, pred, exp in zip(image_paths, predicted, expected):
        is_valid = _normalize_latex(pred) == _normalize_latex(exp)
        if is_valid:
            valid_count += 1
        results.append({
            "image": Path(img).name,
            "expected": exp,
            "predicted": pred,
            "valid": is_valid,
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "total": len(results),
        "valid": valid_count,
        "accuracy": valid_count / len(results) if results else 0,
    }
    print(f"Validation: {stats['valid']}/{stats['total']} ({stats['accuracy']:.2%}) match")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL formula annotator")
    sub = parser.add_subparsers(dest="command")

    ann = sub.add_parser("annotate", help="Annotate formula images")
    ann.add_argument("--images", required=True, help="Directory of formula images")
    ann.add_argument("--output", required=True, help="Output JSONL path")
    ann.add_argument("--backend", choices=["vllm", "transformers"], default="transformers")
    ann.add_argument("--model", default=MODEL_ID)
    ann.add_argument("--batch-size", type=int, default=16)

    val = sub.add_parser("validate", help="Validate synthetic data quality")
    val.add_argument("--labels", required=True, help="Labels JSONL path")
    val.add_argument("--images", required=True, help="Images directory")
    val.add_argument("--output", required=True, help="Validation results output")
    val.add_argument("--backend", choices=["vllm", "transformers"], default="transformers")
    val.add_argument("--sample-size", type=int, default=None)

    args = parser.parse_args()

    if args.command == "annotate":
        annotate_directory(
            args.images, args.output,
            backend=args.backend,
            model_id=args.model,
            batch_size=args.batch_size,
        )
    elif args.command == "validate":
        validate_synthetic_data(
            args.labels, args.images, args.output,
            backend=args.backend,
            sample_size=args.sample_size,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
