"""
ONNX export and quantization for TeXer model.

Exports the encoder and decoder as separate ONNX models for efficient
deployment on browser (ONNX Runtime Web) and mobile (ONNX Runtime Mobile).

The encoder runs once per image, then the decoder runs autoregressively.
Splitting them avoids re-encoding the image at each decode step.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model.config import TexerConfig
from model.architecture import build_model, TeXerModel
from model.tokenizer import LaTeXTokenizer


class EncoderWrapper(nn.Module):
    """Wraps the encoder for ONNX export with a clean interface."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)


class DecoderWrapper(nn.Module):
    """Wraps the decoder for ONNX export with a clean interface."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.decoder(input_ids, encoder_output)
        return logits


def export_encoder(
    model: TeXerModel,
    output_path: str,
    image_size: int = 224,
    opset_version: int = 17,
) -> None:
    encoder = EncoderWrapper(model.encoder)
    encoder.eval()

    dummy_image = torch.randn(1, 1, image_size, image_size)

    torch.onnx.export(
        encoder,
        (dummy_image,),
        output_path,
        opset_version=opset_version,
        input_names=["image"],
        output_names=["encoder_output"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "encoder_output": {0: "batch_size"},
        },
    )
    print(f"Encoder exported to {output_path}")


def export_decoder(
    model: TeXerModel,
    output_path: str,
    max_seq_len: int = 512,
    opset_version: int = 17,
) -> None:
    decoder = DecoderWrapper(model.decoder)
    decoder.eval()

    encoder_dim = model.encoder.output_dim
    dummy_input_ids = torch.randint(0, 100, (1, 10))
    dummy_encoder_output = torch.randn(1, 49, encoder_dim)

    torch.onnx.export(
        decoder,
        (dummy_input_ids, dummy_encoder_output),
        output_path,
        opset_version=opset_version,
        input_names=["input_ids", "encoder_output"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "encoder_output": {0: "batch_size", 1: "enc_seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
    )
    print(f"Decoder exported to {output_path}")


def quantize_model(input_path: str, output_path: str, quantize_type: str = "dynamic") -> None:
    """Apply INT8 quantization to reduce model size."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        input_path,
        output_path,
        weight_type=QuantType.QUInt8,
    )

    original_size = Path(input_path).stat().st_size / 1024 / 1024
    quantized_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"Quantized: {original_size:.1f}MB -> {quantized_size:.1f}MB "
          f"({(1 - quantized_size/original_size)*100:.1f}% reduction)")


def verify_onnx(
    encoder_path: str,
    decoder_path: str,
    model: TeXerModel,
    image_size: int = 224,
) -> bool:
    """Verify ONNX model outputs match PyTorch model outputs."""
    import onnxruntime as ort

    model.eval()
    dummy_image = torch.randn(1, 1, image_size, image_size)
    dummy_input_ids = torch.randint(0, 100, (1, 10))

    with torch.no_grad():
        pt_encoder_out = model.encoder(dummy_image)
        pt_logits = model.decoder(dummy_input_ids, pt_encoder_out)

    enc_session = ort.InferenceSession(encoder_path)
    enc_result = enc_session.run(None, {"image": dummy_image.numpy()})
    onnx_encoder_out = enc_result[0]

    dec_session = ort.InferenceSession(decoder_path)
    dec_result = dec_session.run(None, {
        "input_ids": dummy_input_ids.numpy(),
        "encoder_output": onnx_encoder_out,
    })
    onnx_logits = dec_result[0]

    enc_close = np.allclose(pt_encoder_out.numpy(), onnx_encoder_out, atol=1e-4)
    dec_close = np.allclose(pt_logits.numpy(), onnx_logits, atol=1e-4)

    print(f"Encoder match: {enc_close}")
    print(f"Decoder match: {dec_close}")
    return enc_close and dec_close


def export_full(
    checkpoint_path: str,
    output_dir: str,
    config: TexerConfig = None,
    quantize: bool = True,
    verify: bool = True,
) -> None:
    """Full export pipeline: load checkpoint -> export ONNX -> quantize -> verify."""
    if config is None:
        config = TexerConfig()

    tokenizer = LaTeXTokenizer()
    if Path(config.data.vocab_path).exists():
        tokenizer = LaTeXTokenizer.load(config.data.vocab_path)
    config.decoder.vocab_size = tokenizer.vocab_size

    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    encoder_path = str(out / "encoder.onnx")
    decoder_path = str(out / "decoder.onnx")

    print("=== Exporting encoder ===")
    export_encoder(model, encoder_path, config.encoder.image_size)

    print("\n=== Exporting decoder ===")
    export_decoder(model, decoder_path, config.decoder.max_seq_len)

    if quantize:
        print("\n=== Quantizing models ===")
        enc_quant_path = str(out / "encoder_quantized.onnx")
        dec_quant_path = str(out / "decoder_quantized.onnx")
        quantize_model(encoder_path, enc_quant_path)
        quantize_model(decoder_path, dec_quant_path)

    if verify:
        print("\n=== Verifying ONNX models ===")
        verify_onnx(encoder_path, decoder_path, model, config.encoder.image_size)

    tokenizer.save(str(out / "vocab.json"))

    model_info = {
        "image_size": config.encoder.image_size,
        "max_seq_len": config.decoder.max_seq_len,
        "vocab_size": config.decoder.vocab_size,
        "in_channels": config.encoder.in_channels,
        "encoder_file": "encoder_quantized.onnx" if quantize else "encoder.onnx",
        "decoder_file": "decoder_quantized.onnx" if quantize else "decoder.onnx",
        "vocab_file": "vocab.json",
    }
    with open(out / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"\n=== Export complete -> {output_dir} ===")
    for p in sorted(out.glob("*")):
        size = p.stat().st_size / 1024 / 1024
        print(f"  {p.name}: {size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export TeXer model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    config = TexerConfig.load(args.config) if args.config else TexerConfig()

    export_full(
        args.checkpoint,
        args.output,
        config=config,
        quantize=not args.no_quantize,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
