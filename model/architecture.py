"""
TeXer model architecture: lightweight Swin Transformer encoder + Transformer decoder.

Target: ~20M parameters total.
- Encoder (~12M): Swin Transformer Tiny variant with reduced dimensions
- Decoder (~8M): 4-layer Transformer decoder with cross-attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from model.config import EncoderConfig, DecoderConfig, TexerConfig


# ---------------------------------------------------------------------------
# Swin Transformer Encoder (lightweight)
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, C, H/P, W/P)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.norm(x)
        return x, H, W


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, attn_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinBlock(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, window_size: int = 7,
        shift_size: int = 0, mlp_ratio: float = 4.0,
        drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._compute_mask(Hp, Wp, x.device)
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def _compute_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


class PatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        pad_r = W % 2
        pad_b = H % 2
        if pad_r or pad_b:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            H += pad_b
            W += pad_r

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)

        H, W = H // 2, W // 2
        x = x.view(B, H * W, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H, W


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class SwinEncoder(nn.Module):
    """Lightweight Swin Transformer encoder (~12M params with default config)."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.patch_embed = PatchEmbedding(config.in_channels, config.embed_dim, config.patch_size)

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        self.layers = nn.ModuleList()
        dim = config.embed_dim
        dp_idx = 0
        for i, (depth, heads) in enumerate(zip(config.depths, config.num_heads)):
            blocks = nn.ModuleList()
            for j in range(depth):
                blocks.append(SwinBlock(
                    dim=dim,
                    num_heads=heads,
                    window_size=config.window_size,
                    shift_size=0 if j % 2 == 0 else config.window_size // 2,
                    mlp_ratio=config.mlp_ratio,
                    drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    drop_path=dpr[dp_idx],
                ))
                dp_idx += 1
            self.layers.append(blocks)

            if i < len(config.depths) - 1:
                self.layers.append(nn.ModuleList([PatchMerging(dim)]))
                dim *= 2

        self.norm = nn.LayerNorm(dim)
        self.output_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, H, W = self.patch_embed(x)

        for layer in self.layers:
            if isinstance(layer[0], PatchMerging):
                x, H, W = layer[0](x, H, W)
            else:
                for block in layer:
                    x = block(x, H, W)

        x = self.norm(x)
        return x  # (B, seq_len, dim)


# ---------------------------------------------------------------------------
# Transformer Decoder
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        alloc_len = max(max_len * 2, 1024)
        pe = torch.zeros(alloc_len, d_model)
        position = torch.arange(0, alloc_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class OnnxFriendlyMultiHeadAttention(nn.Module):
    """Multi-head attention that avoids problematic reshapes during ONNX trace."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = query.shape
        _, S, _ = key.shape

        q = self.q_proj(query).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


class OnnxFriendlyDecoderLayer(nn.Module):
    """Single decoder layer with self-attention + cross-attention, ONNX-safe."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = OnnxFriendlyMultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = OnnxFriendlyMultiHeadAttention(d_model, nhead, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.norm1(tgt)
        x = tgt + self.dropout(self.self_attn(x, x, x, attn_mask=tgt_mask))
        res = x
        x = self.norm2(x)
        x = res + self.dropout(self.cross_attn(x, memory, memory))
        x = x + self.ff(self.norm3(x))
        return x


class TeXerDecoder(nn.Module):
    """Transformer decoder with cross-attention to encoder features (~8M params).

    Uses custom ONNX-friendly attention layers instead of nn.TransformerDecoder
    to ensure dynamic sequence lengths work correctly after ONNX export.
    """

    def __init__(self, config: DecoderConfig, encoder_dim: int):
        super().__init__()
        self.d_model = config.d_model
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)

        self.encoder_proj = nn.Linear(encoder_dim, config.d_model)

        self.layers = nn.ModuleList([
            OnnxFriendlyDecoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)

        self.output_proj = nn.Linear(config.d_model, config.vocab_size)
        self.max_seq_len = config.max_seq_len

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.encoder_proj(memory)

        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(tgt.size(1), tgt.device)

        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask)

        x = self.final_norm(x)
        logits = self.output_proj(x)
        return logits

    @staticmethod
    def _generate_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class TeXerModel(nn.Module):
    """
    Complete TeXer model: Swin encoder + Transformer decoder.

    ~20M parameters with default configuration.
    """

    def __init__(self, config: TexerConfig):
        super().__init__()
        self.config = config
        self.encoder = SwinEncoder(config.encoder)
        self.decoder = TeXerDecoder(config.decoder, self.encoder.output_dim)

    def forward(
        self,
        images: torch.Tensor,
        tgt: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W) input images
            tgt: (B, T) target token IDs (teacher forcing)
            tgt_key_padding_mask: (B, T) True where padded
        Returns:
            logits: (B, T, vocab_size)
        """
        memory = self.encoder(images)
        logits = self.decoder(tgt, memory, tgt_key_padding_mask=tgt_key_padding_mask)
        return logits

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 512,
        temperature: float = 1.0,
    ) -> list[list[int]]:
        """Autoregressive greedy decoding."""
        self.eval()
        B = images.size(0)
        memory = self.encoder(images)

        sequences = torch.full((B, 1), bos_id, dtype=torch.long, device=images.device)

        for _ in range(max_len - 1):
            logits = self.decoder(sequences, memory)
            next_logits = logits[:, -1, :] / temperature
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            sequences = torch.cat([sequences, next_token], dim=1)

            if (next_token == eos_id).all():
                break

        result = []
        for seq in sequences:
            tokens = seq.tolist()
            if eos_id in tokens:
                tokens = tokens[:tokens.index(eos_id) + 1]
            result.append(tokens)
        return result

    def count_parameters(self) -> dict:
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        total = enc_params + dec_params
        return {
            "encoder": enc_params,
            "decoder": dec_params,
            "total": total,
            "encoder_mb": enc_params * 4 / 1024 / 1024,
            "decoder_mb": dec_params * 4 / 1024 / 1024,
            "total_mb": total * 4 / 1024 / 1024,
        }


def build_model(config: Optional[TexerConfig] = None) -> TeXerModel:
    if config is None:
        config = TexerConfig()
    return TeXerModel(config)


if __name__ == "__main__":
    config = TexerConfig()
    model = build_model(config)
    params = model.count_parameters()
    print("Model parameter counts:")
    for k, v in params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f} MB")
        else:
            print(f"  {k}: {v:,}")

    dummy_img = torch.randn(2, 1, 224, 224)
    dummy_tgt = torch.randint(0, 600, (2, 20))
    logits = model(dummy_img, dummy_tgt)
    print(f"\nInput image: {dummy_img.shape}")
    print(f"Input target: {dummy_tgt.shape}")
    print(f"Output logits: {logits.shape}")

    generated = model.generate(dummy_img, bos_id=1, eos_id=2, max_len=50)
    print(f"Generated sequences: {[len(s) for s in generated]}")
