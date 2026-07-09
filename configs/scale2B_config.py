"""
~2B Parameter Model Scaling Config — Ultimate (BitLinear + SubQSA).

Precisely targets 2.09B parameters using the 2B4T spec ratios:
  GQA 4:1, ReLU² FFN, BitLinear ternary weights, SubQSA 3-branch attention.

Usage:
    from configs.scale2B_config import ModelConfig2B, count_params
    cfg = ModelConfig2B()
    stats = count_params(cfg)
    print(f"Parameters: {stats['total_B']:.2f}B")
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


# ── 2B4T Sizing Rules ─────────────────────────────────────────────
#
# hidden_dim     = 2560         (base width)
# num_layers     = 24           (depth for ~2B)
# num_heads      = 20           (Q heads)
# num_kv_heads   = 5            (GQA 4:1, same ratio as original 2B4T 20:5)
# head_dim       = 128
# intermediate   = 6912         (ReLU² ratio ~2.7× hidden, smaller than SwiGLU 8/3)
# vocab_size     = 128256       (Gemma 2 tokenizer vocab)
#
# Parameter breakdown (tied embeddings):
#   Embedding:       128256 × 2560 = 328.3M
#   Per layer:       73.5M  (attention 18.0M + compression 2.2M + gate 0.2M + FFN 53.1M)
#   24 layers:       24 × 73.5M = 1,762.8M
#   Final norm:      2,560
#   Total:           2,093.9M ≈ 2.09B


@dataclass
class ModelConfig2B:
    """~2B parameter Ultimate model configuration."""

    # ── Architecture ──
    vocab_size: int = 128256
    hidden_dim: int = 2560
    intermediate_dim: int = 6912       # ReLU² ratio: ~2.7× hidden
    num_layers: int = 24               # depth tuned for ~2B
    num_attention_heads: int = 20
    num_kv_heads: int = 5              # GQA 4:1
    head_dim: int = 128
    max_seq_len: int = 4096            # initial, extendable via staged context
    rope_theta: float = 10_000.0
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    norm_eps: float = 1e-5

    # ── BitNet 2B4T ──
    use_bitlinear: bool = True
    activation_bits: int = 8
    full_precision_embeddings: bool = True

    # ── SubQSA (NSA-style 3-branch sparse attention) ──
    use_subqsa: bool = True
    cmp_block: int = 32
    cmp_stride: int = 16
    slc_block: int = 64
    slc_topk: int = 16
    win_size: int = 512

    # ── Compute / memory ──
    use_checkpoint: bool = False        # enable for long-context training
    use_activation_warmup: bool = True  # gradual quant ramp over 5K steps

    def __post_init__(self):
        assert self.hidden_dim % self.num_attention_heads == 0
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads


@dataclass
class TrainingConfig2B:
    """Training configuration for the ~2B model."""

    # ── Data ──
    dataset_name: str = "HuggingFaceFW/fineweb"
    max_seq_len: int = 4096
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 8   # effective batch = 16
    max_steps: int = 50_000
    warmup_steps: int = 2000
    max_grad_norm: float = 1.0

    # ── Optimizer (AdamW, 2B4T-style) ──
    learning_rate: float = 4e-4
    min_lr: float = 4e-5
    lr_schedule: str = "cosine"
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # ── Logging & saving ──
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 2000
    output_dir: str = "checkpoints/ultimate-2B"
    run_name: str = "ultimate-2B-run1"

    # ── Distributed ──
    distributed: bool = True
    dtype: str = "bfloat16"

    # ── Context extension stages: (max_seq_len, steps, rope_base) ──
    context_stages: tuple = field(
        default_factory=lambda: (
            (4096, 30_000, 10_000),       # P1: base
            (8192, 10_000, 20_000),        # P2: 2×
            (16384, 5_000, 40_000),        # P3: 4×
            (32768, 3_000, 80_000),        # P4: 8×
        )
    )

    def get_torch_dtype(self):
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype]


# ── Parameter Count Utility ─────────────────────────────────────────


def count_params(cfg: ModelConfig2B) -> dict:
    """Estimate total parameter count for the config.

    Includes BitLinear (ternary), SubQSA (compression/gate MLPs),
    RMSNorm weights, and tied embedding/LM head.

    Returns:
        dict with keys: embedding_M, per_layer_M, total_M, total_B,
        ternary_storage_GB (effective at 1.58 bits/param)
    """
    V = cfg.vocab_size
    H = cfg.hidden_dim
    I = cfg.intermediate_dim
    L = cfg.num_layers
    Q = cfg.num_attention_heads
    K = cfg.num_kv_heads
    D = cfg.head_dim

    # Embedding (tied with LM head, counted once)
    embed = V * H

    # Per-layer breakdown
    # Attention projections (BitLinear) + routing_k_proj (nn.Linear)
    attn_Q = H * Q * D          # Q projection
    attn_K = H * K * D          # K projection
    attn_V = H * K * D          # V projection
    attn_O = Q * D * H          # O projection
    routing_k = H * K * D       # FP routing projection
    attn_total = attn_Q + attn_K + attn_V + attn_O + routing_k

    # Compression branch MLPs
    phi_k0 = (D * 32) * (D * 2)   # φ_k layer 0
    phi_k2 = (D * 2) * D          # φ_k layer 2
    phi_v0 = (D * 32) * (D * 2)   # φ_v layer 0
    phi_v2 = (D * 2) * D          # φ_v layer 2
    compression = phi_k0 + phi_k2 + phi_v0 + phi_v2

    # Gate MLP
    gate_in = H * 64
    gate_out = 64 * (3 * Q)
    gate_bias = 3 * Q
    gate_total = gate_in + gate_out + gate_bias

    # RMSNorm weights
    norms = (
        H                    # attn_norm
        + H                  # ffn_norm
        + I                  # ffn_out_norm
        + (K * D)            # out_norm (in SubQSAAttention)
    )

    # ReLU² FFN projections (BitLinear)
    ffn_gate = H * I
    ffn_up = H * I
    ffn_down = I * H
    ffn_total = ffn_gate + ffn_up + ffn_down

    per_layer = attn_total + compression + gate_total + norms + ffn_total
    final_norm = H

    total = embed + per_layer * L + final_norm

    return {
        "embedding_M": embed / 1e6,
        "per_layer_M": per_layer / 1e6,
        "attn_per_layer_M": attn_total / 1e6,
        "compression_per_layer_M": compression / 1e6,
        "gate_per_layer_M": gate_total / 1e6,
        "ffn_per_layer_M": ffn_total / 1e6,
        "norms_per_layer_K": norms / 1e3,
        "total_M": total / 1e6,
        "total_B": total / 1e9,
        "ternary_storage_GB": total * 1.58 / 8 / 1e9,
    }


def print_config(cfg: ModelConfig2B = None):
    """Print the config and parameter breakdown."""
    if cfg is None:
        cfg = ModelConfig2B()
    stats = count_params(cfg)

    print("╔══════════════════════════════════════════════════╗")
    print("║   ~2B Ultimate Model — Scaling Config           ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"\nArchitecture:")
    print(f"  Layers:               {cfg.num_layers}")
    print(f"  Hidden dim:           {cfg.hidden_dim}")
    print(f"  Intermediate dim:     {cfg.intermediate_dim}")
    print(f"  Heads (Q/KV):         {cfg.num_attention_heads}/{cfg.num_kv_heads}")
    print(f"  Head dim:             {cfg.head_dim}")
    print(f"  Vocab size:           {cfg.vocab_size:,}")
    print(f"  Max seq len:          {cfg.max_seq_len}")
    print(f"\nParameter Breakdown:")
    print(f"  Embedding (tied):     {stats['embedding_M']:.1f}M")
    print(f"  Per layer:            {stats['per_layer_M']:.1f}M")
    print(f"    ├─ Attention:       {stats['attn_per_layer_M']:.1f}M")
    print(f"    ├─ Compression:     {stats['compression_per_layer_M']:.1f}M")
    print(f"    ├─ Gate MLP:        {stats['gate_per_layer_M']:.2f}M")
    print(f"    ├─ Norms:           {stats['norms_per_layer_K']:.0f}K")
    print(f"    └─ FFN (ReLU²):     {stats['ffn_per_layer_M']:.1f}M")
    print(f"  Total ({cfg.num_layers} layers):    {stats['total_B']:.3f}B")
    print(f"\nStorage:")
    print(f"  Ternary effective:    {stats['ternary_storage_GB']:.2f} GB")
    print(f"  FP16 master weights:  {stats['total_M'] * 2 / 1e9:.2f} GB")
    print(f"\nSubQSA (NSA-style):")
    print(f"  Compression block:    {cfg.cmp_block}")
    print(f"  Compression stride:   {cfg.cmp_stride}")
    print(f"  Selection block:      {cfg.slc_block}")
    print(f"  Selection top-k:      {cfg.slc_topk}")
    print(f"  Sliding window:       {cfg.win_size}")
    print(f"\nTraining:")
    print(f"  Use checkpoint:       {cfg.use_checkpoint}")
    print(f"  Quant warmup:         {cfg.use_activation_warmup}")
    return stats


if __name__ == "__main__":
    stats = print_config()
    print(f"\nTo use in training:\n"
          f"  from configs.scale2B_config import ModelConfig2B, TrainingConfig2B\n"
          f"  mc = ModelConfig2B()\n"
          f"  tc = TrainingConfig2B()")
