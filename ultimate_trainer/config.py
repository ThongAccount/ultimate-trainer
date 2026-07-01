"""
Ultimate Trainer config — BitNet b1.58 2B4T spec + NSA SubQSA.
Anchored to 2B params for validated scaling.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


# ── 2B4T model sizing ──────────────────────────────────────────────


@dataclass
class UltimateModelConfig:
    vocab_size: int = 128256
    hidden_dim: int = 2560
    intermediate_dim: int = 6912
    num_layers: int = 30
    num_attention_heads: int = 20
    num_kv_heads: int = 5
    head_dim: int = 128
    max_seq_len: int = 4096
    rope_theta: float = 10_000.0
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0

    # ── BitNet 2B4T ──
    use_bitlinear: bool = True
    activation_bits: int = 8
    full_precision_embeddings: bool = True
    norm_eps: float = 1e-5

    # ── SubQSA (NSA) ──
    use_subqsa: bool = True
    cmp_block: int = 32
    cmp_stride: int = 16
    slc_block: int = 64
    slc_topk: int = 16
    win_size: int = 512

    def __post_init__(self):
        assert self.hidden_dim % self.num_attention_heads == 0
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads


@dataclass
class UltimateTrainingConfig:
    dataset_path: str = "data/dummy"
    max_seq_len: int = 4096
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 1000
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    output_dir: str = "checkpoints/ultimate_trainer"
    run_name: str = "ultimate-run1"

    distributed: bool = False
    dtype: str = "float32"

    # Staged context extension: (max_seq_len, steps)
    context_stages: tuple = field(
        default_factory=lambda: ((4096, 200), (8192, 100), (32768, 50))
    )

    def get_torch_dtype(self):
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype]
