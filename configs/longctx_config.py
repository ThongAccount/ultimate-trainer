"""1B-parameter model config for 1M-context stress test.

Uses 2B4T spec ratios (GQA 4:1, ReLU², subln, absmax activations).
Scaled to ~900M params for practical GPU training."""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class ModelConfig1B:
    # Architecture (~900M params)
    vocab_size: int = 128256
    hidden_dim: int = 1536
    intermediate_dim: int = 4096  # 2.67× hidden (ReLU² ratio, smaller than SwiGLU 8/3)
    num_layers: int = 24
    num_attention_heads: int = 12
    num_kv_heads: int = 3  # GQA 4:1 (same ratio as 2B4T 20:5)
    head_dim: int = 128
    max_seq_len: int = 4096  # initial, extended in stages
    rope_theta: float = 10_000.0
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    norm_eps: float = 1e-5

    # BitNet 2B4T
    use_bitlinear: bool = True
    activation_bits: int = 8
    full_precision_embeddings: bool = True

    # SubQSA (NSA-style)
    use_subqsa: bool = True
    cmp_block: int = 32
    cmp_stride: int = 16
    slc_block: int = 64
    slc_topk: int = 16
    win_size: int = 512

    # Compute / memory
    use_checkpoint: bool = True  # activation checkpointing saves memory for long context

    def __post_init__(self):
        assert self.hidden_dim % self.num_attention_heads == 0
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads


@dataclass
class TrainingConfig1M:
    dataset_name: str = "HuggingFaceFW/fineweb"
    max_seq_len: int = 4096
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 100_000
    warmup_steps: int = 2000
    max_grad_norm: float = 1.0

    # Two-stage LR (2B4T spec)
    learning_rate: float = 1.5e-3  # higher LR for 1-bit
    min_lr: float = 3e-4
    weight_decay: float = 0.1

    # Stage 2 (cooldown)
    cooldown_lr: float = 3e-4
    cooldown_wd: float = 0.0
    cooldown_start_step: int = 70_000

    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 2000
    output_dir: str = "checkpoints/1B-stress-test"
    run_name: str = "1B-subqsa-1M"

    distributed: bool = True
    dtype: str = "bfloat16"

    # Staged context extension: (seq_len, steps, rope_base)
    context_stages: tuple = field(
        default_factory=lambda: (
            (4096, 30_000, 10_000),  # P1: base context
            (32_768, 15_000, 80_000),  # P2: extend 8×, RoPE base ×8
            (131_072, 8_000, 320_000),  # P4: 128K, RoPE base ×32
            (262_144, 5_000, 640_000),  # P5: 256K, sequence parallelism
            (524_288, 3_000, 1_280_000),  # P6: 512K
            (1_048_576, 3_000, 2_560_000),  # P7: 1M target
        )
    )

    def get_torch_dtype(self):
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype]


def count_params(cfg: ModelConfig1B) -> dict:
    """Estimate parameter count for the config."""
    V = cfg.vocab_size
    H = cfg.hidden_dim
    I = cfg.intermediate_dim
    L = cfg.num_layers
    Q = cfg.num_attention_heads
    K = cfg.num_kv_heads
    D = cfg.head_dim

    embed = V * H
    per_layer = (
        H * (Q + K + K) * D  # QKV projections
        + Q * D * H  # O projection
        + H * I * 3  # gate, up, down
    )
    total = embed + L * per_layer
    return {
        "embedding_M": embed / 1e6,
        "per_layer_M": per_layer / 1e6,
        "total_M": total / 1e6,
        "total_B": total / 1e9,
        "effective_ternary_GB": total * 1.58 / 8 / 1e9,
    }


if __name__ == "__main__":
    cfg = ModelConfig1B()
    stats = count_params(cfg)
    print(f"ModelConfig 1B:")
    print(f"  Layers:          {cfg.num_layers}")
    print(f"  Hidden:          {cfg.hidden_dim}")
    print(f"  Heads Q/KV:      {cfg.num_attention_heads}/{cfg.num_kv_heads}")
    print(f"  FFN intermediate: {cfg.intermediate_dim}")
    print(f"  Parameters:      {stats['total_M']:.0f}M ({stats['total_B']:.2f}B)")
    print(f"  Ternary storage: {stats['effective_ternary_GB']:.2f} GB")
    print(f"  Context stages:  {len(cfg.num_layers)} stages to 1M")
