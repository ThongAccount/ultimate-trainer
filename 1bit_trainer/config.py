"""
Configuration for native 1-bit (ternary) LLM training.
Based on BitNet b1.58: "The Era of 1-bit LLMs" (Ma et al., 2024).
All weights are ternary {-1, 0, +1} trained from scratch.
Activations are 8-bit signed integers.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
import torch


@dataclass
class ModelConfig:
    # Architecture
    vocab_size: int = 32768
    hidden_dim: int = 2048
    intermediate_dim: int = 5632  # SwiGLU: 8/3 * hidden_dim rounded
    num_layers: int = 24
    num_attention_heads: int = 16
    num_kv_heads: Optional[int] = None       # None = multi-head, int = GQA
    head_dim: int = 128
    max_seq_len: int = 4096
    rope_theta: float = 10_000.0
    rope_scaling: Optional[dict] = None
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0

    # 1-bit specific
    # BitLinear replaces ALL nn.Linear with ternary weights {-1, 0, +1}
    use_bitlinear: bool = True
    # Activation quantization: 8-bit signed per token
    activation_bits: int = 8
    # Whether to keep embedding/output layers in full precision
    # (paper keeps embeddings in FP16, only transformer layers are 1-bit)
    full_precision_embeddings: bool = True

    # Normalization
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads
        assert self.hidden_dim % self.num_attention_heads == 0
        self.head_dim = self.head_dim or (self.hidden_dim // self.num_attention_heads)


@dataclass
class TrainingConfig:
    # Dataset
    dataset_path: str = "data/redpajama"
    dataset_name: str = "togethercomputer/RedPajama-Data-1T-Sample"
    max_seq_len: int = 4096

    # Batch & sequence
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_steps: int = 100_000
    warmup_steps: int = 2000
    max_grad_norm: float = 1.0

    # Optimizer
    learning_rate: float = 4e-4
    min_lr: float = 4e-5
    lr_schedule: str = "cosine"               # cosine, linear, constant
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    # BitNet-specific: no gradient scaling for quantized weights.
    # Straight-through estimator used for weight quantization.
    ste_scale: float = 1.0                    # scale for STE backward

    # Activation quantization warmup
    # Paper: gradually enable activation quantization during training
    act_quant_warmup_steps: int = 5000

    # Logging & saving
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 2000
    output_dir: str = "checkpoints/1bit_trainer"
    run_name: str = "bitnet-b1.58-run1"

    # Distributed
    distributed: bool = True
    tensor_parallel: bool = False

    # Dtype
    dtype: str = "bfloat16"                   # bfloat16, float16, float32

    def get_torch_dtype(self) -> torch.dtype:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype]
