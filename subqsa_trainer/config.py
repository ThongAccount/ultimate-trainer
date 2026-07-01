"""
Configuration for SubQSA (NSA-style Subquadratic Sparse Attention) trainer.
Based on "Native Sparse Attention" (Yuan et al., ACL 2025) and SubQ's SSA.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class SubQSAConfig:
    """SubQSA-specific hyperparameters (NSA three-branch design)."""

    # Compression branch
    cmp_block: int = 32  # block length l for compression
    cmp_stride: int = 16  # stride d between compression blocks

    # Selection branch
    slc_block: int = 64  # selection block size l'
    slc_topk: int = 16  # number of selected blocks n per query

    # Sliding window
    win_size: int = 512  # window size w for sliding branch

    # Gate: MLP produces 3 scalars per query
    gate_hidden: Optional[int] = None  # if None, uses head_dim

    def __post_init__(self):
        if self.gate_hidden is None:
            self.gate_hidden = 64


@dataclass
class ModelConfig:
    # Architecture
    vocab_size: int = 32_768
    hidden_dim: int = 1024  # smaller for CPU testing
    intermediate_dim: int = 2816  # 8/3 * hidden ≈
    num_layers: int = 6
    num_attention_heads: int = 8
    num_kv_heads: Optional[int] = None
    head_dim: int = 128
    max_seq_len: int = 4096
    rope_theta: float = 10_000.0
    rope_scaling: Optional[dict] = None
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0

    # SubQSA
    subqsa: SubQSAConfig = field(default_factory=SubQSAConfig)
    use_subqsa: bool = True

    # Normalization
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads
        assert self.hidden_dim % self.num_attention_heads == 0
        self.head_dim = self.head_dim or (self.hidden_dim // self.num_attention_heads)


@dataclass
class TrainingConfig:
    dataset_path: str = "data/dummy"
    max_seq_len: int = 4096
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    max_steps: int = 1000
    warmup_steps: int = 50
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
    output_dir: str = "checkpoints/subqsa_trainer"
    run_name: str = "subqsa-run1"

    distributed: bool = False
    dtype: str = "float32"

    # Staged context extension
    context_stages: tuple = (4096, 8192, 16384, 32768)
    stage_steps: tuple = (500, 300, 200, 100)

    def get_torch_dtype(self) -> torch.dtype:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype]
