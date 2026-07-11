"""Integration tests for CUDA kernel integration into SubQSAAttention."""

import pytest
import torch
import importlib


def _make_subqsa_attn(use_cuda_kernels=False):
    """Instantiate a small SubQSAAttention for shape checks."""
    from ultimate_trainer.subqsa import SubQSAAttention

    return SubQSAAttention(
        hidden_dim=256,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        max_seq_len=512,
        cmp_block=16,
        cmp_stride=8,
        slc_block=32,
        slc_topk=4,
        win_size=128,
        use_bitlinear=False,
        use_cuda_kernels=use_cuda_kernels,
    )


class TestForwardPyTorch:
    """SubQSAAttention(use_cuda_kernels=False) produces correct output shape."""

    def test_forward_pytorch(self):
        attn = _make_subqsa_attn(use_cuda_kernels=False)
        attn.eval()

        B, T, D = 2, 64, 256
        x = torch.randn(B, T, D)
        position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)

        with torch.no_grad():
            out = attn(x, position_ids)

        assert out.shape == (B, T, D), (
            f"Expected ({B}, {T}, {D}), got {out.shape}"
        )


class TestKernelImports:
    """All four CUDA kernel modules are importable and expose callable functions."""

    def test_compressed_attn_import(self):
        from kernels.compressed_attn.compressed_attn import compressed_attn_forward
        assert callable(compressed_attn_forward)

    def test_selective_attn_import(self):
        from kernels.selective_attn.selective_attn import selective_attn_forward
        assert callable(selective_attn_forward)

    def test_block_sparse_import(self):
        from kernels.block_sparse_ternary.block_sparse_ternary import block_sparse_ternary_matmul
        assert callable(block_sparse_ternary_matmul)

    def test_subqsa_combine_import(self):
        from kernels.subqsa_combine.subqsa_combine import subqsa_combine_forward
        assert callable(subqsa_combine_forward)
