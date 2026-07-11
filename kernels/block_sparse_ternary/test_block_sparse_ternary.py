"""Tests for block_sparse_ternary kernel."""

import torch
import pytest
from kernels.block_sparse_ternary.block_sparse_ternary import (
    block_sparse_ternary_matmul, _block_sparse_ternary_eager, compute_block_mask
)


def test_dense_vs_ternary():
    """When all blocks active, output shape matches dense ternary matmul."""
    M, N, K = 128, 64, 64
    x = torch.randn(M, K)
    weight = torch.randn(N, K)
    gamma = weight.abs().mean() + 1e-5

    num_n_tiles = (N + 63) // 64
    num_k_tiles = (K + 63) // 64
    num_ints = max(1, (num_n_tiles * num_k_tiles + 63) // 64)
    block_mask = torch.full((num_ints,), ~0, dtype=torch.int64)

    y = block_sparse_ternary_matmul(x, weight, gamma, block_mask)
    assert y.shape == (M, N)
    assert not torch.isnan(y).any()


def test_sparse_mask():
    """Block-masked tiles are zero, active tiles are non-zero."""
    M, N, K = 128, 128, 64
    x = torch.randn(M, K)
    weight = torch.randn(N, K)
    gamma = weight.abs().mean() + 1e-5

    BN = 64
    num_n_tiles = (N + BN - 1) // BN
    num_k_tiles = (K + 63) // 64
    num_ints = max(1, (num_n_tiles * num_k_tiles + 63) // 64)

    block_mask = torch.full((num_ints,), ~0, dtype=torch.int64)
    # Mask out the second output tile
    for tk in range(num_k_tiles):
        bit_pos = 1 * num_k_tiles + tk
        block_mask[bit_pos // 64] &= ~(1 << (bit_pos % 64))

    y = block_sparse_ternary_matmul(x, weight, gamma, block_mask)
    assert torch.all(y[:, BN:2*BN] == 0), "Masked tile should be zero"
    assert not torch.all(y[:, :BN] == 0), "Active tile should be non-zero"


def test_compute_block_mask():
    """Verify bitmask from top_idx."""
    n_sel, topk = 8, 3
    top_idx = torch.tensor([[[1, 3, 5]]])
    num_k_tiles = 4
    num_n_tiles = n_sel
    mask = compute_block_mask(top_idx, n_sel, 64, num_n_tiles, num_k_tiles)
    for tn in [1, 3, 5]:
        for tk in range(num_k_tiles):
            bit_pos = tn * num_k_tiles + tk
            assert mask[bit_pos // 64] & (1 << (bit_pos % 64)), f"Bit {bit_pos} should be set"


def test_empty_mask():
    """All-zero mask => all-zero output."""
    M, N, K = 32, 32, 32
    x = torch.randn(M, K)
    weight = torch.randn(N, K)
    gamma = weight.abs().mean() + 1e-5

    num_n_tiles = (N + 63) // 64
    num_k_tiles = (K + 63) // 64
    num_ints = max(1, (num_n_tiles * num_k_tiles + 63) // 64)
    block_mask = torch.zeros(num_ints, dtype=torch.int64)

    y = block_sparse_ternary_matmul(x, weight, gamma, block_mask)
    assert torch.all(y == 0), "All-zero mask should produce zero output"


if __name__ == "__main__":
    test_dense_vs_ternary()
    test_sparse_mask()
    test_compute_block_mask()
    test_empty_mask()
    print("All block_sparse_ternary tests passed!")
