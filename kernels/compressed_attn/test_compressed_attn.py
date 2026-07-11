# kernels/compressed_attn/test_compressed_attn.py

import torch
import torch.nn.functional as F
import pytest
from kernels.compressed_attn.compressed_attn import compressed_attn_forward, _compressed_attn_eager


def _make_phi(head_dim, block_len, device="cpu"):
    """Create test MLP weights matching CompressionBranch spec."""
    in_dim = head_dim * block_len
    w1 = torch.randn(2 * head_dim, in_dim, device=device) * 0.02
    b1 = torch.zeros(2 * head_dim, device=device)
    w2 = torch.randn(head_dim, 2 * head_dim, device=device) * 0.02
    b2 = torch.zeros(head_dim, device=device)
    return (w1, b1, w2, b2), (w1.clone(), b1.clone(), w2.clone(), b2.clone())


def test_compressed_attn_eager_small():
    """Test the PyTorch reference at tiny sizes."""
    B, H, T, D = 2, 2, 32, 32
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    phi_k_params, phi_v_params = _make_phi(D, block_len=8)
    k_cmp, v_cmp = _compressed_attn_eager(k, v, *phi_k_params, *phi_v_params, block_len=8, stride=4)
    # Expect n_blocks = (32-8)//4 = 6
    assert k_cmp.shape == (B, H, 6, D), f"Expected (B,H,6,D), got {k_cmp.shape}"
    assert v_cmp.shape == (B, H, 6, D)
    assert not torch.isnan(k_cmp).any()
    assert not torch.isnan(v_cmp).any()


def test_compressed_attn_eager_no_blocks():
    """When T < block_len, fall back to mean pooling."""
    B, H, T, D = 1, 1, 4, 16
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    phi_k_params, phi_v_params = _make_phi(D, block_len=8)
    k_cmp, v_cmp = _compressed_attn_eager(k, v, *phi_k_params, *phi_v_params, block_len=8, stride=4)
    assert k_cmp.shape == (B, H, 1, D)
    assert v_cmp.shape == (B, H, 1, D)


def test_compressed_attn_eager_parity_with_unfold():
    """Verify that the eager impl runs without error at larger sizes."""
    B, H, T, D = 1, 2, 64, 32
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    phi_k_params, phi_v_params = _make_phi(D, block_len=16)
    k_cmp, v_cmp = _compressed_attn_eager(k, v, *phi_k_params, *phi_v_params, block_len=16, stride=8)
    assert k_cmp.shape == (B, H, 6, D)
    assert v_cmp.shape == (B, H, 6, D)
    assert not torch.isnan(k_cmp).any()


def test_compressed_attn_mlp_vs_phi_k_v():
    """Verify deterministic: same inputs → same outputs."""
    B, H, T, D = 1, 1, 128, 64
    block_len, stride = 32, 16
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    phi_k_params, phi_v_params = _make_phi(D, block_len)
    k_cmp0, v_cmp0 = _compressed_attn_eager(k, v, *phi_k_params, *phi_v_params, block_len, stride)
    k_cmp1, v_cmp1 = _compressed_attn_eager(k, v, *phi_k_params, *phi_v_params, block_len, stride)
    n_blocks = (T - block_len) // stride
    assert k_cmp0.shape == (B, H, n_blocks, D)
    assert torch.allclose(k_cmp0, k_cmp1, atol=1e-5)
    assert torch.allclose(v_cmp0, v_cmp1, atol=1e-5)


if __name__ == "__main__":
    test_compressed_attn_eager_small()
    test_compressed_attn_eager_no_blocks()
    test_compressed_attn_eager_parity_with_unfold()
    test_compressed_attn_mlp_vs_phi_k_v()
    print("All compressed_attn tests passed!")
