# kernels/selective_attn/test_selective_attn.py
# CPU-based tests for the selective attention eager reference.
# Tests only the PyTorch fallback (no GPU required).

import torch
import torch.nn.functional as F
import pytest
from kernels.selective_attn.selective_attn import _selective_attn_eager, selective_attn_forward


def test_small():
    """Basic forward at tiny sizes."""
    B, H, T, D = 1, 2, 32, 16
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    scores_agg = torch.randn(B, H, T // 8)  # n_sel = 4
    out = _selective_attn_eager(q, k, v, scores_agg, topk=2, block_size=8)
    assert out.shape == (B, H, T, D), f"Expected (B,H,T,D), got {out.shape}"
    assert not torch.isnan(out).any()


def test_causal_mask():
    """First vs last token differ due to causal mask."""
    B, H, T, D = 1, 1, 16, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    # Uniform scores -- only causal mask differentiates
    scores_agg = torch.ones(B, H, T // 4)
    out = _selective_attn_eager(q, k, v, scores_agg, topk=4, block_size=4)

    out_first = out[0, 0, 0, :]
    out_last = out[0, 0, -1, :]
    assert not torch.allclose(out_first, out_last, atol=1e-4), \
        "First and last token outputs should differ (causal mask effect)"


def test_topk_less_than_nsel():
    """When topk > n_sel, select all available blocks (no crash)."""
    B, H, T, D = 1, 1, 8, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    scores_agg = torch.randn(B, H, 1)  # n_sel = 1
    out = _selective_attn_eager(q, k, v, scores_agg, topk=4, block_size=8)
    assert out.shape == (B, H, T, D)
    assert not torch.isnan(out).any()


def test_deterministic():
    """Same inputs produce the same outputs."""
    B, H, T, D = 1, 1, 16, 8
    torch.manual_seed(42)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    scores_agg = torch.randn(B, H, T // 8)  # n_sel = 2

    out0 = _selective_attn_eager(q, k, v, scores_agg, topk=2, block_size=8)
    out1 = _selective_attn_eager(q, k, v, scores_agg, topk=2, block_size=8)

    assert torch.allclose(out0, out1, atol=1e-5), \
        "Same inputs should produce same outputs"


def test_gradient():
    """Verify gradients flow through selective_attn."""
    B, H, T, D = 1, 1, 16, 8
    q = torch.randn(B, H, T, D, requires_grad=True)
    k = torch.randn(B, H, T, D, requires_grad=True)
    v = torch.randn(B, H, T, D, requires_grad=True)
    scores_agg = torch.randn(B, H, T // 4, requires_grad=True)

    out = _selective_attn_eager(q, k, v, scores_agg, topk=2, block_size=4)
    out.sum().backward()

    assert q.grad is not None, "q.grad is None"
    assert k.grad is not None, "k.grad is None"
    assert v.grad is not None, "v.grad is None"
    assert not torch.isnan(q.grad).any(), "NaN in q.grad"
    print(f"  selective_attn gradient: ✅")


if __name__ == "__main__":
    test_small()
    test_causal_mask()
    test_topk_less_than_nsel()
    test_deterministic()
    test_gradient()
    print("All selective_attn tests passed!")
