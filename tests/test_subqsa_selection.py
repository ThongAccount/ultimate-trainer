"""Tests for SelectionBranch top-k block selection.

Run directly with:
    python tests/test_subqsa_selection.py
"""

import math
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/debian/ultimate-ai-model")

from ultimate_trainer.subqsa import SelectionBranch


SHAPE_CASES = [
    # (B, H, T, D, l_prime, topk)
    (2, 4, 8, 128, 64, 4),
    (1, 2, 64, 32, 16, 4),
    (2, 4, 128, 64, 32, 8),
    (3, 2, 32, 16, 8, 4),
]


def _make_inputs(B, H, T, D, l_prime):
    n_sel = max(1, T // l_prime)
    k_len = max(T, n_sel * l_prime)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, k_len, D)
    v = torch.randn(B, H, k_len, D)
    n_cmp = 8
    p_cmp = torch.randn(B, H, T, n_cmp).softmax(dim=-1)
    return q, k, v, p_cmp, n_cmp


def test_selection_output_shape():
    for B, H, T, D, l_prime, topk in SHAPE_CASES:
        sb = SelectionBranch(block_size=l_prime, topk=topk)
        q, k, v, p_cmp, n_cmp = _make_inputs(B, H, T, D, l_prime)
        out, idx = sb(q, k, v, p_cmp, n_cmp)
        n_sel = max(1, T // l_prime)
        topk_actual = min(topk, n_sel)
        assert out.shape == (B, H, T, D), f"out shape mismatch for {(B,H,T,D,l_prime,topk)}"
        assert idx.shape == (B, H, T, topk_actual), f"idx shape mismatch for {(B,H,T,D,l_prime,topk)}"


def test_selection_topk_count_when_enough_blocks():
    for B, H, T, D, l_prime, topk in SHAPE_CASES:
        sb = SelectionBranch(block_size=l_prime, topk=topk)
        q, k, v, p_cmp, n_cmp = _make_inputs(B, H, T, D, l_prime)
        out, idx = sb(q, k, v, p_cmp, n_cmp)
        n_sel = max(1, T // l_prime)
        topk_actual = min(topk, n_sel)
        assert idx.shape[-1] == topk_actual, f"topk count mismatch for {(B,H,T,D,l_prime,topk)}"
        assert (idx >= 0).all(), f"negative index for {(B,H,T,D,l_prime,topk)}"
        assert (idx < n_sel).all(), f"out-of-range index for {(B,H,T,D,l_prime,topk)}"


def test_selection_matches_manual_topk_gather():
    """Check that the branch attends over the same keys as a manual top-k gather."""
    B, H, T, D = 1, 1, 16, 16
    l_prime, topk = 4, 3
    n_sel = T // l_prime  # 4
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    n_cmp = n_sel
    p_cmp = torch.randn(B, H, T, n_cmp).softmax(dim=-1)

    sb = SelectionBranch(block_size=l_prime, topk=topk)
    out, idx = sb(q, k, v, p_cmp, n_cmp)

    # Reproduce the gather and attention manually.
    k_blocks = k.reshape(B, H, n_sel, l_prime, D)
    v_blocks = v.reshape(B, H, n_sel, l_prime, D)
    b_idx = torch.arange(B).view(B, 1, 1, 1)
    h_idx = torch.arange(H).view(1, H, 1, 1)
    k_sel = k_blocks[b_idx, h_idx, idx].reshape(B, H, T, topk * l_prime, D)
    v_sel = v_blocks[b_idx, h_idx, idx].reshape(B, H, T, topk * l_prime, D)
    scores = torch.einsum("bhtd,bhtld->bhtl", q, k_sel) / math.sqrt(D)
    attn = F.softmax(scores, dim=-1)
    expected = torch.einsum("bhtl,bhtld->bhtd", attn, v_sel)
    assert torch.allclose(out, expected, atol=1e-5)


def test_selection_interpolates_compression_scores():
    """When n_cmp != n_sel the branch should still produce a valid top-k selection."""
    B, H, T, D = 2, 2, 32, 16
    l_prime, topk = 8, 4
    n_sel = T // l_prime  # 4
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    n_cmp = 12
    p_cmp = torch.randn(B, H, T, n_cmp).softmax(dim=-1)
    sb = SelectionBranch(block_size=l_prime, topk=topk)
    out, idx = sb(q, k, v, p_cmp, n_cmp)
    assert out.shape == (B, H, T, D)
    assert idx.shape == (B, H, T, topk)
    assert (idx >= 0).all() and (idx < n_sel).all()


if __name__ == "__main__":
    test_selection_output_shape()
    print("test_selection_output_shape passed")
    test_selection_topk_count_when_enough_blocks()
    print("test_selection_topk_count_when_enough_blocks passed")
    test_selection_matches_manual_topk_gather()
    print("test_selection_matches_manual_topk_gather passed")
    test_selection_interpolates_compression_scores()
    print("test_selection_interpolates_compression_scores passed")
    print("All tests passed")
