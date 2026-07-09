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
        assert out.shape == (B, H, T, D), (
            f"out shape mismatch for {(B, H, T, D, l_prime, topk)}"
        )
        assert idx.shape == (B, H, topk_actual), (
            f"idx shape mismatch for {(B, H, T, D, l_prime, topk)}"
        )


def test_selection_topk_count_when_enough_blocks():
    for B, H, T, D, l_prime, topk in SHAPE_CASES:
        sb = SelectionBranch(block_size=l_prime, topk=topk)
        q, k, v, p_cmp, n_cmp = _make_inputs(B, H, T, D, l_prime)
        out, idx = sb(q, k, v, p_cmp, n_cmp)
        n_sel = max(1, T // l_prime)
        topk_actual = min(topk, n_sel)
        assert idx.shape[-1] == topk_actual, (
            f"topk count mismatch for {(B, H, T, D, l_prime, topk)}"
        )
        assert (idx >= 0).all(), f"negative index for {(B, H, T, D, l_prime, topk)}"
        assert (idx < n_sel).all(), (
            f"out-of-range index for {(B, H, T, D, l_prime, topk)}"
        )


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

    # Reproduce the gather and attention manually using the same aggregated
    # indices (B, H, K) and the same gather+reshape+SDPA logic as the branch.
    k_blocks = k[:, :, :n_sel * l_prime, :].reshape(B, H, n_sel, l_prime, D)
    v_blocks = v[:, :, :n_sel * l_prime, :].reshape(B, H, n_sel, l_prime, D)

    bi = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, l_prime, D)
    k_sel = torch.gather(k_blocks, dim=2, index=bi).reshape(B, H, topk * l_prime, D)
    v_sel = torch.gather(v_blocks, dim=2, index=bi).reshape(B, H, topk * l_prime, D)

    # Reproduce the causal mask exactly as SelectionBranch does.
    offsets = torch.arange(l_prime, device=q.device)
    orig_k_pos = (idx * l_prime).unsqueeze(-1) + offsets  # (B, H, K, lp)
    orig_k_pos = orig_k_pos.reshape(B, H, topk * l_prime)  # (B, H, K*lp)

    q_pos = torch.arange(T, device=q.device).view(1, 1, T, 1)
    valid = orig_k_pos.unsqueeze(2) <= q_pos  # (B, H, T, K*lp)
    mask = torch.where(valid, 0.0, float("-inf"))

    scores = torch.einsum("bhtd,bhkd->bhtk", q, k_sel) / math.sqrt(D)
    scores = scores + mask
    attn = F.softmax(scores, dim=-1)
    expected = torch.einsum("bhtk,bhkd->bhtd", attn, v_sel)
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
    assert idx.shape == (B, H, topk)
    assert (idx >= 0).all() and (idx < n_sel).all()


def test_subqsa_trainer_score_and_select_uses_p_cmp_topk():
    """Regression: _score_and_select() must select blocks via p_cmp.topk() from full KV.

    Shapes: T=64, slc_block=32 → n_sel=2, slc_topk=4 → k_actual=2.
    n_cmp = 2 (matches compression for T=64, cmp_block=32, cmp_stride=16).
    p_cmp sets block 1 as high importance → top_idx must select [1, 0].
    k_sel/v_sel must be (B, H, k_actual * slc_block, D) = (1, 1, 64, 16).
    """
    from subqsa_trainer.subqsa import SubQSA

    torch.manual_seed(42)
    B, H, D = 1, 1, 16
    T = 64
    slc_block = 32
    slc_topk = 4
    n_sel = T // slc_block
    n_cmp = 2
    topk_actual = min(slc_topk, n_sel)

    subqsa = SubQSA(
        hidden_dim=64,
        num_heads=H,
        num_kv_heads=H,
        head_dim=D,
        slc_block=slc_block,
        slc_topk=slc_topk,
    )
    subqsa.eval()

    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    p_cmp_agg = torch.zeros(B, H, n_cmp)
    p_cmp_agg[..., 1] = 100.0

    k_sel, v_sel, top_idx = subqsa._score_and_select(q, k, v, p_cmp_agg, n_cmp)

    expected = torch.tensor([[1, 0]]).view(1, 1, -1).expand(B, H, -1)
    assert top_idx.shape == (B, H, topk_actual), (
        f"Expected (B,H,k) shape, got {top_idx.shape}"
    )
    assert (top_idx == expected).all(), f"Expected selection {expected}, got {top_idx}"
    assert k_sel.shape == (B, H, topk_actual * slc_block, D), (
        f"k_sel shape mismatch: {k_sel.shape}"
    )
    assert v_sel.shape == (B, H, topk_actual * slc_block, D), (
        f"v_sel shape mismatch: {v_sel.shape}"
    )


if __name__ == "__main__":
    test_selection_output_shape()
    print("test_selection_output_shape passed")
    test_selection_topk_count_when_enough_blocks()
    print("test_selection_topk_count_when_enough_blocks passed")
    test_selection_matches_manual_topk_gather()
    print("test_selection_matches_manual_topk_gather passed")
    test_selection_interpolates_compression_scores()
    print("test_selection_interpolates_compression_scores passed")
    test_subqsa_trainer_score_and_select_uses_p_cmp_topk()
    print("test_subqsa_trainer_score_and_select_uses_p_cmp_topk passed")
    print("All tests passed")
