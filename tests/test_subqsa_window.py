"""Tests for sliding_window_attention.

Run directly with:
    python tests/test_subqsa_window.py
"""

import math
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/debian/ultimate-ai-model")

from ultimate_trainer.subqsa import sliding_window_attention


SHAPE_CASES = [
    # (B, H, T, D, win_size)
    (2, 4, 128, 128, 32),
    (1, 2, 8, 16, 8),
    (2, 3, 10, 16, 4),
    (1, 1, 4, 8, 16),
]


def _reference_sliced_attention(q, k, v, win_size):
    """Reference that explicitly slices KV and applies the same (T, w) mask."""
    B, H, T, D = q.shape
    w = min(win_size, T)
    k_win = k[..., -w:, :]
    v_win = v[..., -w:, :]

    scores = torch.einsum("bhtd,bhwd->bhtw", q, k_win) / math.sqrt(D)
    t_idx = torch.arange(T, device=q.device).unsqueeze(1)
    j_idx = torch.arange(w, device=q.device).unsqueeze(0)
    valid = (T - w + j_idx) <= t_idx

    scores = scores.masked_fill(~valid.view(1, 1, T, w), -1e9)
    attn = F.softmax(scores, dim=-1)
    # Rows with no valid key would be uniform; force them to zero to match the
    # zeroing done by ``sliding_window_attention``.
    has_valid = valid.any(dim=-1).view(1, 1, T, 1)
    attn = torch.where(has_valid, attn, torch.zeros_like(attn))

    return torch.einsum("bhtw,bhwd->bhtd", attn, v_win)


def test_sliding_window_output_shape():
    for B, H, T, D, win_size in SHAPE_CASES:
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        out = sliding_window_attention(q, k, v, win_size)
        assert out.shape == (B, H, T, D), f"shape mismatch for {(B, H, T, D, win_size)}"


def test_sliding_window_matches_sliced_reference():
    for B, H, T, D, win_size in SHAPE_CASES:
        torch.manual_seed(42)
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        out = sliding_window_attention(q, k, v, win_size)
        expected = _reference_sliced_attention(q, k, v, win_size)
        assert torch.allclose(out, expected, atol=1e-4), (
            f"mismatch for {(B, H, T, D, win_size)}"
        )


def test_sliding_window_is_causal_and_limited_to_window():
    """With uniform attention over valid keys, the output at position t is the
    average of the original positions in the window that are <= t.
    """
    B, H, T, D = 2, 3, 16, 8
    win_size = 7
    torch.manual_seed(0)
    q = torch.ones(B, H, T, D)
    k = torch.ones(B, H, T, D)
    # Each position's value is its original index.
    pos_vals = torch.arange(T, dtype=torch.float32).view(1, 1, T, 1)
    v = pos_vals.expand(B, H, T, D).contiguous()

    out = sliding_window_attention(q, k, v, win_size)
    assert out.shape == (B, H, T, D)

    w_eff = min(win_size, T)
    start = T - w_eff
    for t in range(T):
        if t < start:
            expected = 0.0
        else:
            valid_positions = torch.arange(start, t + 1, dtype=torch.float32)
            expected = valid_positions.mean().item()
        assert torch.allclose(
            out[..., t, :].mean(),
            torch.tensor(expected),
            atol=1e-4,
        ), f"causal average mismatch at t={t}"


if __name__ == "__main__":
    test_sliding_window_output_shape()
    print("test_sliding_window_output_shape passed")
    test_sliding_window_matches_sliced_reference()
    print("test_sliding_window_matches_sliced_reference passed")
    test_sliding_window_is_causal_and_limited_to_window()
    print("test_sliding_window_is_causal_and_limited_to_window passed")
    print("All tests passed")
