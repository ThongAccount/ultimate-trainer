"""Tests for subqsa_combine: gate MLP -> blend -> RMSNorm -> O projection.

Tests run on CPU via the PyTorch eager fallback.
"""

import os
import sys
import math

import torch
import torch.nn.functional as F

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kernels.subqsa_combine.subqsa_combine import (
    subqsa_combine_forward,
    _subqsa_combine_eager,
)


def _make_inputs(B, T, H, D_head, D_out, device="cpu", seed=42):
    """Create synthetic inputs for the subqsa_combine kernel.

    Returns a dict with all inputs.
    """
    torch.manual_seed(seed)
    D = H * D_head
    inputs = {
        "x": torch.randn(B, T, D, device=device, dtype=torch.half),
        "o_cmp": torch.randn(B, H, T, D_head, device=device, dtype=torch.half),
        "o_slc": torch.randn(B, H, T, D_head, device=device, dtype=torch.half),
        "o_win": torch.randn(B, H, T, D_head, device=device, dtype=torch.half),
        "gate_w1": torch.randn(64, D, device=device, dtype=torch.half),
        "gate_w2": torch.randn(3 * H, 64, device=device, dtype=torch.half),
        "out_norm_weight": torch.randn(D, device=device, dtype=torch.half),
        "o_proj_weight": torch.randn(D_out, D, device=device, dtype=torch.float),
        "gamma": 0.1,
    }
    return inputs


def test_small():
    """Forward pass at tiny sizes — check shape and no NaN."""
    B, T, H, D_head, D_out = 1, 1, 1, 4, 4
    ins = _make_inputs(B, T, H, D_head, D_out)
    y = _subqsa_combine_eager(**ins)
    assert y.shape == (B, T, D_out), f"Expected ({B},{T},{D_out}), got {y.shape}"
    assert not torch.isnan(y).any(), "Output contains NaN"
    assert not torch.isinf(y).any(), "Output contains Inf"
    print(f"  [PASS] test_small: shape={y.shape}, no NaN/Inf")


def test_small_varied():
    """Forward at slightly varied sizes."""
    for B, T, H, D_head, D_out in [(1, 2, 2, 8, 16), (2, 3, 4, 16, 64)]:
        ins = _make_inputs(B, T, H, D_head, D_out)
        y = _subqsa_combine_eager(**ins)
        assert y.shape == (B, T, D_out), f"({B},{T},{H},{D_head},{D_out}): {y.shape}"
        assert not torch.isnan(y).any()
    print("  [PASS] test_small_varied: all shapes correct")


def test_gate_dominance():
    """Biasing one branch's gate weight changes the output toward that branch.

    Strategy: set gate_w2 so that branch 0 is strongly favoured (large positive
    pre-sigmoid logit). After sigmoid + normalize, g_cmp ≈ 1, g_slc ≈ 0, g_win ≈ 0.
    The output should then approx equal RMSNorm(o_cmp) projected by O.
    """
    B, T, H, D_head, D_out = 1, 1, 2, 8, 16
    ins = _make_inputs(B, T, H, D_head, D_out, seed=123)

    # Override gate_w2 so all branches except the first have large negative logits
    with torch.no_grad():
        # gate_w2 shape: (3*H, 64).  For each head set branch 0 weights to
        # large positive constant and branches 1,2 to large negative constant.
        for h in range(H):
            # Branch 0: positive bias
            ins["gate_w2"][h * 3 + 0, :] = 10.0
            # Branch 1, 2: negative bias
            ins["gate_w2"][h * 3 + 1, :] = -10.0
            ins["gate_w2"][h * 3 + 2, :] = -10.0

    y = _subqsa_combine_eager(**ins)

    # Compute reference: branch 0 (o_cmp) path
    B_, H_, T_, D_head_ = ins["o_cmp"].shape
    D_ = H_ * D_head_
    o_ref = ins["o_cmp"].transpose(1, 2).reshape(B_, T_, -1)
    rms_ref = o_ref.pow(2).mean(-1, keepdim=True).sqrt()
    o_ref = o_ref / (rms_ref + 1e-5) * ins["out_norm_weight"]
    w_q = torch.clamp(torch.round(ins["o_proj_weight"] / ins["gamma"]), -1, 1) * ins["gamma"]
    y_ref = F.linear(o_ref.float(), w_q).to(dtype=o_ref.dtype)

    diff = (y - y_ref).abs().max().item()
    assert diff < 0.5, f"Gate-dominance test: max diff={diff:.4f} (expected < 0.5)"
    print(f"  [PASS] test_gate_dominance: max diff={diff:.4f}")


def test_deterministic():
    """Same inputs => same outputs."""
    B, T, H, D_head, D_out = 2, 4, 2, 8, 16
    ins = _make_inputs(B, T, H, D_head, D_out, seed=456)
    y1 = _subqsa_combine_eager(**ins)
    y2 = _subqsa_combine_eager(**ins)
    diff = (y1 - y2).abs().max().item()
    assert diff == 0.0, f"Determinism violation: max diff={diff:.10f}"
    print(f"  [PASS] test_deterministic: max diff={diff:.2e}")


def test_gradient():
    """Backward pass works and gradients flow to all parameters."""
    B, T, H, D_head, D_out = 2, 3, 2, 8, 16
    ins = _make_inputs(B, T, H, D_head, D_out, seed=789)
    # Use large gamma so o_proj_weight / gamma stays inside [-1,1] (avoids
    # torch.clamp zeroing gradients in the ternary quantisation step).
    gamma = 5.0
    ins.pop("gamma")

    # Make all inputs require grad
    ins["x"].requires_grad_(True)
    ins["o_cmp"].requires_grad_(True)
    ins["o_slc"].requires_grad_(True)
    ins["o_win"].requires_grad_(True)
    ins["gate_w1"].requires_grad_(True)
    ins["gate_w2"].requires_grad_(True)
    ins["out_norm_weight"].requires_grad_(True)
    ins["o_proj_weight"].requires_grad_(True)

    y = _subqsa_combine_eager(gamma=gamma, **ins)
    loss = y.sum()
    loss.backward()

    # Check all gradients exist and are not zero
    param_names = ["x", "o_cmp", "o_slc", "o_win", "gate_w1", "gate_w2",
                   "out_norm_weight", "o_proj_weight"]
    for name in param_names:
        g = ins[name].grad
        assert g is not None, f"{name}.grad is None"
        assert g.abs().sum().item() > 0, f"{name}.grad is all zeros"
    print(f"  [PASS] test_gradient: gradients flow to all {len(param_names)} parameters")


if __name__ == "__main__":
    print("subqsa_combine tests (CPU — eager fallback)")
    print("=" * 40)
    test_small()
    test_small_varied()
    test_gate_dominance()
    test_deterministic()
    test_gradient()
    print("=" * 40)
    print("All tests passed!")
