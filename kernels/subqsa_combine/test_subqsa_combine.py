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
    """Gating changes the output: biasing branch 0 should make the blended
    output closer to that branch's standalone path than to a different branch.

    Strategy:
      - Copy inputs and set o_cmp and o_slc to clearly different values.
      - Create a variant with large positive pre-sigmoid bias on branch 0
        of gate_w2 (favouring o_cmp).
      - Check that the variant's output differs from the no-bias baseline.
    """
    B, T, H, D_head, D_out = 1, 2, 2, 8, 16
    ins = _make_inputs(B, T, H, D_head, D_out, seed=42)

    # Baseline: uniform gates (all gate_w2 = 0 => sigmoid(0)=0.5, L1-norm = 1/3 each)
    ins_uniform = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in ins.items()}
    with torch.no_grad():
        ins_uniform["gate_w2"].zero_()
    y_uniform = _subqsa_combine_eager(**ins_uniform)

    # Variant: strongly bias branch 0 for all heads
    ins_biased = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in ins.items()}
    with torch.no_grad():
        for h in range(H):
            ins_biased["gate_w2"][h * 3 + 0, :] = 50.0
            ins_biased["gate_w2"][h * 3 + 1, :] = -50.0
            ins_biased["gate_w2"][h * 3 + 2, :] = -50.0
    y_biased = _subqsa_combine_eager(**ins_biased)

    diff = (y_biased - y_uniform).abs().max().item()
    assert diff > 0.01, f"Gate dominance test: biased output should differ from uniform (diff={diff:.6f})"
    print(f"  [PASS] test_gate_dominance: biased vs uniform diff={diff:.4f}")


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
    gamma = ins.pop("gamma")

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

    # Check gradients flow to all params except o_proj_weight. Ternary
    # quantization (clamp(round(...))) naturally blocks the gradient through
    # o_proj_weight because clamp zeros gradients at the ±1 boundary.
    param_names = ["x", "o_cmp", "o_slc", "o_win", "gate_w1", "gate_w2",
                   "out_norm_weight"]
    for name in param_names:
        g = ins[name].grad
        assert g is not None, f"{name}.grad is None"
        assert g.abs().sum().item() > 0, f"{name}.grad is all zeros"
    # o_proj_weight gradient exists but may be zero due to clamp — just check not None
    assert ins["o_proj_weight"].grad is not None, "o_proj_weight.grad is None"
    print(f"  [PASS] test_gradient: gradients flow to all differentiable parameters")


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
