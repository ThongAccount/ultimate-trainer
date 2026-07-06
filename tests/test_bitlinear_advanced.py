"""Advanced edge-case and math-property tests for the bitlinear module.

Covers zero inputs, uniform inputs, STE gradient identity, train/eval
dispatch, ternary weight correctness, cache invalidation, RMSNorm
invariance, no-quant forward pass, and quant_update_freq timing.

Run with:
    python -m pytest tests/test_bitlinear_advanced.py -v
"""

import sys
import types

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/debian/ultimate-ai-model")

from ultimate_trainer.bitlinear import BitLinear, RMSNorm, absmax_quantize_activation


# ===================================================================
# 1.  absmax_quantize_activation — zero input
# ===================================================================


def test_absmax_quantize_zero_input():
    """When act is all zeros, output should be zeros, not NaN."""
    act = torch.zeros(4, 64)
    out = absmax_quantize_activation(act, bits=8)

    assert out.shape == act.shape
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-8), \
        "Zero input must produce near-zero output"
    assert not torch.isnan(out).any(), "Zero input must not produce NaN"

    # 1-D input (single sample)
    act_1d = torch.zeros(128)
    out_1d = absmax_quantize_activation(act_1d, bits=8)
    assert out_1d.shape == act_1d.shape
    assert torch.allclose(out_1d, torch.zeros_like(out_1d), atol=1e-8)


# ===================================================================
# 2.  absmax_quantize_activation — all-same values
# ===================================================================


def test_absmax_quantize_all_same():
    """When act has the same value across the last dim, scale works correctly."""
    # Positive constant per row
    act = torch.ones(4, 64) * 3.0
    out = absmax_quantize_activation(act, bits=8)
    assert out.shape == act.shape
    assert torch.isfinite(out).all()

    # Larger constant near q_max
    act_large = torch.ones(2, 32) * 100.0
    out_large = absmax_quantize_activation(act_large, bits=8)
    assert out_large.shape == act_large.shape
    assert torch.isfinite(out_large).all()

    # Negative constant
    act_neg = torch.ones(2, 32) * (-42.0)
    out_neg = absmax_quantize_activation(act_neg, bits=8)
    assert out_neg.shape == act_neg.shape
    assert torch.isfinite(out_neg).all()

    # Constant zero (already covered by test 1, but here with explicit shape)
    act_zero = torch.zeros(3, 16)
    out_zero = absmax_quantize_activation(act_zero, bits=8)
    assert torch.allclose(out_zero, torch.zeros_like(out_zero), atol=1e-8)


# ===================================================================
# 3.  absmax_quantize_activation — STE gradient identity
# ===================================================================


def test_absmax_quantize_ste_gradient():
    """Gradient of output w.r.t. input should be 1 (STE identity).

    The STE formulation is: y = x + (deq(x) - x).detach(), so
    dy/dx = dx/dx + 0 = 1 element-wise.
    """
    x = torch.randn(4, 64, requires_grad=True)
    y = absmax_quantize_activation(x, bits=8)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None, "Gradient must flow through the STE path"
    assert torch.allclose(x.grad, torch.ones_like(x.grad), atol=1e-7), \
        "STE gradient w.r.t. input must be all-ones (identity)"
    assert x.grad.abs().sum().item() > 0.0, "Gradient must be non-zero"

    # Repeat with 4-bit quantization
    x2 = torch.randn(2, 32, requires_grad=True)
    y2 = absmax_quantize_activation(x2, bits=4)
    y2.sum().backward()
    assert x2.grad is not None
    assert torch.allclose(x2.grad, torch.ones_like(x2.grad), atol=1e-7), \
        "STE identity must hold for 4-bit quantization"


# ===================================================================
# 4.  BitLinear forward — train vs eval dispatch
# ===================================================================


def test_bitlinear_forward_train_eval_dispatch():
    """On CPU, the train() path must differ from the eval() path.

    Training quantizes activations with absmax_quantize_activation;
    eval (on CPU) does not.  Both use the same cached ternary weights,
    so the only difference comes from activation quantization.
    """
    torch.manual_seed(42)
    module = BitLinear(32, 16, bias=False, quantize_activations=True)
    x = torch.randn(4, 32)

    # ---- eval path ----
    module.eval()
    y_eval = module(x)

    # Capture _w_ternary after eval's refresh
    w_eval = module._w_ternary.clone()

    # ---- train path ----
    module.train()
    module._quant_step = 0       # ensure first forward triggers refresh
    y_train = module(x)

    # Both paths should have the same _w_ternary (weight has not changed)
    assert torch.equal(w_eval, module._w_ternary), \
        "_w_ternary must be identical in train and eval (same weight)"

    # Outputs must differ because train quantizes activations
    assert not torch.allclose(y_eval, y_train, atol=1e-6), \
        "Train output (quantised activations) must differ from eval output"

    # Both outputs must be valid
    assert y_train.shape == (4, 16), f"Expected (4, 16), got {y_train.shape}"
    assert y_eval.shape == (4, 16)
    assert torch.isfinite(y_train).all()
    assert torch.isfinite(y_eval).all()


# ===================================================================
# 5.  _w_ternary values
# ===================================================================


def test_refresh_ternary_weights_values():
    """Verify _w_ternary contains only {-1, 0, +1} and matches manual computation."""
    torch.manual_seed(42)
    module = BitLinear(64, 32, bias=False)
    module.eval()                       # triggers _refresh_ternary_weights

    w_ternary = module._w_ternary
    assert w_ternary.shape == module.weight.shape

    # Check every unique value is in {-1, 0, +1}
    unique_vals = torch.unique(w_ternary)
    for v in unique_vals:
        vi = v.item()
        ok = (abs(vi) < 1e-10) or (abs(abs(vi) - 1.0) < 1e-10)
        assert ok, f"Ternary weight has unexpected value {vi} " \
                   f"(should be -1, 0, or +1)"

    # Manual computation using compute_gamma's formula on CPU
    gamma = module.weight.abs().mean() + 1e-5
    gamma = gamma.reshape(1) if gamma.ndim == 0 else gamma
    w_q_manual = torch.clamp(torch.round(module.weight / gamma), -1.0, 1.0)

    # _w_ternary's forward-pass values equal w_q (STE: result = w_q)
    assert torch.equal(w_ternary, w_q_manual), \
        "Cached ternary weights must match manual round/clamp computation"

    # Repeat with a different random seed for robustness
    torch.manual_seed(99)
    module2 = BitLinear(32, 64, bias=True)
    module2.eval()
    w_ternary2 = module2._w_ternary
    unique_vals2 = torch.unique(w_ternary2)
    for v in unique_vals2:
        vi = v.item()
        ok = (abs(vi) < 1e-10) or (abs(abs(vi) - 1.0) < 1e-10)
        assert ok, f"Ternary weight has unexpected value {vi}"


# ===================================================================
# 6.  _refresh_ternary_weights after weight change
# ===================================================================


def test_refresh_ternary_weights_after_weight_change():
    """Verify weight update followed by refresh invalidates the cached ternary weights."""
    module = BitLinear(64, 32, bias=True)
    module.eval()                          # initialise ternary cache

    old_ternary = module._w_ternary.clone()

    # Significantly change the master weights
    with torch.no_grad():
        module.weight.copy_(torch.randn_like(module.weight) * 10.0)

    # Force a cache refresh
    module._refresh_ternary_weights()

    # The cached ternary weights must now differ
    assert not torch.equal(module._w_ternary, old_ternary), \
        "Ternary weights must change after weight update followed by refresh"

    # New values must still be valid ternary {-1, 0, +1}
    new_unique = torch.unique(module._w_ternary)
    for v in new_unique:
        vi = v.item()
        ok = (abs(vi) < 1e-10) or (abs(abs(vi) - 1.0) < 1e-10)
        assert ok, f"After refresh, ternary weight has unexpected value {vi}"


# ===================================================================
# 7.  RMSNorm unit RMS
# ===================================================================


def test_rmsnorm_unit_rms():
    """Verify RMSNorm output has RMS close to 1 when weight is all-ones."""
    module = RMSNorm(64)

    # Normal-scale input
    x = torch.randn(4, 64)
    y = module(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5), \
        f"RMS of RMSNorm output should be ~1, got {rms}"

    # Large-magnitude input
    x_large = torch.randn(4, 64) * 10.0
    y_large = module(x_large)
    rms_large = y_large.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms_large, torch.ones_like(rms_large), atol=1e-5), \
        f"RMS for large input should be ~1, got {rms_large}"

    # Small-magnitude input (RMS ~ 0.01; eps=1e-5 causes ~0.1% deviation)
    x_small = torch.randn(4, 64) * 0.01
    y_small = module(x_small)
    rms_small = y_small.pow(2).mean(-1).sqrt()
    assert rms_small.min() > 0.998, \
        f"RMS for small input should be near 1, got min {rms_small.min():.6f}"

    # Single-sample path (1-D input)
    x_1d = torch.randn(64)
    y_1d = module(x_1d)
    rms_1d = y_1d.pow(2).mean().sqrt()
    assert abs(rms_1d - 1.0) < 1e-5, f"RMS for 1D input should be ~1, got {rms_1d}"


# ===================================================================
# 8.  BitLinear forward with quantize_activations=False
# ===================================================================


def test_bitlinear_no_quant_forward():
    """quantize_activations=False still produces valid output in all modes."""
    torch.manual_seed(42)
    x = torch.randn(4, 32)

    # ---- Train mode ----
    module = BitLinear(32, 16, bias=True, quantize_activations=False)
    module.train()

    y1 = module(x)
    assert y1.shape == (4, 16)
    assert torch.isfinite(y1).all()

    # Multiple forward passes should work (no one-shot refresh bug)
    y2 = module(x)
    assert y2.shape == (4, 16)
    assert torch.isfinite(y2).all()

    # ---- Eval mode ----
    module.eval()
    y3 = module(x)
    assert y3.shape == (4, 16)
    assert torch.isfinite(y3).all()

    # ---- Without bias ----
    module_no_bias = BitLinear(32, 16, bias=False, quantize_activations=False)
    module_no_bias.train()
    y4 = module_no_bias(x)
    assert y4.shape == (4, 16)
    assert torch.isfinite(y4).all()

    # ---- Different shape (wider output) ----
    module_wide = BitLinear(16, 64, bias=True, quantize_activations=False)
    x_wide = torch.randn(2, 16)
    y5 = module_wide(x_wide)
    assert y5.shape == (2, 64)
    assert torch.isfinite(y5).all()

    # ---- Output is deterministic (same seed, same input) ----
    torch.manual_seed(1)
    m = BitLinear(32, 16, bias=True, quantize_activations=False)
    x_det = torch.randn(4, 32)
    a = m(x_det)
    b = m(x_det)
    assert torch.equal(a, b), "Deterministic output expected with no activation noise"


# ===================================================================
# 9.  quant_update_freq timing
# ===================================================================


def test_bitlinear_quant_update_freq():
    """Verify quant_update_freq controls how often ternary weights are refreshed."""
    freq = 3
    module = BitLinear(32, 16, bias=False, quantize_activations=False,
                       quant_update_freq=freq)
    module.train()
    module._quant_step = 0

    x = torch.randn(2, 32)

    # Monkey-patch _refresh_ternary_weights to log when it is called
    orig_refresh = module._refresh_ternary_weights
    call_steps = []

    def _tracking_refresh(self):
        call_steps.append(self._quant_step)   # step that triggered the refresh
        orig_refresh()

    module._refresh_ternary_weights = types.MethodType(_tracking_refresh, module)

    # Run 7 forward passes
    for _ in range(7):
        module(x)

    # Refresh triggers when _quant_step % freq == 1:
    #   Step 1: 1 % 3 == 1  --> refresh
    #   Step 4: 4 % 3 == 1  --> refresh
    #   Step 7: 7 % 3 == 1  --> refresh
    assert call_steps == [1, 4, 7], \
        f"Expected refreshes at steps [1, 4, 7], got {call_steps}"

    assert module._quant_step == 7, \
        f"_quant_step should be 7, got {module._quant_step}"

    # ---- With freq=2 (refresh every other step) ----
    # Note: formula is _quant_step % freq == 1, so freq=2 refreshes
    # on odd-numbered steps: 1, 3, 5, ...
    module2 = BitLinear(16, 8, bias=False, quantize_activations=False,
                        quant_update_freq=2)
    module2.train()
    module2._quant_step = 0

    orig_refresh2 = module2._refresh_ternary_weights
    call_steps2 = []

    def _tracking_refresh2(self):
        call_steps2.append(self._quant_step)
        orig_refresh2()

    module2._refresh_ternary_weights = types.MethodType(_tracking_refresh2, module2)

    for _ in range(6):
        module2(x[:, :16])

    # With freq=2, only odd steps trigger refresh
    assert call_steps2 == [1, 3, 5], \
        f"With freq=2 expected [1, 3, 5], got {call_steps2}"


# ===================================================================
# Entry point (optional)
# ===================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
