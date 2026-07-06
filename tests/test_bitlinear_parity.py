"""Parity and dispatch tests for BitLinear eager vs fused paths.

Run with:
    python -m pytest tests/test_bitlinear_parity.py -v
"""

import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/debian/ultimate-ai-model")

from ultimate_trainer.bitlinear import BitLinear, absmax_quantize_activation


def _has_fused_kernel():
    try:
        from kernels.ternary_matmul import fused_bitlinear_forward

        return fused_bitlinear_forward is not None
    except Exception:
        return False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not _has_fused_kernel(), reason="fused kernel not importable")
def test_quantized_activations_eager_fused_parity():
    """With quantize_activations=True, eager and fused outputs match."""
    from kernels.ternary_matmul import fused_bitlinear_forward

    torch.manual_seed(0)
    module = BitLinear(64, 32, bias=True, quantize_activations=True)
    module.train()
    module.cuda()

    x = torch.randn(4, 64, device="cuda")

    # One forward pass to populate _w_ternary and _gamma.
    with torch.no_grad():
        _ = module(x)

    # Quantize activations exactly as BitLinear does in training.
    x_q = absmax_quantize_activation(x, bits=module.activation_bits)

    # Eager path: cached ternary weights + quantized activations.
    eager_out = F.linear(x_q, module._w_ternary, module.bias)

    # Fused path: on-the-fly weight quant + quantized activations.
    fused_out = fused_bitlinear_forward(x_q, module.weight, module._gamma, module.bias)

    assert eager_out.shape == fused_out.shape
    rel_diff = (eager_out - fused_out).abs().max() / eager_out.abs().max()
    assert torch.allclose(
        eager_out, fused_out, rtol=1e-4, atol=1e-5
    ), f"eager/fused mismatch (relative max diff {rel_diff:.3e})"


def test_quantize_activations_false_uses_eager_path():
    """With quantize_activations=False the fused kernel must not be invoked."""
    from kernels import ternary_matmul as tm
    import ultimate_trainer.bitlinear as bl

    torch.manual_seed(1)
    module = BitLinear(64, 32, bias=True, quantize_activations=False)
    module.train()

    x = torch.randn(4, 64)
    if torch.cuda.is_available():
        module = module.cuda()
        x = x.cuda()

    original_fused = tm.fused_bitlinear_forward
    try:

        def raising_fused(*args, **kwargs):
            raise AssertionError("fused_bitlinear_forward should not be called")

        tm.fused_bitlinear_forward = raising_fused
        bl.fused_bitlinear_forward = raising_fused

        out = module(x)
        assert out.shape == (4, 32)
        assert torch.isfinite(out).all()
    finally:
        tm.fused_bitlinear_forward = original_fused
        bl.fused_bitlinear_forward = original_fused


def test_cpu_or_no_triton_fallback():
    """On CPU / when Triton is unavailable the eager fallback runs cleanly."""
    torch.manual_seed(2)
    module = BitLinear(64, 32, bias=True, quantize_activations=True)
    module.train()
    x = torch.randn(4, 64)

    out = module(x)
    assert out.shape == (4, 32)
    assert torch.isfinite(out).all()


def test_quantize_activations_false_sensible_output():
    """quantize_activations=False still produces sensible output."""
    torch.manual_seed(3)
    module = BitLinear(64, 32, bias=True, quantize_activations=False)
    module.eval()
    x = torch.randn(4, 64)

    out = module(x)
    assert out.shape == (4, 32)
    assert torch.isfinite(out).all()


def test_eval_mode_refreshes_ternary_weights():
    """eval() recomputes _gamma and _w_ternary from the current weight."""
    torch.manual_seed(4)
    module = BitLinear(128, 128, bias=False)
    x = torch.randn(2, 128)

    module.eval()
    y = module(x)
    assert y.shape == (2, 128)
    assert torch.isfinite(y).all()

    # After a training update, the cached ternary weights should refresh on eval.
    with torch.no_grad():
        module.weight.add_(torch.randn_like(module.weight) * 0.01)
    stale_w_ternary = module._w_ternary.clone()
    module.eval()
    assert not torch.equal(module._w_ternary, stale_w_ternary)
