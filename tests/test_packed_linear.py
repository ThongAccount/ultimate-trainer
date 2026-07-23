"""Integration test: PackedTernaryLinear nn.Module end-to-end."""

from __future__ import annotations

import sys, os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernels.packed_ternary import PackedTernaryLinear


def test_forward_shapes():
    """Module produces correct output shapes."""
    if not torch.cuda.is_available():
        return
    layer = PackedTernaryLinear(64, 128, threshold=64).cuda()
    x = torch.randn(4, 64, dtype=torch.float16, device="cuda")
    y = layer(x)
    assert y.shape == (4, 128), f"Expected (4,128), got {y.shape}"
    print(f"  ✅ forward shapes: {y.shape}")


def test_forward_backward():
    """Forward + backward executes without error and updates weights."""
    if not torch.cuda.is_available():
        return
    layer = PackedTernaryLinear(32, 16, threshold=32).cuda()
    x = torch.randn(8, 32, dtype=torch.float16, device="cuda", requires_grad=True)

    # Forward
    y = layer(x)
    loss = y.mean()

    # Backward
    loss.backward()

    # x.grad should exist
    assert x.grad is not None, "x.grad should be populated"
    assert x.grad.shape == x.shape, f"Expected {x.shape}, got {x.grad.shape}"

    # Counter should have changed
    assert layer.counter.abs().sum().item() > 0, "Counter should have changed"
    print(f"  ✅ backward: x.grad norm={x.grad.norm().item():.4f}, "
          f"counter sum={layer.counter.abs().sum().item()}")


def test_multistep():
    """Multiple forward/backward steps accumulate and eventually flip bits."""
    if not torch.cuda.is_available():
        return
    layer = PackedTernaryLinear(16, 8, threshold=4).cuda()
    old_W = layer.W_packed.clone()

    # Use negative-mean loss: pushes all weights consistently.  bprop dY = -1/64
    # for a (8,8) output, giving steady gradient direction.
    flips = 0
    for step in range(30):
        x = torch.randn(8, 16, dtype=torch.float16, device="cuda")
        y = layer(x)
        loss = -y.mean()  # pushes all outputs up → consistent sign gradient
        loss.backward()

        if not torch.equal(layer.W_packed, old_W):
            flips += 1
            old_W = layer.W_packed.clone()

    assert flips > 0, "No bit flips in 30 steps — update kernel not working"
    print(f"  ✅ multistep: {flips} flips in 30 steps")


def test_bias():
    """Bias term is added correctly."""
    if not torch.cuda.is_available():
        return
    layer = PackedTernaryLinear(16, 8, bias=True).cuda()
    layer.bias.data = torch.ones(8, dtype=torch.float16, device="cuda")

    x = torch.zeros(2, 16, dtype=torch.float16, device="cuda")
    y = layer(x)
    assert torch.allclose(y, torch.ones(2, 8, dtype=torch.float16, device="cuda"))
    print(f"  ✅ bias: all ones from zero input")


def test_save_load():
    """State dict roundtrip preserves weights and counter."""
    if not torch.cuda.is_available():
        return
    layer = PackedTernaryLinear(32, 16, threshold=64).cuda()
    x = torch.randn(4, 32, dtype=torch.float16, device="cuda")
    y0 = layer(x)

    sd = layer.state_dict()
    layer2 = PackedTernaryLinear(32, 16, threshold=64).cuda()
    layer2.load_state_dict(sd)

    y1 = layer2(x)
    assert torch.allclose(y0, y1), "Outputs should match after save/load"
    assert torch.equal(layer.W_packed, layer2.W_packed), "Weights should match"
    print(f"  ✅ save/load: outputs match")


def test_auto_dispatch():
    """Auto-dispatch chooses the right kernel for different batch sizes."""
    if not torch.cuda.is_available():
        return
    from kernels.packed_ternary.packed_linear import _forward_auto as dispatch

    W = torch.zeros(32, 2, dtype=torch.int32, device="cuda")
    X_small = torch.randn(4, 32, dtype=torch.float16, device="cuda")
    X_large = torch.randn(32, 32, dtype=torch.float16, device="cuda")

    Y_small = dispatch(W, X_small)
    Y_large = dispatch(W, X_large)

    assert Y_small.shape == (4, 32), f"Expected (4,32), got {Y_small.shape}"
    assert Y_large.shape == (32, 32), f"Expected (32,32), got {Y_large.shape}"
    print(f"  ✅ auto-dispatch: small={Y_small.shape}, large={Y_large.shape}")


if __name__ == "__main__":
    tests = [
        ("forward shapes", test_forward_shapes),
        ("forward/backward", test_forward_backward),
        ("multistep", test_multistep),
        ("bias", test_bias),
        ("save/load", test_save_load),
        ("auto-dispatch", test_auto_dispatch),
    ]
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            import traceback; traceback.print_exc()

    print("\nDone")
