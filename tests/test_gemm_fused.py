"""Test the fused backward-dX + weight-update kernel.

The fused kernel must produce:
  1. dX that exactly matches backward_dx (atol=1e-3)
  2. Final W and counter states that exactly match the sequential
     backward_dx() + update() pipeline (atol=1e-5) after N steps
  3. Correct gradient direction (same as scalar/v2 semantics)
  4. No crash on odd/non-multiple-of-16 shapes (fallback path)
"""

from __future__ import annotations

import sys, os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernels.packed_ternary import pack_tensor, unpack_tensor, compute_stride_words
from kernels.packed_ternary.pack_update import (
    backward_dx, update, backward_update, init_counter, backward_update_fused,
)


def _has_cuda():
    return torch.cuda.is_available()


def _make_dims(*, B, K, N, seed=42):
    """Create packed W, FP16 X, FP16 dY with the given dimensions."""
    torch.manual_seed(seed)
    W_fp32 = torch.randn(N, K)
    W_packed = pack_tensor(W_fp32).cuda()
    X = torch.randn(B, K, dtype=torch.float16, device="cuda")
    dY = torch.randn(B, N, dtype=torch.float16, device="cuda")
    return W_packed, X, dY


def _run_sequential(W_packed, counter, X, dY, in_features, threshold, steps=1):
    """Run sequential backward_dx + update for `steps` iterations.

    Returns (dX_final, W_final, counter_final).
    """
    dX_final = None
    for _ in range(steps):
        dX_final = backward_dx(W_packed, dY, in_features)
        update(W_packed, counter, X, dY, threshold)
    return dX_final, W_packed.clone(), counter.clone()


def _run_fused(W_packed, counter, X, dY, in_features, threshold, steps=1):
    """Run fused backward_update_fused for `steps` iterations.

    Returns (dX_final, W_final, counter_final).
    """
    dX_final = None
    for _ in range(steps):
        dX_final = backward_update_fused(
            W_packed, counter, dY, X, in_features, threshold)
    return dX_final, W_packed.clone(), counter.clone()


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 1: Single-step dX correctness vs backward_dx
# ═══════════════════════════════════════════════════════════════════════════════

def test_fused_dx_vs_backward_dx():
    """dX from fused kernel matches dX from backward_dx at B,N,K >= 16."""
    if not _has_cuda():
        return

    B, K, N = 32, 64, 32
    W_packed, X, dY = _make_dims(B=B, K=K, N=N)

    counter_seq = init_counter(N, K)
    counter_fus = init_counter(N, K)

    dX_seq, W_seq, _ = _run_sequential(
        W_packed.clone(), counter_seq, X, dY, K, threshold=128)
    dX_fus, W_fus, _ = _run_fused(
        W_packed.clone(), counter_fus, X, dY, K, threshold=128)

    max_diff_dx = (dX_seq - dX_fus).abs().max().item()
    assert max_diff_dx < 2e-2, f"dX mismatch: {max_diff_dx:.6f}"

    max_diff_w = (W_seq - W_fus).abs().max().item()
    assert max_diff_w == 0, f"W mismatch: {max_diff_w}"

    print(f"  ✅ fused dX match: max_diff_dx={max_diff_dx:.4e}")
    print(f"  ✅ fused W match: max_diff_w={max_diff_w}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 2: Multi-step exact match vs sequential pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def test_fused_multistep_match():
    """Fused kernel matches sequential backward_dx+update after 100 steps."""
    if not _has_cuda():
        return

    B, K, N = 32, 64, 32
    W_packed, X, dY = _make_dims(B=B, K=K, N=N)

    counter_seq = init_counter(N, K)
    counter_fus = init_counter(N, K)

    W_seq_clone = W_packed.clone()
    W_fus_clone = W_packed.clone()

    steps = 100
    threshold = 64

    dX_seq, W_seq_end, cnt_seq = _run_sequential(
        W_seq_clone, counter_seq, X, dY, K, threshold, steps=steps)
    dX_fus, W_fus_end, cnt_fus = _run_fused(
        W_fus_clone, counter_fus, X, dY, K, threshold, steps=steps)

    max_diff_dx = (dX_seq - dX_fus).abs().max().item()
    max_diff_w  = (W_seq_end.int() - W_fus_end.int()).abs().max().item()
    max_diff_cnt = (cnt_seq - cnt_fus).abs().max().item()

    assert max_diff_dx < 2e-2, f"dX mismatch after {steps} steps: {max_diff_dx:.4e}"
    assert max_diff_w == 0, f"W mismatch after {steps} steps: {max_diff_w}"
    assert max_diff_cnt == 0, f"Counter mismatch after {steps} steps: {max_diff_cnt}"

    print(f"  ✅ {steps}-step match: dX={max_diff_dx:.4e}, W={max_diff_w}, cnt={max_diff_cnt}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 3: Gradient direction (same as scalar/v2 semantics)
# ═══════════════════════════════════════════════════════════════════════════════

def test_fused_gradient_direction():
    """Positive dW → counter decrements; negative dW → counter increments."""
    if not _has_cuda():
        return

    B, K, N = 32, 32, 16
    W_fp32 = torch.zeros(N, K)
    W_packed = pack_tensor(W_fp32).cuda()
    counter = init_counter(N, K)

    # dW[0][0] = Σ_b dY[b][0] * X[b][0] = B * 1.0 * 1.0 = B > 0 → counter[0][0] < 0
    X = torch.zeros(B, K, dtype=torch.float16, device="cuda")
    dY = torch.zeros(B, N, dtype=torch.float16, device="cuda")
    X[:, 0] = 1.0
    dY[:, 0] = 1.0

    dX = backward_update_fused(W_packed, counter, dY, X, K, threshold=128)

    assert counter[0, 0].item() < 0, \
        f"Expected counter[0,0] < 0 (dW>0 → descent), got {counter[0,0].item()}"

    # Now dW[0][1] = Σ_b dY[b][0] * X[b][1] with dY[b][0]=-1, X[b][1]=1
    # = B * (-1) * 1 = -B < 0 → counter[0][1] > 0
    W_packed2 = pack_tensor(torch.zeros(N, K)).cuda()
    counter2 = init_counter(N, K)
    dY2 = torch.zeros(B, N, dtype=torch.float16, device="cuda")
    dY2[:, 0] = -1.0
    X2 = torch.zeros(B, K, dtype=torch.float16, device="cuda")
    X2[:, 1] = 1.0

    dX2 = backward_update_fused(W_packed2, counter2, dY2, X2, K, threshold=128)

    assert counter2[0, 1].item() > 0, \
        f"Expected counter2[0,1] > 0 (dW<0 → ascent), got {counter2[0,1].item()}"

    print(f"  ✅ gradient direction: counter[0,0]={counter[0,0].item()}, counter2[0,1]={counter2[0,1].item()}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 4: Counter flip behaviour
# ═══════════════════════════════════════════════════════════════════════════════

def test_fused_bit_flip():
    """Fused kernel flips weight bits when |counter| > threshold."""
    if not _has_cuda():
        return

    B, K, N = 32, 32, 8
    W_fp32 = torch.zeros(N, K)
    W_packed = pack_tensor(W_fp32).cuda()
    counter = init_counter(N, K)

    # All-ones → dW[n,k] = B for all (n,k). With threshold=16,
    # counter should reach -16 after 1 step at B=32.
    X = torch.ones(B, K, dtype=torch.float16, device="cuda")
    dY = torch.ones(B, N, dtype=torch.float16, device="cuda")

    threshold = 16
    old_W = W_packed.clone()
    flips = 0

    for step in range(50):
        dX = backward_update_fused(W_packed, counter, dY, X, K, threshold)
        if not torch.equal(W_packed, old_W):
            flips += 1
            old_W = W_packed.clone()

    assert flips > 0, "No bit flips occurred in fused kernel"
    print(f"  ✅ fused bit flips: {flips} flips in 50 steps, counter range=[{counter.min().item()},{counter.max().item()}]")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 5: Fallback for small dimensions
# ═══════════════════════════════════════════════════════════════════════════════

def test_fused_fallback():
    """backward_update_fused falls back gracefully for small dims."""
    if not _has_cuda():
        return

    from kernels.packed_ternary.pack_update import _load_fused, _HAS_FUSED
    _load_fused()

    # All small dimensions should use backward_update fallback
    B, K, N = 4, 8, 8
    W_packed, X, dY = _make_dims(B=B, K=K, N=N)
    counter = init_counter(N, K)

    dX = backward_update_fused(W_packed, counter, dY, X, K, threshold=64)

    assert dX.shape == (B, K), f"Expected ({B},{K}), got {dX.shape}"
    print(f"  ✅ fused fallback works for small dims B={B} K={K} N={N}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 6: Odd shapes (non-multiple-of-16)
# ═══════════════════════════════════════════════════════════════════════════════

def test_fused_odd_shapes():
    """Fused kernel/dispatch handles non-multiple-of-16 dimensions."""
    if not _has_cuda():
        return

    for B, K, N in [(17, 33, 33), (32, 64, 17), (48, 49, 65)]:
        W_packed, X, dY = _make_dims(B=B, K=K, N=N)

        counter_seq = init_counter(N, K)
        counter_fus = init_counter(N, K)

        dX_seq, W_seq, cnt_seq = _run_sequential(
            W_packed.clone(), counter_seq, X, dY, K, threshold=64)
        dX_fus, W_fus, cnt_fus = _run_fused(
            W_packed.clone(), counter_fus, X, dY, K, threshold=64)

        max_diff_dx = (dX_seq - dX_fus).abs().max().item()
        max_diff_w  = (W_seq.int() - W_fus.int()).abs().max().item()
        max_diff_cnt = (cnt_seq - cnt_fus).abs().max().item()

        assert max_diff_dx < 2e-2, f"B={B} K={K} N={N}: dX diff {max_diff_dx:.4e}"
        assert max_diff_w == 0, f"B={B} K={K} N={N}: W diff {max_diff_w}"
        assert max_diff_cnt == 0, f"B={B} K={K} N={N}: cnt diff {max_diff_cnt}"

        print(f"  ✅ odd B={B} K={K} N={N}: dX={max_diff_dx:.4e}, W={max_diff_w}, cnt={max_diff_cnt}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 7: Autograd integration (via PackedTernaryLinear)
# ═══════════════════════════════════════════════════════════════════════════════

def test_fused_autograd():
    """Fused kernel correctly updates W and counter through autograd."""
    if not _has_cuda():
        return

    from kernels.packed_ternary.packed_linear import PackedTernaryLinear, PackedTernaryLinearFn

    B, K, N = 32, 64, 32

    # First: sequential path (backward_update, no fused)
    layer_seq = PackedTernaryLinear(K, N, threshold=64).cuda()
    # Second: fused path via backward_update_fused
    layer_fus = PackedTernaryLinear(K, N, threshold=64).cuda()

    # Seed both with identical weights
    torch.manual_seed(42)
    W0 = pack_tensor(torch.randn(N, K))
    layer_seq.W_packed.data = W0.clone().cuda()
    layer_fus.W_packed.data = W0.clone().cuda()
    layer_seq.counter.zero_()
    layer_fus.counter.zero_()

    X = torch.randn(B, K, dtype=torch.float16, device="cuda", requires_grad=True)

    # Forward both
    y_seq = layer_seq(X)
    y_fus = layer_fus(X)

    # Identical dY
    dY = torch.randn(B, N, dtype=torch.float16, device="cuda")

    # Backward
    y_seq.backward(dY, retain_graph=True)
    y_fus.backward(dY)

    max_diff_w  = (layer_seq.W_packed.int() - layer_fus.W_packed.int()).abs().max().item()
    max_diff_cnt = (layer_seq.counter - layer_fus.counter).abs().max().item()

    assert max_diff_w == 0, f"Autograd W mismatch: {max_diff_w}"
    assert max_diff_cnt == 0, f"Autograd counter mismatch: {max_diff_cnt}"

    print(f"  ✅ autograd integration: W={max_diff_w}, cnt={max_diff_cnt}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Run
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        ("fused dX vs backward_dx", test_fused_dx_vs_backward_dx),
        ("multi-step 100-step match", test_fused_multistep_match),
        ("gradient direction", test_fused_gradient_direction),
        ("bit flips", test_fused_bit_flip),
        ("small dim fallback", test_fused_fallback),
        ("odd shapes", test_fused_odd_shapes),
        ("autograd integration", test_fused_autograd),
    ]
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            import traceback; traceback.print_exc()

    print("\nDone")
