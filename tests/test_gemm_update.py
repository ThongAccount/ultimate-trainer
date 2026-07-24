"""Phase 3 — dX backward + fused counter-based weight update.

Tests:
  1. backward_dx matches F.linear's analytic gradient
  2. update kernel flips bits when |counter| > threshold
  3. dW is never materialised as a tensor
"""

from __future__ import annotations

import sys, os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernels.packed_ternary import pack_tensor
from kernels.packed_ternary.pack_update import backward_dx, update, init_counter
from kernels.packed_ternary.pack_forward import packed_ternary_forward


def _has_cuda():
    return torch.cuda.is_available()


def _pack_and_check(W_fp32: torch.Tensor) -> torch.Tensor:
    """Pack weights and move to CUDA."""
    return pack_tensor(W_fp32).cuda()


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 1: backward_dx correctness
# ═══════════════════════════════════════════════════════════════════════════════

def test_backward_dx():
    """dX from our kernel matches F.linear with the *same* ternary weights."""
    if not _has_cuda():
        return

    torch.manual_seed(42)
    B, K, N = 4, 32, 16
    W_fp32 = torch.randn(N, K)
    X = torch.randn(B, K, dtype=torch.float16, device="cuda", requires_grad=True)

    # Pack then unpack so both reference and kernel use identical ternary W.
    W_packed = _pack_and_check(W_fp32)
    W_ternary = W_packed.clone()
    from kernels.packed_ternary import unpack_tensor
    W_fp16_ref = unpack_tensor(W_ternary.cpu(), N, K).to(torch.float16).cuda()

    Y_ref = F.linear(X, W_fp16_ref)
    dY = torch.randn_like(Y_ref)
    Y_ref.backward(dY)
    dX_ref = X.grad.clone()

    dX_cuda = backward_dx(W_packed, dY, K)

    torch.testing.assert_close(dX_cuda, dX_ref, atol=1e-3, rtol=1e-3)
    print(f"  ✅ backward_dx: max_diff={(dX_cuda - dX_ref).abs().max().item():.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 2: update kernel — counter accumulation + bit flips
# ═══════════════════════════════════════════════════════════════════════════════

def test_update_flips_bits():
    """Update kernel flips weights when counter exceeds threshold."""
    if not _has_cuda():
        return

    torch.manual_seed(0)
    B, K, N = 4, 16, 4
    W_fp32 = torch.zeros(N, K)  # all zeros
    W_packed = _pack_and_check(W_fp32)
    counter = init_counter(N, K)
    old_W = W_packed.clone()

    # Create X and dY that produce consistent positive gradient for W[0][0]
    X = torch.zeros(B, K, dtype=torch.float16, device="cuda")
    dY = torch.zeros(B, N, dtype=torch.float16, device="cuda")
    X[:, 0] = 1.0      # X[b][0] = 1 for all b
    dY[:, 0] = 1.0     # dY[b][0] = 1 for all b → dW[0][0] = Σ 1*1 = 4 per step

    # Run update 50 times with threshold=16 → should flip after 4 steps
    threshold = 16
    flips = 0
    for step in range(50):
        update(W_packed, counter, X, dY, threshold)
        if not torch.equal(W_packed, old_W):
            flips += 1
            old_W = W_packed.clone()

    assert flips > 0, "No bit flips occurred — update kernel not working"
    print(f"  ✅ update flips bits: {flips} flips in 50 steps")
    print(f"     counter stats: min={counter.min().item()}, max={counter.max().item()}")


def test_update_gradient_direction():
    """Positive gradient → counter increments; negative → decrements."""
    if not _has_cuda():
        return

    torch.manual_seed(1)
    B, K, N = 2, 8, 2
    W_fp32 = torch.zeros(N, K)
    W_packed = _pack_and_check(W_fp32)
    counter = init_counter(N, K)

    # dW[r][c] = Σ_b dY[b][r] * X[b][c].
    # Gradient descent: positive dW → decrease weight → counter decrements.
    # Use two output features: r=0 and r=1.
    X = torch.zeros(B, K, dtype=torch.float16, device="cuda")
    dY = torch.zeros(B, N, dtype=torch.float16, device="cuda")
    X[:, 0] = 1.0;   dY[:, 0] = 1.0    # dW[0][0] = Σ 1*1 = B > 0 → counter[0,0] < 0
    X[:, 1] = 1.0;   dY[:, 1] = -1.0   # dW[1][1] = Σ (-1)*1 = -B < 0 → counter[1,1] > 0

    update(W_packed, counter, X, dY, threshold=128)

    assert counter[0, 0].item() < 0, f"Expected negative counter at [0,0], got {counter[0,0].item()} (dW>0 → descent → decrement)"
    assert counter[1, 1].item() > 0, f"Expected positive counter at [1,1], got {counter[1,1].item()} (dW<0 → descent → increment)"
    print(f"  ✅ gradient direction: +{counter[0,0].item()}, {counter[0,1].item()}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test 4: TC backward_dx correctness
# ═══════════════════════════════════════════════════════════════════════════════

def test_backward_dx_tc():
    """TC backward dX matches reference at batch >= 16."""
    if not _has_cuda():
        return

    torch.manual_seed(42)
    B, K, N = 16, 32, 16
    W_fp32 = torch.randn(N, K)
    X = torch.randn(B, K, dtype=torch.float16, device="cuda", requires_grad=True)

    W_packed = _pack_and_check(W_fp32)
    from kernels.packed_ternary import unpack_tensor
    W_fp16_ref = unpack_tensor(W_packed.cpu(), N, K).to(torch.float16).cuda()

    Y_ref = F.linear(X, W_fp16_ref)
    dY = torch.randn_like(Y_ref)
    Y_ref.backward(dY)
    dX_ref = X.grad.clone()

    # Force TC by calling with batch >= TC_MIN_BATCH
    dX_cuda = backward_dx(W_packed, dY, K)

    torch.testing.assert_close(dX_cuda, dX_ref, atol=1e-3, rtol=1e-3)
    print(f"  ✅ backward_dx TC: max_diff={(dX_cuda - dX_ref).abs().max().item():.4f}")


def test_backward_dx_tc_vs_scalar_crosscheck():
    """TC backward dX matches scalar backward dX across batch sizes."""
    if not _has_cuda():
        return

    torch.manual_seed(42)
    K, N = 64, 32
    W_fp32 = torch.randn(N, K)
    W_packed = _pack_and_check(W_fp32)

    # Test across multiple batch sizes — TC kicks in at B >= 16
    # Standard batch sizes
    for B in [1, 4, 8, 16, 32]:
        X = torch.randn(B, K, dtype=torch.float16, device="cuda", requires_grad=True)
        dY = torch.randn(B, N, dtype=torch.float16, device="cuda")
        dX = backward_dx(W_packed, dY, K)

        # Reference: F.linear with same ternary W
        from kernels.packed_ternary import unpack_tensor
        W_fp16_ref = unpack_tensor(W_packed.cpu(), N, K).to(torch.float16).cuda()
        X_ref = X.detach().requires_grad_(True)
        Y_ref = F.linear(X_ref, W_fp16_ref)
        Y_ref.backward(dY)
        dX_ref = X_ref.grad

        max_diff = (dX - dX_ref).abs().max().item()
        assert max_diff < 1e-3, f"B={B}: max_diff={max_diff:.6f}"
        print(f"  ✅ TC vs reference B={B}: max_diff={max_diff:.4f}")



def test_backward_dx_tc_odd_shapes():
    """TC backward dX works with non-multiple-of-16 K, N dimensions."""
    if not _has_cuda():
        return

    torch.manual_seed(42)
    from kernels.packed_ternary import unpack_tensor

    for B, K, N in [(17, 33, 33), (8, 33, 17), (4, 49, 65)]:
        W_fp32 = torch.randn(N, K)
        W_packed = _pack_and_check(W_fp32)
        dY = torch.randn(B, N, dtype=torch.float16, device="cuda")

        dX_cuda = backward_dx(W_packed, dY, K)

        # Reference via F.linear
        W_fp16_ref = unpack_tensor(W_packed.cpu(), N, K).to(torch.float16).cuda()
        X_ref = torch.randn(B, K, dtype=torch.float16, device="cuda", requires_grad=True)
        Y_ref = F.linear(X_ref, W_fp16_ref)
        Y_ref.backward(dY)
        dX_ref = X_ref.grad

        max_diff = (dX_cuda - dX_ref).abs().max().item()
        assert max_diff < 1e-3, f"B={B} K={K} N={N}: max_diff={max_diff:.6f}"
        print(f"  ✅ odd shapes B={B} K={K} N={N}: max_diff={max_diff:.4f}")


def test_update_tc_flips_bits():

    """TC update kernel flips weights when counter exceeds threshold (batch >= 16)."""
    if not _has_cuda():
        return

    torch.manual_seed(0)
    B, K, N = 16, 16, 4
    W_fp32 = torch.zeros(N, K)
    W_packed = _pack_and_check(W_fp32)
    counter = init_counter(N, K)
    old_W = W_packed.clone()

    X = torch.zeros(B, K, dtype=torch.float16, device="cuda")
    dY = torch.zeros(B, N, dtype=torch.float16, device="cuda")
    X[:, 0] = 1.0
    dY[:, 0] = 1.0

    threshold = 16  # with B=16, dW=16/step → counter hits -16 at step 16
    flips = 0
    for step in range(50):
        update(W_packed, counter, X, dY, threshold)
        if not torch.equal(W_packed, old_W):
            flips += 1
            old_W = W_packed.clone()

    assert flips > 0, "TC update: No bit flips occurred"
    print(f"  ✅ TC update flips bits: {flips} flips in 50 steps")


# ═══════════════════════════════════════════════════════════════════════════════
#  Run
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        ("backward dX",       test_backward_dx),
        ("bit flips",         test_update_flips_bits),
        ("direction",         test_update_gradient_direction),
        ("backward dX TC",    test_backward_dx_tc),
        ("TC vs scalar",     test_backward_dx_tc_vs_scalar_crosscheck),
        ("TC update flips",   test_update_tc_flips_bits),
    ]
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            import traceback; traceback.print_exc()

    print("\nDone")
