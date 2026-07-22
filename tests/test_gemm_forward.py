"""Phase 2A — Correctness: Packed ternary × FP16 forward GEMM.

Success criterion:
    torch.testing.assert_close(Y_cuda, Y_ref, atol=1e-3, rtol=1e-3)

Where:
    Y_cuda = packed_ternary_forward(W_packed, X)
    Y_ref  = F.linear(X_fp16, W_fp16_dequantised)
"""

from __future__ import annotations

import math
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernels.packed_ternary import pack_tensor
from kernels.packed_ternary.pack_forward import (
    has_forward_kernel,
    packed_ternary_forward,
    ref_linear,
)


def _check_cuda():
    if not torch.cuda.is_available():
        return False
    if not has_forward_kernel():
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  Correctness tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_single_row():
    """1 × N: one output feature, varying input sizes."""
    if not _check_cuda():
        return
    for in_features in [1, 8, 15, 16, 17, 31, 32, 64, 128]:
        W_fp32 = torch.randn(1, in_features)
        W_packed = pack_tensor(W_fp32).cuda()
        X = torch.randn(2, in_features, dtype=torch.float16, device="cuda")

        Y_cuda = packed_ternary_forward(W_packed, X)
        Y_ref = ref_linear(W_packed, X)

        torch.testing.assert_close(Y_cuda, Y_ref, atol=1e-3, rtol=1e-3,
                                   msg=f"Failed at in_features={in_features}")
        print(f"  ✅ 1×{in_features}")


def test_single_column():
    """N × 1: multiple output features, one input feature."""
    if not _check_cuda():
        return
    for out_features in [1, 8, 16, 32, 64]:
        W_fp32 = torch.randn(out_features, 1)
        W_packed = pack_tensor(W_fp32).cuda()
        X = torch.randn(2, 1, dtype=torch.float16, device="cuda")

        Y_cuda = packed_ternary_forward(W_packed, X)
        Y_ref = ref_linear(W_packed, X)

        torch.testing.assert_close(Y_cuda, Y_ref, atol=1e-3, rtol=1e-3,
                                   msg=f"Failed at out_features={out_features}")
        print(f"  ✅ {out_features}×1")


def test_square():
    """N × N: square matrices at various sizes."""
    if not _check_cuda():
        return
    for N in [4, 8, 16, 32, 48, 64]:
        W_fp32 = torch.randn(N, N)
        W_packed = pack_tensor(W_fp32).cuda()
        X = torch.randn(4, N, dtype=torch.float16, device="cuda")

        Y_cuda = packed_ternary_forward(W_packed, X)
        Y_ref = ref_linear(W_packed, X)

        torch.testing.assert_close(Y_cuda, Y_ref, atol=1e-3, rtol=1e-3,
                                   msg=f"Failed at N={N}")
        print(f"  ✅ {N}×{N}")


def test_rectangular():
    """M × N: various rectangular shapes."""
    if not _check_cuda():
        return
    cases = [(16, 32), (32, 16), (8, 64), (64, 128), (128, 256)]
    for out_f, in_f in cases:
        W_fp32 = torch.randn(out_f, in_f)
        W_packed = pack_tensor(W_fp32).cuda()
        X = torch.randn(2, in_f, dtype=torch.float16, device="cuda")

        Y_cuda = packed_ternary_forward(W_packed, X)
        Y_ref = ref_linear(W_packed, X)

        torch.testing.assert_close(Y_cuda, Y_ref, atol=1e-3, rtol=1e-3,
                                   msg=f"Failed at {out_f}×{in_f}")
        print(f"  ✅ {out_f}×{in_f}")


def test_multibatch():
    """Large batch: verify every batch element independently."""
    if not _check_cuda():
        return
    W_fp32 = torch.randn(16, 32)
    W_packed = pack_tensor(W_fp32).cuda()
    X = torch.randn(64, 32, dtype=torch.float16, device="cuda")

    Y_cuda = packed_ternary_forward(W_packed, X)
    Y_ref = ref_linear(W_packed, X)

    torch.testing.assert_close(Y_cuda, Y_ref, atol=1e-3, rtol=1e-3)
    print(f"  ✅ batch=64")


def test_zeros_in_weights():
    """W has many zeros → verify kernel handles them (no division by zero, etc.)."""
    if not _check_cuda():
        return
    W_fp32 = torch.zeros(32, 32)
    W_fp32[:8, :8] = 1.0    # block of +1
    W_fp32[8:16, :8] = -1.0  # block of -1
    W_packed = pack_tensor(W_fp32).cuda()
    X = torch.randn(2, 32, dtype=torch.float16, device="cuda")

    Y_cuda = packed_ternary_forward(W_packed, X)
    Y_ref = ref_linear(W_packed, X)

    torch.testing.assert_close(Y_cuda, Y_ref, atol=1e-3, rtol=1e-3)
    print(f"  ✅ zeros in weights")


def test_all_positive():
    """All weights = +1: Y should be row-wise sum of X."""
    if not _check_cuda():
        return
    W_fp32 = torch.ones(4, 16)
    W_packed = pack_tensor(W_fp32).cuda()
    X = torch.randn(2, 16, dtype=torch.float16, device="cuda")

    Y_cuda = packed_ternary_forward(W_packed, X)
    Y_ref = ref_linear(W_packed, X)

    torch.testing.assert_close(Y_cuda, Y_ref, atol=1e-3, rtol=1e-3)
    print(f"  ✅ all +1")


def test_all_negative():
    """All weights = -1: Y should be negative row-wise sum of X."""
    if not _check_cuda():
        return
    W_fp32 = -torch.ones(4, 16)
    W_packed = pack_tensor(W_fp32).cuda()
    X = torch.randn(2, 16, dtype=torch.float16, device="cuda")

    Y_cuda = packed_ternary_forward(W_packed, X)
    Y_ref = ref_linear(W_packed, X)

    torch.testing.assert_close(Y_cuda, Y_ref, atol=1e-3, rtol=1e-3)
    print(f"  ✅ all -1")


def test_deterministic():
    """Same input produces same output (no race conditions)."""
    if not _check_cuda():
        return
    W_fp32 = torch.randn(32, 64)
    W_packed = pack_tensor(W_fp32).cuda()
    X = torch.randn(4, 64, dtype=torch.float16, device="cuda")

    Y_1 = packed_ternary_forward(W_packed, X)
    Y_2 = packed_ternary_forward(W_packed, X)

    torch.testing.assert_close(Y_1, Y_2, atol=0, rtol=0)
    print(f"  ✅ deterministic")


# ═══════════════════════════════════════════════════════════════════════════════
#  Run
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        ("single row",        test_single_row),
        ("single column",     test_single_column),
        ("square",            test_square),
        ("rectangular",       test_rectangular),
        ("multibatch",        test_multibatch),
        ("zeros in weights",  test_zeros_in_weights),
        ("all +1",            test_all_positive),
        ("all -1",            test_all_negative),
        ("deterministic",     test_deterministic),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} passed", end="")
    if failed:
        print(f", {failed} FAILED ❌")
        sys.exit(1)
    else:
        print(" ✅")
