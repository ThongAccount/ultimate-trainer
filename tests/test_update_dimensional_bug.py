"""Test update_tc_v2 correctness against torch reference.

This test specifically checks if the dimensional bug in update_tc_v2 causes
numerical errors by comparing against a ground-truth implementation.

pytest-safe (C13 fix): the original script-style body ran at import time and
called sys.exit(0/1) — pytest would terminate the whole run at collection.
Body is now wrapped in test_update_dimensional_bug(); __main__ preserves the
standalone `python3 tests/test_update_dimensional_bug.py` behavior.
"""
import os
import sys

import pytest

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")


def _has_cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def test_update_dimensional_bug():
    if not _has_cuda():
        pytest.skip("no CUDA on this box")
    import torch

    torch.manual_seed(0)
    sys.path.insert(0, os.getcwd())

    # Test dimensions: Must trigger the bug (N, K >= 64)
    B, N, K = 32, 128, 128
    threshold = 32

    print(f"Testing update_tc_v2 correctness: B={B}, N={N}, K={K}")

    X = torch.randn(B, K, device="cuda", dtype=torch.float16)
    dY = torch.randn(B, N, device="cuda", dtype=torch.float16)

    # Reference gradient: dW = dY^T @ X
    dW_ref = (dY.T.float() @ X.float()).half()  # [N, K]

    kWeightsPerWord = 16
    stride_words = (K + kWeightsPerWord - 1) // kWeightsPerWord
    W_packed = torch.zeros(N, stride_words, dtype=torch.int32, device="cuda")
    counter_ref = torch.zeros(N, K, dtype=torch.int16, device="cuda")
    counter_kernel = torch.zeros(N, K, dtype=torch.int16, device="cuda")

    # Reference update: counter -= sign(dW)
    signs = torch.sign(dW_ref.float()).to(torch.int16)
    counter_ref = (counter_ref - signs).to(torch.int16)

    # Kernel update (production v2_32, loaded directly — bypasses the
    # small-dims v3 dispatch trap this suite previously suffered from)
    from kernels.packed_ternary.pack_update import _load_up_tc_v2_32

    _load_up_tc_v2_32()

    from kernels.packed_ternary import pack_update as pu

    if not pu._HAS_UP_TC_V2_32:
        pytest.fail("CUDA present but TC32 v2 update kernel failed to load")

    W_packed_test = W_packed.clone()
    pu._up_tc_v2_32_fn(W_packed_test, counter_kernel, X, dY, threshold)

    # Compare
    diff = (counter_kernel - counter_ref).abs()
    max_diff = diff.max().item()
    num_errors = (diff > 0).sum().item()
    error_rate = num_errors / (N * K) * 100

    print(f"\n{'=' * 60}")
    print(f"UPDATE_TC_V2 CORRECTNESS TEST")
    print(f"{'=' * 60}")
    print(f"Dimensions: B={B}, N={N}, K={K}")
    print(f"Reference counter changes: {(counter_ref != 0).sum().item()} / {N*K}")
    print(f"Kernel counter changes:    {(counter_kernel != 0).sum().item()} / {N*K}")
    print(f"Max difference: {max_diff}")
    print(f"Error count: {num_errors} / {N*K} ({error_rate:.2f}%)")
    print(f"{'=' * 60}")

    assert max_diff == 0, (
        f"update_tc_v2 dimensional/counter mismatch: {num_errors} errors "
        f"({error_rate:.2f}%), max diff {max_diff}. Sample: "
        f"{[(int(n), int(k), int(counter_ref[n, k]), int(counter_kernel[n, k])) for n, k in torch.nonzero(diff > 0)[:10]]}"
    )


if __name__ == "__main__":
    # Standalone behavior preserved: exit code reflects pass/fail.
    try:
        test_update_dimensional_bug()
    except pytest.skip.Exception:  # Skipped -> treat as neutral exit
        sys.exit(0)
    print("PASS — Kernel matches reference exactly")
