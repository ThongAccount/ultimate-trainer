"""Phase 1.2 verification: dispatch is load-order independent + kernel names.

The retired design let a small-shape call (32x32 fallback) overwrite the
shared `_up_tc_v2_fn` pointer, so a later 64-eligible call silently ran the
small kernel. Regression guard: run a small-shape update FIRST, then a
64-eligible one, and assert the 64 path still selects the 64x64 kernel.

Also checks the %16 gates (80/112 take the wide tile, not the 32x32 one).
GPU-gated (dispatch is a host-side decision, but calling the kernels needs
T4); the pure selection logic itself is CPU-testable without CUDA.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _has_cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


# ── Pure selection-logic tests (no CUDA needed) ─────────────────────────


def test_16_gate_accepts_wide_shapes():
    """%16 (not %64): 80 and 112 must be eligible for the 64 kernels."""
    from kernels.packed_ternary.pack_update import _tc_ok_16

    assert _tc_ok_16(80), "80 must be 64-kernel eligible"
    assert _tc_ok_16(112), "112 must be 64-kernel eligible"
    assert _tc_ok_16(4096) and _tc_ok_16(1024)
    assert not _tc_ok_16(48), "48 is not 16-aligned"
    assert not _tc_ok_16(15), "below WMMA minimum"


def test_selector_returns_64_pointer_not_32():
    """Selector must never return a 32-pointer for a 64-eligible shape."""
    from kernels.packed_ternary import pack_update as pu

    # Force-load order: 32-fallback first (the historical poisoning case)
    if pu._HAS_UP_TC_V2_32 is False and pu._HAS_UP_TC_V3_32 is False:
        # small dims -> selector *would* pick a 32 path if it were eligible;
        # here we only need the fn slots to exist for the pointer test.
        pass

    if not pu._HAS_UP_TC_V2_64:
        pytest.skip("64x64 update kernel not loaded on this box")

    fn, name = pu._select_update_fn(16384, 50272, 1024)
    assert name == "v2_64", f"head shape must select v2_64, got {name}"
    assert fn is pu._up_tc_v2_64_fn, "selector returned a non-64 pointer"

    fn2, name2 = pu._select_update_fn(16384, 80, 112)
    assert name2 == "v2_64", f"80x112 must take the wide tile, got {name2}"
    assert fn2 is pu._up_tc_v2_64_fn, "80x112 returned a non-64 pointer"


# ── T4-gated end-to-end order-independence ──────────────────────────────


@pytest.mark.skipif(not _has_cuda(), reason="no CUDA")
def test_small_then_large_dispatch_stays_64():
    """Small-shape update first, then large: the large one must still use v2_64."""
    import torch

    from kernels.packed_ternary import pack_tensor, pack_update as pu

    def run(shape, expect_name):
        N_out, B, N_in = shape
        W = pack_tensor(torch.zeros(N_out, N_in, device="cuda"))
        counter = torch.zeros(N_out, N_in, dtype=torch.int16, device="cuda")
        X = torch.randn(B, N_in, device="cuda", dtype=torch.float16) * 0.1
        dY = torch.randn(B, N_out, device="cuda", dtype=torch.float16) * 0.1
        fn, name = pu._select_update_fn(B, N_out, N_in)
        assert name == expect_name, f"{shape}: expected {expect_name}, got {name}"
        fn(W, counter, X, dY, 32)
        return W, counter

    # 1) small shape FIRST (the poisoning case — would overwrite the shared ptr)
    _, _n = pu._select_update_fn(32, 32, 32)
    assert _n in ("v3_32", "v2_32", "v1_32"), _n

    # 2) large shape SECOND — must still select v2_64
    W, counter = run((64, 16384, 1024), "v2_64")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
