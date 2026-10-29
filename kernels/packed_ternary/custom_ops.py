"""Register packed ternary kernels as PyTorch custom ops.

This makes them traceable by torch.compile and capturable by CUDAGraphs.
Without this, dynamo can't trace through load_inline extensions.

Usage:
    import kernels.packed_ternary.custom_ops  # registers all ops
    # Now torch.compile can trace through these ops
"""

import os
import torch
from torch.library import custom_op

from .cublas_backend import backward_dx as cublas_dx, forward as cublas_fwd, use_cublas

# Lazy-loaded kernel references (populated on first call)
_fwd_tc = None
_dx_tc = None
_dx_tc_64 = None
_update_tc_v2 = None
_update_tc_v3 = None
_fused_bwd = None
_loaded = False

def _ensure_loaded():
    global _fwd_tc, _dx_tc, _dx_tc_64, _update_tc_v2, _update_tc_v3, _fused_bwd, _loaded
    if _loaded:
        return
    from .pack_forward import has_tc, _load_tc_32, _forward_fn_tc
    from .pack_update import (
        _load_tc_if_needed, _HAS_DX_TC, _dx_tc_fn,
        _HAS_DX_TC_32, _dx_tc_32_fn,
        _HAS_UP_TC_V2_64, _up_tc_v2_64_fn,
        _HAS_UP_TC_V3_32, _up_tc_v3_32_fn,
    )

    if has_tc():
        _load_tc_32()
        from .pack_forward import _forward_fn_tc
        _fwd_tc = _forward_fn_tc

    _load_tc_if_needed()
    from . import pack_update as pu
    pu._load_up_tc_v2()  # 64x64 variant: NOT loaded by _load_tc_if_needed()
    pu._load_fused()
    _dx_tc = pu._dx_tc_32_fn if pu._HAS_DX_TC_32 else None
    _dx_tc_64 = pu._dx_tc_fn if pu._HAS_DX_TC else None
    # Tile-size-specific pointers — no shared aliasing (the load-order bug).
    _update_tc_v2 = pu._up_tc_v2_64_fn if pu._HAS_UP_TC_V2_64 else None
    _update_tc_v3 = pu._up_tc_v3_32_fn if pu._HAS_UP_TC_V3_32 else None
    _fused_bwd = pu._fused_fn if pu._HAS_FUSED else None
    _loaded = True


@custom_op("packed_ternary::forward_tc", mutates_args=())
def forward_tc(W: torch.Tensor, X: torch.Tensor, K: int) -> torch.Tensor:
    """Ternary forward GEMM: Y = X @ W^T via WMMA Tensor Cores.

    PT_BACKEND=cublas routes to decode-once + cuBLAS (measured 20.7x faster
    than the custom kernels at B=16384; decode costs 0.105 ms — Phase 0).
    """
    if use_cublas():
        return cublas_fwd(W, X, K)
    _ensure_loaded()
    if _fwd_tc is None:
        raise RuntimeError("TC forward kernel not available")
    return _fwd_tc(W.contiguous(), X.contiguous())


@forward_tc.register_fake
def _forward_tc_fake(W, X, K):
    B = X.size(0)
    N = W.size(0)
    return torch.empty(B, N, dtype=torch.float16, device=X.device)


@custom_op("packed_ternary::backward_dx_tc", mutates_args=())
def backward_dx_tc(W: torch.Tensor, dY: torch.Tensor, K: int) -> torch.Tensor:
    """Backward dX = dY @ W via WMMA Tensor Cores (auto 64x64/32x32).

    PT_BACKEND=cublas routes to decode-once + cuBLAS.
    """
    if use_cublas():
        return cublas_dx(W, dY, K)
    _ensure_loaded()
    B = dY.size(0)
    N_out = dY.size(1)
    if _dx_tc_64 is not None and B % 64 == 0 and N_out % 16 == 0 and K % 64 == 0:
        return _dx_tc_64(W.contiguous(), dY.contiguous(), K)
    if _dx_tc is None:
        raise RuntimeError("TC backward kernel not available")
    return _dx_tc(W.contiguous(), dY.contiguous(), K)


@backward_dx_tc.register_fake
def _backward_dx_tc_fake(W, dY, K):
    B = dY.size(0)
    return torch.empty(B, K, dtype=torch.float16, device=dY.device)


@custom_op("packed_ternary::update_tc_v2", mutates_args=("W", "counter", "flip_count"))
def update_tc_v2(W: torch.Tensor, counter: torch.Tensor,
                 X: torch.Tensor, dY: torch.Tensor, threshold: int,
                 dead_zone: float = 1e-3,
                 flip_count: torch.Tensor | None = None,
                 flip_budget: int = 0) -> None:
    """Fused gradient → counter → bit-flip (TC v2, vectorized counter).

    Dispatch goes through pack_update._select_update_fn — the load-order-
    independent selector that owns tile choice and lazy-loads the 64x64 /
    32x32 variants. The old body read _HAS_UP_TC_V2_64 right after
    _load_tc_if_needed(), which never loads the 64x64 build, so the flag was
    always False and every training backward raised
    "TC update v2 kernel not available".
    dead_zone: gradients with |dW| < dead_zone do not accumulate in the
    counter, breaking the limit cycle from a consistent residual gradient.
    flip_count: per-weight flip budget counter (int16 [N, K]); when provided
    and flip_budget > 0, a weight freezes after flip_budget flips.
    flip_budget: max flips per weight over a run (0 = off).
    """
    # Guard BEFORE backend dispatch: at flip_budget > 0 with flip_count=None
    # the cublas epilogue would OOB on its 1-element dummy — fail loudly on
    # both backends instead of silently mis-measuring.
    if flip_budget > 0:
        if flip_count is None:
            raise RuntimeError(
                "flip_budget > 0 requires a real flip_count tensor of shape [N, K]")
        if not (0 < int(flip_budget) <= 32767):
            raise RuntimeError(
                f"flip_budget must be in [1, 32767] (int16 counter), got {flip_budget}")
    if os.environ.get("PT_UPDATE_BACKEND", "custom").lower() == "cublas":
        # Ranked item #1: cuBLAS dW (fp32 out) + counter/flip epilogue.
        from .update_cublas import update_cublas
        update_cublas(W, counter, X, dY, threshold, dead_zone,
                      flip_count, flip_budget)
        return
    _ensure_loaded()
    from . import pack_update as pu
    fn, _name = pu._select_update_fn(X.size(0), dY.size(1), X.size(1))
    if fn is not None:
        if _name in ("v2_64", "v2_32"):
            # 8-param wrappers:
            # (W, counter, flip_count, X, dY, threshold, dead_zone, flip_budget)
            fc = flip_count
            if fc is None:
                # the wrapper takes data_ptr() unconditionally, but the kernel
                # only dereferences flip_count when flip_budget > 0 — a dummy
                # keeps flip_budget == 0 identical to pre-budget behavior
                fc = torch.empty((1,), dtype=torch.int16, device=W.device)
            fn(W, counter, fc, X.contiguous(), dY.contiguous(),
               int(threshold), float(dead_zone), int(flip_budget))
            return
        if flip_budget > 0:
            raise RuntimeError(
                f"flip_budget is only supported on v2_64/v2_32 dispatch, got {_name}")
        # v3_32 / v1_32 wrappers are 5-param: (W, counter, X, dY, threshold);
        # their kernels have no dead_zone/flip_budget parameters.
        fn(W, counter, X.contiguous(), dY.contiguous(), int(threshold))
        return
    pu._load_if_needed()
    if not pu._HAS_UP:
        raise RuntimeError("TC update v2 kernel not available")
    if flip_budget > 0:
        raise RuntimeError(
            "flip_budget requires a TC v2 kernel; the scalar fallback does not support it")
    pu._up_fn(W, counter, X.contiguous(), dY.contiguous(), int(threshold))


@custom_op("packed_ternary::backward_update_fused", mutates_args=("W", "counter"))
def backward_update_fused(W: torch.Tensor, counter: torch.Tensor,
                          dY: torch.Tensor, X: torch.Tensor,
                          in_features: int, threshold: int) -> torch.Tensor:
    """Fused backward dX + counter-based weight update (single launch).

    Returns dX for upstream gradient propagation.
    W and counter are updated in-place.
    """
    _ensure_loaded()
    if _fused_bwd is None:
        raise RuntimeError("Fused backward kernel not available")
    assert W.is_contiguous()
    assert counter.is_contiguous()
    dY_c = dY.contiguous()
    X_c = X.contiguous()
    dX = torch.zeros(dY.size(0), in_features, dtype=torch.float16, device=dY.device)
    _fused_bwd(dY_c, X_c, W, counter, in_features, int(threshold), dX)
    return dX


@backward_update_fused.register_fake
def _backward_update_fused_fake(W, counter, dY, X, in_features, threshold):
    B = dY.size(0)
    return torch.empty(B, in_features, dtype=torch.float16, device=dY.device)


@custom_op("packed_ternary::update_tc_v3", mutates_args=("W", "counter"))
def update_tc_v3(W: torch.Tensor, counter: torch.Tensor,
                 X: torch.Tensor, dY: torch.Tensor, threshold: int) -> None:
    """Fused gradient → magnitude-scaled counter → bit-flip (TC v3)."""
    _ensure_loaded()
    if _update_tc_v3 is None:
        raise RuntimeError("TC update v3 kernel not available")
    _update_tc_v3(W, counter, X.contiguous(), dY.contiguous(), int(threshold))


# ── Convenience: auto-dispatch wrappers ─────────────────────────────

TC_MIN_DIM = 16


def is_tc_available():
    """Check if TC kernels are available."""
    _ensure_loaded()
    return _fwd_tc is not None


def forward_auto(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Auto-dispatch forward: TC when dims allow, else scalar."""
    B, K = X.shape
    N = W.shape[0]
    if B >= TC_MIN_DIM and N >= TC_MIN_DIM and K >= TC_MIN_DIM and is_tc_available():
        return forward_tc(W, X, K)
    # Fallback to standard packed forward
    from .pack_forward import packed_ternary_forward
    return packed_ternary_forward(W, X)
