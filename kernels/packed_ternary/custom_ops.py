"""Register packed ternary kernels as PyTorch custom ops.

This makes them traceable by torch.compile and capturable by CUDAGraphs.
Without this, dynamo can't trace through load_inline extensions.

Usage:
    import kernels.packed_ternary.custom_ops  # registers all ops
    # Now torch.compile can trace through these ops
"""

import torch
from torch.library import custom_op

# Lazy-loaded kernel references (populated on first call)
_fwd_tc = None
_dx_tc = None
_update_tc_v2 = None
_update_tc_v3 = None
_fused_bwd = None
_loaded = False


def _ensure_loaded():
    global _fwd_tc, _dx_tc, _update_tc_v2, _update_tc_v3, _fused_bwd, _loaded
    if _loaded:
        return
    from .pack_forward import has_tc, _load_tc_32, _forward_fn_tc
    from .pack_update import _load_tc_if_needed, _HAS_DX_TC, _dx_tc_fn, _HAS_UP_TC_V2, _up_tc_v2_fn, _HAS_UP_TC_V3, _up_tc_v3_fn

    if has_tc():
        _load_tc()
        from .pack_forward import _forward_fn_tc
        _fwd_tc = _forward_fn_tc

    _load_tc_if_needed()
    from . import pack_update as pu
    pu._load_fused()
    _dx_tc = pu._dx_tc_fn if pu._HAS_DX_TC else None
    _update_tc_v2 = pu._up_tc_v2_fn if pu._HAS_UP_TC_V2 else None
    _update_tc_v3 = pu._up_tc_v3_fn if pu._HAS_UP_TC_V3 else None
    _fused_bwd = pu._fused_fn if pu._HAS_FUSED else None
    _loaded = True


@custom_op("packed_ternary::forward_tc", mutates_args=())
def forward_tc(W: torch.Tensor, X: torch.Tensor, K: int) -> torch.Tensor:
    """Ternary forward GEMM: Y = X @ W^T via WMMA Tensor Cores."""
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
    """Backward dX = dY @ W via WMMA Tensor Cores."""
    _ensure_loaded()
    if _dx_tc is None:
        raise RuntimeError("TC backward kernel not available")
    return _dx_tc(W.contiguous(), dY.contiguous(), K)


@backward_dx_tc.register_fake
def _backward_dx_tc_fake(W, dY, K):
    B = dY.size(0)
    return torch.empty(B, K, dtype=torch.float16, device=dY.device)


@custom_op("packed_ternary::update_tc_v2", mutates_args=("W", "counter"))
def update_tc_v2(W: torch.Tensor, counter: torch.Tensor,
                 X: torch.Tensor, dY: torch.Tensor, threshold: int) -> None:
    """Fused gradient → counter → bit-flip (TC v2, vectorized counter)."""
    _ensure_loaded()
    if _update_tc_v2 is None:
        raise RuntimeError("TC update v2 kernel not available")
    _update_tc_v2(W, counter, X.contiguous(), dY.contiguous(), int(threshold))


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
