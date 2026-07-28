"""Fused backward (dX) + counter-based weight update (dW→sign→counter→flip).

Two operations:
    backward_dx(W, dY, in_features) -> dX      # W^T @ dY, needed upstream
    update(W, counter, X, dY, threshold)        # dW consumed, never stored

TC variants (WMMA Tensor Cores) are used when batch_size >= 16;
scalar kernels serve as fallback for smaller batches.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CUH_PATH = os.path.join(HERE, "packed_ternary.cuh")
DX_PATH  = os.path.join(HERE, "gemm_backward_dx.cu")
UP_PATH  = os.path.join(HERE, "gemm_update.cu")
DX_TC_PATH = os.path.join(HERE, "gemm_backward_dx_tc.cu")
UP_TC_PATH = os.path.join(HERE, "gemm_update_tc.cu")
UP_TC_V2_PATH = os.path.join(HERE, "gemm_update_tc_v2.cu")
UP_TC_V3_PATH = os.path.join(HERE, "gemm_update_tc_v3.cu")
FUSED_PATH = os.path.join(HERE, "gemm_fused_backward_update.cu")

# 32x32 tile variants (for dims 16-63, when 64x64 doesn't fit)
DX_TC_32_PATH = os.path.join(HERE, "gemm_backward_dx_tc_32.cu")
UP_TC_V2_32_PATH = os.path.join(HERE, "gemm_update_tc_v2_32.cu")
UP_TC_V3_32_PATH = os.path.join(HERE, "gemm_update_tc_v3_32.cu")

# ── Scalar kernels ─────────────────────────────────────────────────────

_HAS_DX = _HAS_UP = False
_dx_fn = _up_fn = None


def _load_dx():
    global _HAS_DX, _dx_fn
    if _HAS_DX:
        return
    try:
        from torch.utils.cpp_extension import load_inline
        with open(CUH_PATH) as f:
            cuh = f.read()
        with open(DX_PATH) as f:
            cu = f.read()
        combined = cuh + "\n" + cu.replace('#include "packed_ternary.cuh"', "")
        _lib = load_inline(
            name="packed_ternary_dx_ext",
            cpp_sources=r"""
            #include <cuda_runtime.h>
            #include <torch/extension.h>
            extern "C" {
                void launch_packed_ternary_backward_dx(
                    const uint32_t* W, const void* dY, void* dX,
                    int B, int K, int N, int stride, cudaStream_t s);
            }
            torch::Tensor dx_wrapper(torch::Tensor W, torch::Tensor dY, int K) {
                int B = dY.size(0);
                int N = dY.size(1);
                auto dX = torch::empty({B, K}, torch::dtype(torch::kFloat16).device(dY.device()));
                launch_packed_ternary_backward_dx(
                    reinterpret_cast<const uint32_t*>(W.data_ptr<int32_t>()),
                    dY.data_ptr<at::Half>(), dX.data_ptr<at::Half>(),
                    B, K, N, W.size(1), nullptr);
                return dX;
            }
            PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                m.def("backward_dx", &dx_wrapper, "dX = W^T @ dY");
            }
            """,
            cuda_sources=[combined], verbose=False, extra_cuda_cflags=["-O2"],
        )
        _dx_fn = _lib.backward_dx
        _HAS_DX = True
    except Exception as e:
        print(f"[dx] load failed: {e}")


def _load_up():
    global _HAS_UP, _up_fn
    if _HAS_UP:
        return
    try:
        from torch.utils.cpp_extension import load_inline
        with open(CUH_PATH) as f:
            cuh = f.read()
        with open(UP_PATH) as f:
            cu = f.read()
        combined = cuh + "\n" + cu.replace('#include "packed_ternary.cuh"', "")
        _lib = load_inline(
            name="packed_ternary_update_ext",
            cpp_sources=r"""
            #include <cuda_runtime.h>
            #include <torch/extension.h>
            extern "C" {
                void launch_packed_ternary_update(
                    const void* X, const void* dY, uint32_t* W, int16_t* counter,
                    int B, int K, int N, int stride, int16_t threshold, cudaStream_t s);
            }
            void update_wrapper(
                torch::Tensor W, torch::Tensor counter,
                torch::Tensor X, torch::Tensor dY, int16_t threshold)
            {
                launch_packed_ternary_update(
                    X.data_ptr<at::Half>(), dY.data_ptr<at::Half>(),
                    reinterpret_cast<uint32_t*>(W.data_ptr<int32_t>()),
                    counter.data_ptr<int16_t>(),
                    X.size(0), X.size(1), dY.size(1), W.size(1),
                    threshold, nullptr);
            }
            PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                m.def("update", &update_wrapper, "Fused dW->sign->counter->flip");
            }
            """,
            cuda_sources=[combined], verbose=False, extra_cuda_cflags=["-O2"],
        )
        _up_fn = _lib.update
        _HAS_UP = True
    except Exception as e:
        print(f"[up] load failed: {e}")


# ── Fused backward + update kernel ─────────────────────────────────────

_HAS_FUSED = False
_fused_fn = None


def _load_fused():
    global _HAS_FUSED, _fused_fn
    if _HAS_FUSED:
        return
    try:
        from torch.utils.cpp_extension import load_inline
        with open(CUH_PATH) as f:
            cuh = f.read()
        with open(FUSED_PATH) as f:
            cu = f.read()
        combined = cuh + "\n" + cu.replace('#include "packed_ternary.cuh"', "")
        _lib = load_inline(
            name="packed_ternary_fused_ext",
            cpp_sources=r"""
            #include <cuda_runtime.h>
            #include <torch/extension.h>
            extern "C" {
                void launch_packed_ternary_backward_update_fused(
                    const void* dY, const void* X, const uint32_t* W_read,
                    void* dX, uint32_t* W_mut, int16_t* counter,
                    int B, int K, int N, int stride_words,
                    int16_t threshold, cudaStream_t s);
            }
            void fused_wrapper(
                torch::Tensor dY, torch::Tensor X,
                torch::Tensor W, torch::Tensor counter,
                int in_features, int16_t threshold,
                torch::Tensor dX)
            {
                launch_packed_ternary_backward_update_fused(
                    dY.data_ptr<at::Half>(), X.data_ptr<at::Half>(),
                    reinterpret_cast<const uint32_t*>(W.data_ptr<int32_t>()),
                    dX.data_ptr<at::Half>(),
                    const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(W.data_ptr<int32_t>())),
                    counter.data_ptr<int16_t>(),
                    dY.size(0), in_features, dY.size(1), W.size(1),
                    threshold, nullptr);
            }
            PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                m.def("backward_update_fused", &fused_wrapper,
                      "Fused dX backward + counter update (WMMA)");
            }
            """,
            cuda_sources=[combined], verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
        _fused_fn = _lib.backward_update_fused
        _HAS_FUSED = True
    except Exception as e:
        print(f"[fused] load failed: {e}")


# ── TC (Tensor Core WMMA) kernels ──────────────────────────────────────

_HAS_DX_TC = _HAS_UP_TC = _HAS_UP_TC_V2 = _HAS_UP_TC_V3 = False
_HAS_DX_TC_32 = _HAS_UP_TC_V2_32 = _HAS_UP_TC_V3_32 = False
_dx_tc_fn = _up_tc_fn = _up_tc_v2_fn = _up_tc_v3_fn = None
_dx_tc_32_fn = _up_tc_v2_32_fn = _up_tc_v3_32_fn = None


def _load_dx_tc_32():
    """Load 32x32 backward dX TC (legacy, for dims 16-63)."""
    global _HAS_DX_TC_32, _dx_tc_32_fn
    if _HAS_DX_TC_32:
        return
    try:
        from torch.utils.cpp_extension import load_inline
        with open(CUH_PATH) as f:
            cuh = f.read()
        with open(DX_TC_32_PATH) as f:
            cu = f.read()
        combined = cuh + "\n" + cu.replace('#include "packed_ternary.cuh"', "")
        _lib = load_inline(
            name="packed_ternary_dx_tc_32_ext",
            cpp_sources=r"""
            #include <cuda_runtime.h>
            #include <torch/extension.h>
            extern "C" {
                void launch_packed_ternary_backward_dx_tc(
                    const uint32_t* W, const void* dY, void* dX,
                    int B, int K, int N, int stride, cudaStream_t s);
            }
            torch::Tensor dx_tc_32_wrapper(torch::Tensor W, torch::Tensor dY, int K) {
                int B = dY.size(0);
                int N = dY.size(1);
                auto dX = torch::empty({B, K}, torch::dtype(torch::kFloat16).device(dY.device()));
                launch_packed_ternary_backward_dx_tc(
                    reinterpret_cast<const uint32_t*>(W.data_ptr<int32_t>()),
                    dY.data_ptr<at::Half>(), dX.data_ptr<at::Half>(),
                    B, K, N, W.size(1), nullptr);
                return dX;
            }
            PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                m.def("backward_dx_tc_32", &dx_tc_32_wrapper, "dX = W^T @ dY (TC 32x32)");
            }
            """,
            cuda_sources=[combined], verbose=False, extra_cuda_cflags=["-O2"],
        )
        _dx_tc_32_fn = _lib.backward_dx_tc_32
        _HAS_DX_TC_32 = True
    except Exception as e:
        print(f"[dx_tc_32] load failed: {e}")
    global _HAS_DX_TC, _dx_tc_fn
    if _HAS_DX_TC:
        return
    try:
        from torch.utils.cpp_extension import load_inline
        with open(CUH_PATH) as f:
            cuh = f.read()
        with open(DX_TC_PATH) as f:
            cu = f.read()
        combined = cuh + "\n" + cu.replace('#include "packed_ternary.cuh"', "")
        _lib = load_inline(
            name="packed_ternary_dx_tc_ext",
            cpp_sources=r"""
            #include <cuda_runtime.h>
            #include <torch/extension.h>
            extern "C" {
                void launch_packed_ternary_backward_dx_tc(
                    const uint32_t* W, const void* dY, void* dX,
                    int B, int K, int N, int stride, cudaStream_t s);
            }
            torch::Tensor dx_tc_wrapper(torch::Tensor W, torch::Tensor dY, int K) {
                int B = dY.size(0);
                int N = dY.size(1);
                auto dX = torch::empty({B, K}, torch::dtype(torch::kFloat16).device(dY.device()));
                launch_packed_ternary_backward_dx_tc(
                    reinterpret_cast<const uint32_t*>(W.data_ptr<int32_t>()),
                    dY.data_ptr<at::Half>(), dX.data_ptr<at::Half>(),
                    B, K, N, W.size(1), nullptr);
                return dX;
            }
            PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                m.def("backward_dx_tc", &dx_tc_wrapper, "dX = W^T @ dY (TC)");
            }
            """,
            cuda_sources=[combined], verbose=False, extra_cuda_cflags=["-O2"],
        )
        _dx_tc_fn = _lib.backward_dx_tc
        _HAS_DX_TC = True
    except Exception as e:
        print(f"[dx_tc] load failed: {e}")


def _load_up_tc():
    global _HAS_UP_TC, _up_tc_fn
    if _HAS_UP_TC:
        return
    try:
        from torch.utils.cpp_extension import load_inline
        with open(CUH_PATH) as f:
            cuh = f.read()
        with open(UP_TC_PATH) as f:
            cu = f.read()
        combined = cuh + "\n" + cu.replace('#include "packed_ternary.cuh"', "")
        _lib = load_inline(
            name="packed_ternary_update_tc_ext",
            cpp_sources=r"""
            #include <cuda_runtime.h>
            #include <torch/extension.h>
            extern "C" {
                void launch_packed_ternary_update_tc(
                    const void* X, const void* dY, uint32_t* W, int16_t* counter,
                    int B, int K, int N, int stride, int16_t threshold, cudaStream_t s);
            }
            void update_tc_wrapper(
                torch::Tensor W, torch::Tensor counter,
                torch::Tensor X, torch::Tensor dY, int16_t threshold)
            {
                launch_packed_ternary_update_tc(
                    X.data_ptr<at::Half>(), dY.data_ptr<at::Half>(),
                    reinterpret_cast<uint32_t*>(W.data_ptr<int32_t>()),
                    counter.data_ptr<int16_t>(),
                    X.size(0), X.size(1), dY.size(1), W.size(1),
                    threshold, nullptr);
            }
            PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                m.def("update_tc", &update_tc_wrapper, "Fused dW->sign->counter->flip (TC)");
            }
            """,
            cuda_sources=[combined], verbose=False, extra_cuda_cflags=["-O2"],
        )
        _up_tc_fn = _lib.update_tc
        _HAS_UP_TC = True
    except Exception as e:
        print(f"[up_tc] load failed: {e}")


def _load_up_tc_v2():
    global _HAS_UP_TC_V2, _up_tc_v2_fn
    if _HAS_UP_TC_V2:
        return
    try:
        from torch.utils.cpp_extension import load_inline
        with open(CUH_PATH) as f:
            cuh = f.read()
        with open(UP_TC_V2_PATH) as f:
            cu = f.read()
        combined = cuh + "\n" + cu.replace('#include "packed_ternary.cuh"', "")
        _lib = load_inline(
            name="packed_ternary_update_tc_v2_ext",
            cpp_sources=r"""
            #include <cuda_runtime.h>
            #include <torch/extension.h>
            extern "C" {
                void launch_packed_ternary_update_tc_v2(
                    const void* X, const void* dY, uint32_t* W, int16_t* counter,
                    int B, int K, int N, int stride, int16_t threshold, cudaStream_t s);
            }
            void update_tc_v2_wrapper(
                torch::Tensor W, torch::Tensor counter,
                torch::Tensor X, torch::Tensor dY, int16_t threshold)
            {
                launch_packed_ternary_update_tc_v2(
                    X.data_ptr<at::Half>(), dY.data_ptr<at::Half>(),
                    reinterpret_cast<uint32_t*>(W.data_ptr<int32_t>()),
                    counter.data_ptr<int16_t>(),
                    X.size(0), X.size(1), dY.size(1), W.size(1),
                    threshold, nullptr);
            }
            PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                m.def("update_tc_v2", &update_tc_v2_wrapper,
                      "Optimized dW->sign->counter->flip (TC v2)");
            }
            """,
            cuda_sources=[combined], verbose=False, extra_cuda_cflags=["-O2", "--use_fast_math"],
        )
        _up_tc_v2_fn = _lib.update_tc_v2
        _HAS_UP_TC_V2 = True
    except Exception as e:
        print(f"[up_tc_v2] load failed: {e}")


def _load_up_tc_v3():
    global _HAS_UP_TC_V3, _up_tc_v3_fn
    if _HAS_UP_TC_V3:
        return
    try:
        from torch.utils.cpp_extension import load_inline
        with open(CUH_PATH) as f:
            cuh = f.read()
        with open(UP_TC_V3_PATH) as f:
            cu = f.read()
        combined = cuh + "\n" + cu.replace('#include "packed_ternary.cuh"', "")
        _lib = load_inline(
            name="packed_ternary_update_tc_v3_ext",
            cpp_sources=r"""
            #include <cuda_runtime.h>
            #include <torch/extension.h>
            extern "C" {
                void launch_packed_ternary_update_tc_v3(
                    const void* X, const void* dY, uint32_t* W, int16_t* counter,
                    int B, int K, int N, int stride, int16_t threshold, cudaStream_t s);
            }
            void update_tc_v3_wrapper(
                torch::Tensor W, torch::Tensor counter,
                torch::Tensor X, torch::Tensor dY, int16_t threshold)
            {
                launch_packed_ternary_update_tc_v3(
                    X.data_ptr<at::Half>(), dY.data_ptr<at::Half>(),
                    reinterpret_cast<uint32_t*>(W.data_ptr<int32_t>()),
                    counter.data_ptr<int16_t>(),
                    X.size(0), X.size(1), dY.size(1), W.size(1),
                    threshold, nullptr);
            }
            PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                m.def("update_tc_v3", &update_tc_v3_wrapper,
                      "Magnitude-scaled dW->counter->flip (TC v3)");
            }
            """,
            cuda_sources=[combined], verbose=False, extra_cuda_cflags=["-O2", "--use_fast_math"],
        )
        _up_tc_v3_fn = _lib.update_tc_v3
        _HAS_UP_TC_V3 = True
    except Exception as e:
        print(f"[up_tc_v3] load failed: {e}")


# ── Auto-dispatch public API ───────────────────────────────────────────

TC_MIN_DIM = 16  # WMMA m16n16k16 needs M, N, K all >= 16
TC_ALIGN = 16  # WMMA requires all dims to be multiples of 16 for correctness


def _tc_ok(dim: int) -> bool:
    """True if dim is valid for 32x32 TC dispatch."""
    return dim >= TC_MIN_DIM and (dim % TC_ALIGN) == 0


def _tc_ok_64(dim: int) -> bool:
    """True if dim is valid for 64x64 TC dispatch."""
    return dim >= 64 and (dim % 64) == 0


def _load_if_needed():
    """Ensure scalar kernels are loaded (lazy load on first use)."""
    if not _HAS_DX:
        _load_dx()
    if not _HAS_UP:
        _load_up()


def _load_tc_if_needed():
    """Ensure TC kernels are loaded (lazy load on first use)."""
    if not _HAS_DX_TC:
        _load_dx_tc()
    if not _HAS_UP_TC:
        _load_up_tc()
    if not _HAS_UP_TC_V2:
        _load_up_tc_v2()
    if not _HAS_UP_TC_V3:
        _load_up_tc_v3()


def backward_dx(W: torch.Tensor, dY: torch.Tensor, in_features: int) -> torch.Tensor:
    """dX = dY @ W  (gradient w.r.t. input).

    Auto-dispatches to TC (WMMA) when all GEMM dimensions >= 16,
    otherwise uses the scalar kernel.

    GEMM layout: [B, N_out] × [N_out, N_in] → [B, N_in]
      M = B (batch), K = N_out (dY cols), N = N_in (in_features)
    """
    B = dY.size(0)
    N_out = dY.size(1)   # K dimension of the GEMM

    # 64x64 TC: 4x faster for large dims
    if _tc_ok_64(B) and _tc_ok_64(N_out) and _tc_ok_64(in_features):
        _load_dx_tc()
        if _HAS_DX_TC:
            return _dx_tc_fn(W, dY, in_features)
    # 32x32 TC (legacy, for dims 16-63)
    if _tc_ok(B) and _tc_ok(N_out) and _tc_ok(in_features):
        _load_dx_tc_32()
        if _HAS_DX_TC_32:
            return _dx_tc_32_fn(W, dY, in_features)

    # Fallback to scalar
    _load_if_needed()
    if not _HAS_DX:
        raise RuntimeError("dX kernel not available")
    return _dx_fn(W, dY, in_features)


def update(W: torch.Tensor, counter: torch.Tensor, X: torch.Tensor,
           dY: torch.Tensor, threshold: int = 64):
    """Fused gradient -> counter -> bit-flip.  W is updated in-place.

    Auto-dispatches to TC (WMMA) when all GEMM dimensions >= 16,
    otherwise uses the scalar kernel.

    GEMM layout: [N_out, B] × [B, N_in] → [N_out, N_in]
      M = N_out (dY cols), K = B (batch), N = N_in (X cols)
    """
    B = X.size(0)
    N_out = dY.size(1)   # M dimension of dW
    N_in = X.size(1)     # N dimension of dW
    # CRITICAL: Do NOT call .contiguous() on W or counter — that can return
    # a COPY if the tensor is non-contiguous, and the kernel would modify
    # the copy silently.  Assert contiguity instead so we catch issues early.
    assert W.is_contiguous(), "W must be contiguous for in-place update"
    assert counter.is_contiguous(), "counter must be contiguous for in-place update"
    X = X.contiguous()  # fine — X is only read
    dY = dY.contiguous()  # fine — dY is only read

    # 64x64 TC (fast path for large dims)
    if _tc_ok_64(B) and _tc_ok_64(N_out) and _tc_ok_64(N_in):
        _load_up_tc_v2()
        if _HAS_UP_TC_V2:
            _up_tc_v2_fn(W, counter, X, dY, int(threshold))
            return
    # 32x32 TC (legacy, for dims 16-63)
    if _tc_ok(B) and _tc_ok(N_out) and _tc_ok(N_in):
        _load_tc_if_needed()
        # Prefer v3 (magnitude-scaled) > v2 (vectorized) > v1
        if _HAS_UP_TC_V3:
            _up_tc_v3_fn(W, counter, X, dY, int(threshold))
            return
        if _HAS_UP_TC_V2:
            _up_tc_v2_fn(W, counter, X, dY, int(threshold))
            return
        if _HAS_UP_TC:
            _up_tc_fn(W, counter, X, dY, int(threshold))
            return

    # Fallback to scalar
    _load_if_needed()
    if not _HAS_UP:
        raise RuntimeError("update kernel not available")
    _up_fn(W, counter, X, dY, int(threshold))


def backward_update(W: torch.Tensor, counter: torch.Tensor,
                    dY: torch.Tensor, X: torch.Tensor,
                    in_features: int, threshold: int = 64) -> torch.Tensor:
    """Fused backward (dX) + update (counter flip).  One Python call.

    Returns dX for upstream gradient.  W and counter are updated in-place.
    Shares .contiguous() calls for dY between backward and update.
    """
    B = dY.size(0)
    N_out = dY.size(1)

    # Make contiguous once (shared between backward and update)
    assert W.is_contiguous(), "W must be contiguous"
    assert counter.is_contiguous(), "counter must be contiguous"
    dY_c = dY.contiguous()
    X_c = X.contiguous()

    # ── Backward: dX = dY @ W ──
    # W is already asserted contiguous above — skip redundant .contiguous()
    if _tc_ok(B) and _tc_ok(N_out) and _tc_ok(in_features):
        _load_tc_if_needed()
        if _HAS_DX_TC:
            dX = _dx_tc_fn(W, dY_c, in_features)
        else:
            dX = _dx_fn(W, dY_c, in_features)
    else:
        _load_if_needed()
        dX = _dx_fn(W, dY_c, in_features)

    # ── Update: dW→sign→counter→flip ──
    N_in = X_c.size(1)
    if _tc_ok(B) and _tc_ok(N_out) and _tc_ok(N_in):
        _load_tc_if_needed()
        # Prefer v3 (magnitude-scaled) > v2 (vectorized) > v1
        if _HAS_UP_TC_V3:
            _up_tc_v3_fn(W, counter, X_c, dY_c, int(threshold))
        elif _HAS_UP_TC_V2:
            _up_tc_v2_fn(W, counter, X_c, dY_c, int(threshold))
        elif _HAS_UP_TC:
            _up_tc_fn(W, counter, X_c, dY_c, int(threshold))
    else:
        _load_if_needed()
        _up_fn(W, counter, X_c, dY_c, int(threshold))

    return dX


def backward_update_fused(
    W: torch.Tensor, counter: torch.Tensor,
    dY: torch.Tensor, X: torch.Tensor,
    in_features: int, threshold: int = 64) -> torch.Tensor:
    """Fused backward dX + weight update via single WMMA kernel.

    Replaces separate backward_dx() + update() calls with one launch.
    The kernel reads W and dY once from global memory, computes both
    dX and dW simultaneously, and applies counter/bit-flip in-place.

    Returns dX (gradient w.r.t. input) for upstream.
    W and counter are updated in-place.

    TC dimensions required: B >= 16, N >= 16, in_features >= 16.
    Falls back to backward_update() if dimensions are too small.
    """
    B = dY.size(0)
    N_out = dY.size(1)
    N_in = X.size(1)

    # Validate dimensions for WMMA fused kernel
    fused_ok = (B >= TC_MIN_DIM and N_out >= TC_MIN_DIM and N_in >= TC_MIN_DIM
                and in_features >= TC_MIN_DIM)

    if fused_ok:
        _load_fused()
        if _HAS_FUSED:
            assert W.is_contiguous(), "W must be contiguous for in-place update"
            assert counter.is_contiguous(), "counter must be contiguous"
            dY_c = dY.contiguous()
            X_c = X.contiguous()
            # dX must be pre-zeroed for atomic accumulation
            dX = torch.zeros(B, in_features, dtype=torch.float16, device=dY.device)
            _fused_fn(dY_c, X_c, W, counter, in_features, int(threshold), dX)
            return dX

    # Fallback to sequential backward_update
    return backward_update(W, counter, dY, X, in_features, threshold)


def init_counter(out_features: int, in_features: int) -> torch.Tensor:
    """Create a zeroed int16 counter tensor."""
    return torch.zeros(out_features, in_features, dtype=torch.int16, device="cuda")
