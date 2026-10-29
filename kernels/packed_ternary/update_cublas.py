"""Update re-hosted on cuBLAS (ranked item #1, 2026-10-20).

PT_UPDATE_BACKEND=cublas routes custom_ops.update_tc_v2 here:
    dW32 = dY^T @ X          # cublasGemmEx, fp16 in / fp32 out (no rounding)
    epilogue: counter/flip   # exact gemm_update_tc_v2.cu semantics

fp32-out dW is the parity requirement: torch.matmul on fp16 inputs returns
fp16, which rounds dW and can flip near-zero signs.

Env:
    PT_UPDATE_BACKEND=cublas|custom (default custom — existing kernel path)

Dead zone:
    DEAD_ZONE_DEFAULT = 1e-3  (pre-registered 2026-10-28; counters only
    accumulate |dW| >= dead_zone, breaking the limit cycle from a consistent
    residual gradient.)
"""
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CUH_PATH = os.path.join(HERE, "packed_ternary.cuh")
CU_PATH = os.path.join(HERE, "gemm_update_cublas.cu")

# Pre-registered 2026-10-28: counter dead-zone default. |dW| below this does
# not accumulate, so a consistent residual gradient can no longer drive the
# counter to threshold every ~T steps. Sensitivity values 1e-4 / 1e-2 to be
# swept on T4 if the default does not break the limit cycle.
DEAD_ZONE_DEFAULT = 1e-3

_HAS_UPDATE_CUBLAS = False
_dw_gemm_f32 = None
_update_epilogue = None


def _load_update_cublas():
    global _HAS_UPDATE_CUBLAS, _dw_gemm_f32, _update_epilogue
    if _HAS_UPDATE_CUBLAS:
        return
    try:
        from torch.utils.cpp_extension import load_inline
        with open(CUH_PATH) as f:
            cuh = f.read()
        with open(CU_PATH) as f:
            cu = f.read()
        combined = cuh + "\n" + cu.replace('#include "packed_ternary.cuh"', "")
        _lib = load_inline(
            name="packed_ternary_update_cublas_ext",
            cpp_sources=r"""
            #include <cuda_runtime.h>
            #include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
extern "C" {
    void packed_ternary_dw_gemm_f32(
        const void* dY, const void* X, void* dW,
        int B, int N, int K, cudaStream_t stream);
    void launch_update_epilogue(
        void* W, void* counter, void* flip_count, const void* dW,
        int N, int K, int stride_words, int64_t threshold, float dead_zone,
        int16_t flip_budget, cudaStream_t stream);
}

void dw_gemm_f32(torch::Tensor dY, torch::Tensor X, torch::Tensor dW) {
    int B = dY.size(0), N = dY.size(1), K = X.size(1);
    packed_ternary_dw_gemm_f32(
        dY.data_ptr<at::Half>(), X.data_ptr<at::Half>(),
        dW.data_ptr<float>(),
        B, N, K, c10::cuda::getCurrentCUDAStream());
}

void update_epilogue(torch::Tensor W, torch::Tensor counter,
                     torch::Tensor flip_count, torch::Tensor dW,
                     int64_t threshold, float dead_zone, int16_t flip_budget) {
    int N = W.size(0), K = dW.size(1);
    int stride_words = W.size(1);
    launch_update_epilogue(
        W.data_ptr<int32_t>(), counter.data_ptr<int16_t>(),
        flip_count.data_ptr<int16_t>(), dW.data_ptr<float>(),
        N, K, stride_words, (int16_t)threshold, dead_zone, flip_budget,
        c10::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dw_gemm_f32", &dw_gemm_f32, "dW = dY^T @ X, fp16-in/fp32-out cuBLAS");
    m.def("update_epilogue", &update_epilogue, "counter/flip epilogue (v2_64 semantics)");
}
            """,
            cuda_sources=[combined], verbose=False,
            extra_cuda_cflags=["-O2"],
        )
        _dw_gemm_f32 = _lib.dw_gemm_f32
        _update_epilogue = _lib.update_epilogue
        _HAS_UPDATE_CUBLAS = True
    except Exception as e:
        print(f"[update_cublas] load failed: {e}")


def use_update_cublas() -> bool:
    return os.environ.get("PT_UPDATE_BACKEND", "custom").lower() == "cublas"


def update_cublas(W: torch.Tensor, counter: torch.Tensor,
                  X: torch.Tensor, dY: torch.Tensor, threshold: int = 32,
                  dead_zone: float = DEAD_ZONE_DEFAULT,
                  flip_count: torch.Tensor | None = None,
                  flip_budget: int = 0):
    """Re-hosted update: cuBLAS dW (fp32 out) + counter/flip epilogue.

    Same in-place semantics as pack_update.update: W and counter are mutated.
    dead_zone: gradients with |dW| < dead_zone do not accumulate in the
    counter, breaking the limit cycle from a consistent residual gradient.
    flip_count: per-weight flip budget counter (int16 [N, K]); when provided
    and flip_budget > 0, a weight freezes after flip_budget flips.
    flip_budget: max flips per weight over a run (0 = off).
    """
    _load_update_cublas()
    if not _HAS_UPDATE_CUBLAS:
        raise RuntimeError("update cublas backend not available")
    assert W.is_contiguous(), "W must be contiguous for in-place update"
    assert counter.is_contiguous(), "counter must be contiguous"
    if flip_count is not None:
        assert flip_count.is_contiguous(), "flip_count must be contiguous"
    if not (0 <= int(threshold) <= 32767):
        raise RuntimeError(f"threshold must be in [0, 32767], got {threshold}")
    if not (0.0 <= float(dead_zone) <= 1.0):
        raise RuntimeError(f"dead_zone must be in [0, 1], got {dead_zone}")
    if not (0 <= int(flip_budget) <= 32767):
        raise RuntimeError(f"flip_budget must be in [0, 32767], got {flip_budget}")
    if flip_budget > 0 and flip_count is None:
        raise RuntimeError(
            "flip_budget > 0 requires a real flip_count tensor of shape [N, K] "
            "(the 1-element dummy below is only safe at flip_budget == 0, where "
            "the epilogue guards every flip_count deref with flip_budget > 0)")
    X = X.contiguous()
    dY = dY.contiguous()
    N, K = W.size(0), X.size(1)
    dW = torch.empty((N, K), dtype=torch.float32, device=X.device)
    _dw_gemm_f32(dY, X, dW)
    if flip_count is None:
        flip_count = torch.empty((1,), dtype=torch.int16, device=X.device)
    _update_epilogue(W, counter, flip_count, dW, int(threshold),
                     float(dead_zone), int(flip_budget))
    return dW
