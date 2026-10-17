"""Standalone packed-ternary -> fp16 decode (Phase 0/2).

The unpack_tensor() in kernels/packed_ternary/__init__.py is a triple-nested
Python CPU loop (checkpoint/debug only — unusable for benchmarking or the
cuBLAS backend). The kernels decode inline per-tile; for the pre-decoded
"decode once then cuBLAS" arm we need a fast GPU decode. This is that.

Layout: 16 ternary values per uint32, code 01=+1, 10=-1, 00/11=0
(kernels/packed_ternary/packed_ternary.cuh). Output is [N, K] row-major fp16.
"""
import os

import torch

_DECODE_FN = None
_HAS_DECODE = False

_CU = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

torch::Tensor decode_wrapper(torch::Tensor W_packed, int N, int K);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode", &decode_wrapper, "packed ternary -> fp16 decode");
}
"""

_CU_KERNEL = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

__device__ __forceinline__ int8_t d_decode(uint32_t code2) {
    // 00 -> 0, 01 -> +1, 10 -> -1, 11 -> 0 (INVALID sentinel -> 0)
    return (int8_t)(((int)(code2 & 1)) - ((int)(code2 >> 1) & 1));
}

__global__ __launch_bounds__(256) void decode_kernel(
    const uint32_t* __restrict__ W, __half* __restrict__ W16,
    int N, int K, int stride_words)
{
    // one thread per OUTPUT element; each thread decodes its own 2-bit field
    int nk = N * K;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nk;
         idx += gridDim.x * blockDim.x) {
        int n = idx / K;
        int k = idx % K;
        int w = W[n * stride_words + (k >> 4)];
        uint32_t code = (w >> ((k & 15) * 2)) & 3u;
        W16[idx] = __float2half((float)d_decode(code));
    }
}

torch::Tensor decode_wrapper(torch::Tensor W_packed, int N, int K) {
    TORCH_CHECK(W_packed.is_cuda(), "W_packed must be CUDA");
    auto W16 = torch::empty({N, K},
                            torch::dtype(torch::kFloat16).device(W_packed.device()));
    int stride_words = W_packed.size(1);
    int nk = N * K;
    int threads = 256;
    int blocks = (nk + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    decode_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint32_t*>(W_packed.data_ptr<int32_t>()),
        reinterpret_cast<__half*>(W16.data_ptr<at::Half>()),
        N, K, stride_words);
    return W16;
}
"""


def load_decode():
    """Compile/return the decode module (idempotent)."""
    global _DECODE_FN, _HAS_DECODE
    if _HAS_DECODE:
        return _DECODE_FN
    from torch.utils.cpp_extension import load_inline

    lib = load_inline(
        name="pt_decode_standalone",
        cpp_sources=_CU,            # cpp carries the kernel decl + PYBIND11 module
        cuda_sources=[_CU_KERNEL],  # .cu carries the kernel + launcher
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )
    _DECODE_FN = lib.decode
    _HAS_DECODE = True
    return _DECODE_FN


def decode_packed(W_packed: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Decode packed [N, stride_words] int32 -> fp16 [N, K] on GPU."""
    return load_decode()(W_packed, N, K)
