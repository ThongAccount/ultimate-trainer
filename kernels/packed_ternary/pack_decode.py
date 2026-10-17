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
_DECODE_MOD = None
_HAS_DECODE = False

_CU = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

torch::Tensor decode_wrapper(torch::Tensor W_packed, int N, int K);
void decode_into_wrapper(torch::Tensor W_packed, torch::Tensor W16, int N, int K);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode", &decode_wrapper, "packed ternary -> fp16 decode (allocating)");
    m.def("decode_into", &decode_into_wrapper, "packed ternary -> fp16 decode into existing buffer");
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

void decode_into_launcher(
    const uint32_t* W, __half* W16,
    int N, int K, int stride_words, cudaStream_t stream)
{
    int nk = N * K;
    int threads = 256;
    int blocks = (nk + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    decode_kernel<<<blocks, threads, 0, stream>>>(W, W16, N, K, stride_words);
}

// Allocating form (bench/probe convenience)
torch::Tensor decode_wrapper(torch::Tensor W_packed, int N, int K) {
    TORCH_CHECK(W_packed.is_cuda(), "W_packed must be CUDA");
    auto W16 = torch::empty({N, K},
                            torch::dtype(torch::kFloat16).device(W_packed.device()));
    decode_into_launcher(
        reinterpret_cast<const uint32_t*>(W_packed.data_ptr<int32_t>()),
        reinterpret_cast<__half*>(W16.data_ptr<at::Half>()),
        N, K, W_packed.size(1),
        c10::cuda::getCurrentCUDAStream());
    return W16;
}

// Persistent-buffer form: graph-capturable (stable pointers across replays)
void decode_into_wrapper(
    torch::Tensor W_packed, torch::Tensor W16, int N, int K)
{
    TORCH_CHECK(W_packed.is_cuda() && W16.is_cuda(), "must be CUDA");
    TORCH_CHECK(W16.dtype() == torch::kFloat16, "W16 must be fp16");
    TORCH_CHECK(W16.size(0) == N && W16.size(1) == K, "W16 shape mismatch");
    decode_into_launcher(
        reinterpret_cast<const uint32_t*>(W_packed.data_ptr<int32_t>()),
        reinterpret_cast<__half*>(W16.data_ptr<at::Half>()),
        N, K, W_packed.size(1),
        c10::cuda::getCurrentCUDAStream());
}
"""


def load_decode():
    """Compile/return the decode extension module (idempotent).

    Always returns the MODULE (has both .decode and .decode_into) — never a
    bare function — so callers can't end up calling an object that lacks one.
    """
    global _DECODE_MOD, _HAS_DECODE
    if _DECODE_MOD is not None:
        return _DECODE_MOD
    from torch.utils.cpp_extension import load_inline

    lib = load_inline(
        name="pt_decode_standalone",
        cpp_sources=_CU,            # cpp carries the kernel decl + PYBIND11 module
        cuda_sources=[_CU_KERNEL],  # .cu carries the kernel + launcher
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )
    _DECODE_MOD = lib
    _DECODE_FN = lib.decode
    _HAS_DECODE = True
    return _DECODE_MOD


def decode_packed(W_packed: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Decode packed [N, stride_words] int32 -> fp16 [N, K] on GPU (allocating)."""
    return load_decode().decode(W_packed, N, K)


def decode_into(W_packed: torch.Tensor, W16: torch.Tensor, N: int, K: int) -> None:
    """Decode into an existing fp16 [N, K] buffer (graph-capturable)."""
    load_decode().decode_into(W_packed, W16, N, K)
