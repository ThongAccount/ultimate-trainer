#!/usr/bin/env python3
"""Build ALL CUDA kernels in a single load_inline invocation.

Compiles every .cu file in kernels/ as a separate compilation unit,
links them into one shared library, and returns a module with all
pybind11 bindings.

Usage:
    python build_kernels.py          # build & list available functions
    python build_kernels.py --bench  # build then run quick benchmark

Flags applied to nvcc:
    -O3                  max optimization
    --use_fast_math      fast intrinsics
    --threads=0          all CPU cores for parallel PTX→SASS
    -keep                cache intermediate .ptx/.cubin for incremental builds
"""

import os
import sys
import time

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")  # T4 — skip other archs

import torch
from torch.utils.cpp_extension import load_inline

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── All kernel sources ─────────────────────────────────────────────
# Each .cu is a separate compilation unit. Headers (.cuh) are included
# by the .cu files themselves — don't list them here.

KERNEL_CU_FILES = [
    # Packed ternary — forward
    "kernels/packed_ternary/gemm_forward.cu",
    "kernels/packed_ternary/gemm_forward_v2.cu",
    "kernels/packed_ternary/gemm_forward_v3.cu",
    "kernels/packed_ternary/gemm_forward_v4.cu",
    "kernels/packed_ternary/gemm_forward_tc.cu",
    # Packed ternary — backward
    "kernels/packed_ternary/gemm_backward_dx.cu",
    "kernels/packed_ternary/gemm_backward_dx_tc.cu",
    # Packed ternary — update
    "kernels/packed_ternary/gemm_update.cu",
    "kernels/packed_ternary/gemm_update_tc.cu",
    "kernels/packed_ternary/gemm_update_tc_v2.cu",
    # Elementwise
    "kernels/elementwise/addsub.cu",
    # Ternary matmul
    "kernels/ternary/ternary_matmul.cu",
    # Block sparse ternary
    "kernels/block_sparse_ternary/block_sparse_ternary.cu",
    # Fused ternary
    "kernels/fused_ternary/fused_ternary_kernel.cu",
    # Selective attention
    "kernels/selective_attn/selective_attn_kernel.cu",
    # Compressed attention
    "kernels/compressed_attn/compressed_attn_kernel.cu",
    # SubQSA combine
    "kernels/subqsa_combine/subqsa_combine_kernel.cu",
]

# ── C++ wrapper with pybind11 bindings ─────────────────────────────
# Declares every extern "C" launch function and wraps them for Python.

CPP_WRAPPER = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>

// ── Packed ternary forward ─────────────────────────────────────────
extern "C" {
    void launch_packed_ternary_forward(
        const uint32_t* W, const void* X, void* Y,
        int B, int K, int N, int stride, cudaStream_t s);
    void launch_packed_ternary_forward_v2(
        const uint32_t* W, const void* X, void* Y,
        int B, int K, int N, int stride, cudaStream_t s);
    void launch_packed_ternary_forward_v3(
        const uint32_t* W, const void* X, void* Y,
        int B, int K, int N, int stride, cudaStream_t s);
    void launch_packed_ternary_forward_v4(
        const uint32_t* W, const void* X, void* Y,
        int B, int K, int N, int stride, cudaStream_t s);
    void launch_packed_ternary_tc(
        const uint32_t* W, const void* X, void* Y,
        int B, int K, int N, int stride, cudaStream_t s);
}

// ── Packed ternary backward ────────────────────────────────────────
extern "C" {
    void launch_packed_ternary_backward_dx(
        const uint32_t* W, const void* dY, void* dX,
        int B, int K, int N, int stride, cudaStream_t s);
    void launch_packed_ternary_backward_dx_tc(
        const uint32_t* W, const void* dY, void* dX,
        int B, int K, int N, int stride, cudaStream_t s);
}

// ── Packed ternary update ──────────────────────────────────────────
extern "C" {
    void launch_packed_ternary_update(
        const void* X, const void* dY, uint32_t* W, int16_t* counter,
        int B, int K, int N, int stride, int16_t threshold, cudaStream_t s);
    void launch_packed_ternary_update_tc(
        const void* X, const void* dY, uint32_t* W, int16_t* counter,
        int B, int K, int N, int stride, int16_t threshold, cudaStream_t s);
    void launch_packed_ternary_update_tc_v2(
        const void* X, const void* dY, uint32_t* W, int16_t* counter,
        int B, int K, int N, int stride, int16_t threshold, cudaStream_t s);
}

// ── Forward wrappers ───────────────────────────────────────────────

#define FWD_WRAPPER(name, fn)                                               \
    torch::Tensor name(torch::Tensor W, torch::Tensor X, int K) {           \
        int B = X.size(0), N = W.size(0);                                   \
        auto Y = torch::empty({B, N},                                       \
            torch::dtype(torch::kFloat16).device(X.device()));               \
        fn(reinterpret_cast<const uint32_t*>(W.data_ptr<int32_t>()),         \
           X.data_ptr<at::Half>(), Y.data_ptr<at::Half>(),                   \
           B, K, N, W.size(1), nullptr);                                     \
        return Y;                                                            \
    }

FWD_WRAPPER(fwd_scalar,  launch_packed_ternary_forward)
FWD_WRAPPER(fwd_v2,      launch_packed_ternary_forward_v2)
FWD_WRAPPER(fwd_v3,      launch_packed_ternary_forward_v3)
FWD_WRAPPER(fwd_v4,      launch_packed_ternary_forward_v4)
FWD_WRAPPER(fwd_tc,      launch_packed_ternary_tc)

#undef FWD_WRAPPER

// ── Backward wrappers ──────────────────────────────────────────────

torch::Tensor dx_scalar(torch::Tensor W, torch::Tensor dY, int K) {
    int B = dY.size(0), N = dY.size(1);
    auto dX = torch::empty({B, K},
        torch::dtype(torch::kFloat16).device(dY.device()));
    launch_packed_ternary_backward_dx(
        reinterpret_cast<const uint32_t*>(W.data_ptr<int32_t>()),
        dY.data_ptr<at::Half>(), dX.data_ptr<at::Half>(),
        B, K, N, W.size(1), nullptr);
    return dX;
}

torch::Tensor dx_tc(torch::Tensor W, torch::Tensor dY, int K) {
    int B = dY.size(0), N = dY.size(1);
    auto dX = torch::empty({B, K},
        torch::dtype(torch::kFloat16).device(dY.device()));
    launch_packed_ternary_backward_dx_tc(
        reinterpret_cast<const uint32_t*>(W.data_ptr<int32_t>()),
        dY.data_ptr<at::Half>(), dX.data_ptr<at::Half>(),
        B, K, N, W.size(1), nullptr);
    return dX;
}

// ── Update wrappers ────────────────────────────────────────────────

#define UPD_WRAPPER(name, fn)                                               \
    void name(torch::Tensor W, torch::Tensor counter,                       \
              torch::Tensor X, torch::Tensor dY, int16_t threshold) {       \
        fn(X.data_ptr<at::Half>(), dY.data_ptr<at::Half>(),                 \
           reinterpret_cast<uint32_t*>(W.data_ptr<int32_t>()),               \
           counter.data_ptr<int16_t>(),                                      \
           X.size(0), X.size(1), dY.size(1), W.size(1),                     \
           threshold, nullptr);                                              \
    }

UPD_WRAPPER(upd_scalar,  launch_packed_ternary_update)
UPD_WRAPPER(upd_tc,       launch_packed_ternary_update_tc)
UPD_WRAPPER(upd_tc_v2,    launch_packed_ternary_update_tc_v2)

#undef UPD_WRAPPER

// ── Module ─────────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Forward
    m.def("forward_scalar",  &fwd_scalar,  "Ternary forward (scalar v1)");
    m.def("forward_v2",      &fwd_v2,      "Ternary forward (scalar v2)");
    m.def("forward_v3",      &fwd_v3,      "Ternary forward (scalar v3)");
    m.def("forward_v4",      &fwd_v4,      "Ternary forward (scalar v4)");
    m.def("forward_tc",      &fwd_tc,      "Ternary forward (WMMA TC)");
    // Backward
    m.def("backward_dx",     &dx_scalar,   "dX = dY @ W (scalar)");
    m.def("backward_dx_tc",  &dx_tc,       "dX = dY @ W (WMMA TC)");
    // Update
    m.def("update",          &upd_scalar,  "dW→sign→counter→flip (scalar)");
    m.def("update_tc",       &upd_tc,      "dW→sign→counter→flip (WMMA TC)");
    m.def("update_tc_v2",    &upd_tc_v2,   "dW→sign→counter→flip (TC v2, vectorized)");
}
"""


def build():
    """Compile all kernels in one nvcc invocation."""

    # Read file contents — load_inline expects source strings, not paths
    cuda_sources = []
    for rel in KERNEL_CU_FILES:
        abs_path = os.path.join(ROOT, rel)
        if not os.path.isfile(abs_path):
            print(f"  ⚠  Skipping missing: {rel}")
            continue
        with open(abs_path) as f:
            src = f.read()
        cuda_sources.append(src)

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Building {len(cuda_sources)} CUDA kernels (single invocation)    ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print(f"  nvcc flags: -O3 --use_fast_math --threads=0 -keep")
    print(f"  arch: sm_75 (T4)")
    print()

    t0 = time.time()

    lib = load_inline(
        name="ultimate_kernels",
        cpp_sources=CPP_WRAPPER,
        cuda_sources=cuda_sources,
        verbose=True,
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--threads=0",
            "-keep",
        ],
        extra_include_paths=[
            os.path.join(ROOT, "kernels", "packed_ternary"),
            os.path.join(ROOT, "kernels"),
        ],
    )

    dt = time.time() - t0
    print(f"\n  ✅ Built in {dt:.1f}s")
    return lib


def list_functions(lib):
    """Print all available functions."""
    print(f"\n  Available functions:")
    for name in sorted(dir(lib)):
        if name.startswith("_"):
            continue
        fn = getattr(lib, name)
        if callable(fn):
            doc = getattr(fn, "__doc__", "") or ""
            print(f"    lib.{name:<20s}  {doc}")


if __name__ == "__main__":
    lib = build()
    list_functions(lib)

    if "--bench" in sys.argv:
        print("\n  Running quick smoke test...")
        B, K, N = 32, 1024, 1024
        W = torch.randint(0, 3, (N, K // 16), dtype=torch.int32, device="cuda")
        X = torch.randn(B, K, dtype=torch.float16, device="cuda")
        Y = lib.forward_tc(W, X, K)
        print(f"    forward_tc: X {list(X.shape)} × W {list(W.shape)} → Y {list(Y.shape)} ✅")
