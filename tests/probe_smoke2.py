"""Smoke for probe batch 2: compile BOTH variant .cu files (dx_unroll2,
fwd_uint4) plus their prod counterparts and check bit-exact parity at a tiny
shape (B=128, K=64, N=64). Catches load_inline/compile plumbing errors in
~2 min instead of after the full pytest gate.

Usage: python tests/probe_smoke2.py
"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CUH = os.path.join(REPO, "kernels/packed_ternary/packed_ternary.cuh")

with open(CUH) as f:
    CUH_SRC = f.read()


def build(cu_path, name, launch_fn, wrapper):
    from torch.utils.cpp_extension import load_inline
    with open(cu_path) as f:
        cu = f.read()
    combined = CUH_SRC + "\n" + cu.replace('#include "packed_ternary.cuh"', "")
    cpp = wrapper.replace("LAUNCH_FN", launch_fn)
    return load_inline(name=name, cpp_sources=cpp, cuda_sources=[combined],
                       verbose=False, extra_cuda_cflags=["-O3"])


CPP_DX = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
extern "C" { void LAUNCH_FN(const uint32_t* W, const void* dY, void* dX,
    int B, int K, int N, int stride, cudaStream_t s); }
torch::Tensor dx(torch::Tensor W, torch::Tensor dY, int64_t K) {
    auto dX = torch::empty({dY.size(0), K}, torch::dtype(torch::kFloat16).device(dY.device()));
    LAUNCH_FN(reinterpret_cast<const uint32_t*>(W.data_ptr<int32_t>()),
              dY.data_ptr<at::Half>(), dX.data_ptr<at::Half>(),
              dY.size(0), K, W.size(0), W.size(1), nullptr);
    return dX;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("dx", &dx); }
"""

CPP_FWD = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
extern "C" { void LAUNCH_FN(const uint32_t* W, const void* X, void* Y,
    int B, int K, int N, int stride, cudaStream_t s); }
torch::Tensor fwd(torch::Tensor W, torch::Tensor X) {
    auto Y = torch::empty({X.size(0), W.size(0)}, torch::dtype(torch::kFloat16).device(X.device()));
    LAUNCH_FN(reinterpret_cast<const uint32_t*>(W.data_ptr<int32_t>()),
              X.data_ptr<at::Half>(), Y.data_ptr<at::Half>(),
              X.size(0), X.size(1), W.size(0), W.size(1), nullptr);
    return Y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fwd", &fwd); }
"""


def main():
    if not torch.cuda.is_available():
        print("no cuda", flush=True)
        return 1
    dev = "cuda"
    torch.manual_seed(0)
    B, K, N = 128, 64, 64
    W = torch.randint(0, 4, (N, (K + 15) // 16), device=dev, dtype=torch.int32)
    W = torch.where(W == 3, 0, W)
    dY = torch.randn(B, N, device=dev, dtype=torch.float16)
    X = torch.randn(B, K, device=dev, dtype=torch.float16)

    dx_prod = build(os.path.join(REPO, "kernels/packed_ternary/gemm_backward_dx_tc.cu"),
                    "smoke2_dx_prod", "launch_packed_ternary_backward_dx_tc_64", CPP_DX)
    dx_var = build(os.path.join(HERE, "dx_unroll2.cu"),
                   "smoke2_dx_var", "probe_launch_dx_unroll2", CPP_DX)
    fwd_prod = build(os.path.join(REPO, "kernels/packed_ternary/gemm_forward_tc.cu"),
                     "smoke2_fwd_prod", "launch_packed_ternary_forward_tc_64", CPP_FWD)
    fwd_var = build(os.path.join(HERE, "fwd_uint4.cu"),
                    "smoke2_fwd_var", "probe_launch_fwd_uint4", CPP_FWD)
    print("all four variants built", flush=True)

    ok = torch.equal(dx_prod.dx(W, dY, K), dx_var.dx(W, dY, K))
    print(f"dx parity: {'EXACT' if ok else 'DIFF'}", flush=True)
    ok2 = torch.equal(fwd_prod.fwd(W, X), fwd_var.fwd(W, X))
    print(f"fwd parity: {'EXACT' if ok2 else 'DIFF'}", flush=True)
    print("SMOKE2:", "PASS" if (ok and ok2) else "FAIL", flush=True)
    return 0 if (ok and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
