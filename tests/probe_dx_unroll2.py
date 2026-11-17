"""Probe #10: dX 64x64 kernel with kK2=64 (4 K-slices x 4 frags = 16 MMAs/sync).

Compiles prod and variant as SEPARATE extensions (both .cu files define the
same globals — one TU would be a redefinition error), checks parity on fc1
and head shapes, then interleaved A/B via bench_protocol.ab_median (null arm
first). No production dispatch touched.

Hypothesis (prereg 2026-11-03-probe-batch-2): fewer sync windows per N-tile,
same MMA count — expected -3..-8% on head dX if sync/ldg overhead dominates.
Ship bar: delta <= -max(2*sigma_null, 1.0)% on BOTH shapes, bit-exact parity.

Usage: python tests/probe_dx_unroll2.py
"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CUH = os.path.join(REPO, "kernels/packed_ternary/packed_ternary.cuh")
PROD_CU = os.path.join(REPO, "kernels/packed_ternary/gemm_backward_dx_tc.cu")
VAR_CU = os.path.join(HERE, "dx_unroll2.cu")

with open(CUH) as f:
    CUH_SRC = f.read()


def make_cpp(launch_fn, fn_name):
    return r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
extern "C" {
    void LAUNCH_FN(
        const uint32_t* W, const void* dY, void* dX,
        int batch_size, int in_features, int out_features,
        int stride_words, cudaStream_t stream);
}
torch::Tensor FN_NAME(torch::Tensor W, torch::Tensor dY, int64_t K) {
    auto dX = torch::empty({dY.size(0), K}, torch::dtype(torch::kFloat16).device(dY.device()));
    LAUNCH_FN(
        reinterpret_cast<const uint32_t*>(W.data_ptr<int32_t>()),
        dY.data_ptr<at::Half>(), dX.data_ptr<at::Half>(),
        dY.size(0), K, W.size(0), W.size(1), nullptr);
    return dX;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("FN_NAME", &FN_NAME); }
""".replace("LAUNCH_FN", launch_fn).replace("FN_NAME", fn_name)


def build(cu_path, name, launch_fn, fn_name):
    from torch.utils.cpp_extension import load_inline
    with open(cu_path) as f:
        cu = f.read()
    combined = CUH_SRC + "\n" + cu.replace('#include "packed_ternary.cuh"', "")
    return load_inline(name=name, cpp_sources=make_cpp(launch_fn, fn_name),
                       cuda_sources=[combined], verbose=False,
                       extra_cuda_cflags=["-O3", "--use_fast_math"])


def make_W(N, K, dev):
    W = torch.randint(0, 4, (N, (K + 15) // 16), device=dev, dtype=torch.int32)
    return torch.where(W == 3, 0, W)


def main():
    from bench_protocol import ab_median, report_line
    dev = "cuda"
    torch.manual_seed(0)
    B = 16352
    print("building prod + unroll2 variants...", flush=True)
    prod = build(PROD_CU, "probe_dx_prod_ext",
                 "launch_packed_ternary_backward_dx_tc_64", "dx")
    var = build(VAR_CU, "probe_dx_unroll2_ext",
                "probe_launch_dx_unroll2", "dx")
    print("built.", flush=True)

    ok = True
    for name, K, N in [("fc1", 1024, 4096), ("head", 1024, 50272)]:
        dY = torch.randn(B, N, device=dev, dtype=torch.float16)
        W = make_W(N, K, dev)
        a = prod.dx(W, dY, K)
        b = var.dx(W, dY, K)
        exact = torch.equal(a, b)
        md = (a.float() - b.float()).abs().max().item()
        print(f"parity {name}: {'EXACT' if exact else f'DIFF maxdiff={md:.2e}'}", flush=True)
        ok = ok and exact
        res = ab_median(lambda: prod.dx(W, dY, K),
                        lambda: var.dx(W, dY, K), null_arm=True)
        print(report_line(name, res), flush=True)
    print("PROBE_DX_UNROLL2:", "PASS" if ok else "PARITY-FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
