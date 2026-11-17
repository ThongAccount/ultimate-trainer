"""Probe #11: fwd TC64 kernel with uint4 W loads (K-window 64, one 16 B load
per W row per window; MMA structure otherwise unchanged, 16 MMAs/sync).

Prod and variant compiled as SEPARATE extensions (same globals — one TU would
be a redefinition error). Parity on fc1 and head shapes, then interleaved A/B
via bench_protocol.ab_median (null arm first). No production dispatch touched.

Hypothesis (prereg 2026-11-03-probe-batch-2): W global loads are scalar
32-bit; uint4 cuts L1 transactions 4x — expected -1..-3% on fwd.
Caution: K64 window is adjacent to the closed "fwd K-widening" dead end;
if delta >= +2% on both shapes this closes as negative like that one.
Ship bar: delta <= -max(2*sigma_null, 1.0)% on BOTH shapes, bit-exact parity.

Usage: python tests/probe_fwd_uint4.py
"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CUH = os.path.join(REPO, "kernels/packed_ternary/packed_ternary.cuh")
PROD_CU = os.path.join(REPO, "kernels/packed_ternary/gemm_forward_tc.cu")
VAR_CU = os.path.join(HERE, "fwd_uint4.cu")

with open(CUH) as f:
    CUH_SRC = f.read()


def make_cpp(launch_fn, fn_name):
    return r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
extern "C" {
    void LAUNCH_FN(
        const uint32_t* W, const void* X, void* Y,
        int batch_size, int in_features, int out_features,
        int stride_words, cudaStream_t stream);
}
torch::Tensor FN_NAME(torch::Tensor W, torch::Tensor X) {
    auto Y = torch::empty({X.size(0), W.size(0)}, torch::dtype(torch::kFloat16).device(X.device()));
    LAUNCH_FN(
        reinterpret_cast<const uint32_t*>(W.data_ptr<int32_t>()),
        X.data_ptr<at::Half>(), Y.data_ptr<at::Half>(),
        X.size(0), X.size(1), W.size(0), W.size(1), nullptr);
    return Y;
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
    print("building prod + uint4 variants...", flush=True)
    prod = build(PROD_CU, "probe_fwd_prod_ext",
                 "launch_packed_ternary_forward_tc_64", "fwd")
    var = build(VAR_CU, "probe_fwd_uint4_ext",
                "probe_launch_fwd_uint4", "fwd")
    print("built.", flush=True)

    ok = True
    for name, K, N in [("fc1", 1024, 4096), ("head", 1024, 50272)]:
        X = torch.randn(B, K, device=dev, dtype=torch.float16)
        W = make_W(N, K, dev)
        a = prod.fwd(W, X)
        b = var.fwd(W, X)
        exact = torch.equal(a, b)
        md = (a.float() - b.float()).abs().max().item()
        print(f"parity {name}: {'EXACT' if exact else f'DIFF maxdiff={md:.2e}'}", flush=True)
        ok = ok and exact
        res = ab_median(lambda: prod.fwd(W, X),
                        lambda: var.fwd(W, X), null_arm=True)
        print(report_line(name, res), flush=True)
    print("PROBE_FWD_UINT4:", "PASS" if ok else "PARITY-FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
