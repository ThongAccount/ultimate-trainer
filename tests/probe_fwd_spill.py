"""Parity + timing check for the fwd warp-private spill epilogue (GO item #1, C13 sweep).

OLD (production kernels/packed_ternary/gemm_forward_tc.cu): spills all 4
fragments per warp into a 16 KB spill[4][4][256] buffer, one __syncthreads(),
then a whole-CTA cooperative copy of all 4096 floats. The 16 KB of static smem
(20480 B total) caps occupancy at 3 CTAs/SM on the 64 KB T4 SM.

NEW (tests/fwd_spill_warpprivate.cu): the shipped dX epilogue pattern —
one warp-private 16x16 slot (4 KB) reused per fragment, each warp copies its
own tile to global, __syncwarp between fragments, zero CTA barriers in the
epilogue. Static smem 20480 -> 8192 B.

Parity gate: BIT-EXACT on Y (torch.equal) at fc1/fc2/head shapes, plus an FP32
reference check. Both are pure compute kernels — same values, different copy
ownership; there is no cross-warp data flow in either epilogue form.

Timing: bench_protocol.ab_median (10-trial median, interleaved A/B/A/B,
clock-settle barrier), with an A/A null arm on the first shape to
calibrate sigma_null. This is the mandatory C13 protocol; the old 3-trial min
protocol is retired.

Usage:
    python tests/probe_fwd_spill.py                # parity + full A/B timing
    python tests/probe_fwd_spill.py --probe fwdspill --skip-timing   # parity only
    python tests/probe_fwd_spill.py --probe fwdspill --null arm      # explicit null
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from bench_protocol import ab_median, report_line

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CUH = os.path.join(REPO, "kernels/packed_ternary/packed_ternary.cuh")
OLD_CU = os.path.join(REPO, "kernels/packed_ternary/gemm_forward_tc.cu")
NEW_CU = os.path.join(HERE, "fwd_spill_warpprivate.cu")

with open(CUH) as f:
    CUH_SRC = f.read()


def build(cu_path, name):
    from torch.utils.cpp_extension import load_inline

    with open(cu_path) as f:
        cu = f.read()
    combined = CUH_SRC + "\n" + cu.replace('#include "packed_ternary.cuh"', "")
    return load_inline(
        name=name,
        cpp_sources=r"""
        #include <cuda_runtime.h>
        #include <torch/extension.h>
        extern "C" {
            void launch_packed_ternary_forward_tc_64(
                const uint32_t* W, const void* X, void* Y,
                int batch_size, int in_features, int out_features,
                int stride_words, cudaStream_t stream);
        }
        torch::Tensor fwd(torch::Tensor W, torch::Tensor X) {
            auto Y = torch::empty({X.size(0), W.size(0)}, torch::dtype(torch::kFloat16).device(X.device()));
            launch_packed_ternary_forward_tc_64(
                reinterpret_cast<const uint32_t*>(W.data_ptr<int32_t>()),
                X.data_ptr<at::Half>(), Y.data_ptr<at::Half>(),
                X.size(0), X.size(1), W.size(0), W.size(1), nullptr);
            return Y;
        }
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fwd", &fwd); }
        """,
        cuda_sources=[combined],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],  # production flags (pack_forward.py:505)
    )


def unpack_ref(W, K):
    """Unpack W[N, stride_words] into a [N, K] FP16 ternary matrix."""
    N, stride = W.shape
    words = W.long().unsqueeze(-1)
    shifts = torch.arange(16, device=W.device).view(1, 1, 16) * 2
    codes = (words >> shifts) & 3
    vals = torch.where(codes == 1, 1.0, torch.where(codes == 2, -1.0, 0.0))
    return vals.reshape(N, stride * 16)[:, :K].to(torch.float16)


def main():
    parser = argparse.ArgumentParser(description="fwd warp-private spill epilogue probe")
    parser.add_argument("--probe", default="fwdspill",
                        help="probe name tag (default: fwdspill)")
    parser.add_argument("--skip-timing", action="store_true",
                        help="parity checks only, skip the A/B timing arms")
    parser.add_argument("--null", default="arm", choices=["arm", "skip"],
                        help="run the A/A null arm on the first shape (default: arm)")
    parser.add_argument("--batch", type=int, default=16384)
    args = parser.parse_args()

    dev = "cuda"
    print("building OLD (prod 16KB spill + CTA cooperative copy) ...", flush=True)
    old = build(OLD_CU, "fwd_spill_old")
    print("built OLD ok", flush=True)
    print("building NEW (warp-private 4KB spill, dX pattern) ...", flush=True)
    new = build(NEW_CU, "fwd_spill_new")
    print("built NEW ok", flush=True)

    torch.manual_seed(0)
    BATCH = args.batch
    shapes = [("fc1", 1024, 4096), ("fc2", 4096, 1024), ("head", 1024, 50272)]

    print(
        f"\n{'name':<6} {'in':>5} {'out':>6}   {'old_err':>9} {'new_err':>9} {'old==new':>9}",
        flush=True,
    )

    results = {}
    for si, (name, inn, out) in enumerate(shapes):
        X = torch.randn(BATCH, inn, device=dev, dtype=torch.float16)
        W = torch.randint(0, 4, (out, (inn + 15) // 16), device=dev, dtype=torch.int32)
        W = torch.where(W == 3, 0, W)
        Wf = unpack_ref(W, inn)
        # kernel: Y[b, n] = SUM_k X[b, k] * W[n, k]  =>  Y = X @ Wf.t()
        ref = (X.float() @ Wf.float().t()).half()

        yo = old.fwd(W, X)
        yn = new.fwd(W, X)
        eo = (yo.float() - ref.float()).abs().max().item()
        en = (yn.float() - ref.float()).abs().max().item()
        same = torch.equal(yo, yn)

        print(
            f"{name:<6} {inn:>5} {out:>6}   {eo:9.3e} {en:9.3e} {str(same):>9}",
            flush=True,
        )
        if not same:
            print(f"FATAL: {name} NOT bit-exact — do not ship", flush=True)
            sys.exit(1)

        if args.skip_timing:
            continue

        null_arm = si == 0 and args.null == "arm"
        res = ab_median(
            lambda: old.fwd(W, X),
            lambda: new.fwd(W, X),
            null_arm=null_arm,
        )
        results[name] = res
        print(report_line(f"{args.probe}/{name}", res), flush=True)

    if results:
        print("\nRESULTS " + __import__("json").dumps({
            name: {
                "old_ms": round(res.a_ms, 3),
                "new_ms": round(res.b_ms, 3),
                "delta_pct": round(res.delta_pct, 3),
                "sigma_null_pct": (round(res.sigma_null_pct, 3)
                                   if res.sigma_null_pct is not None else None),
            }
            for name, res in results.items()
        }), flush=True)


if __name__ == "__main__":
    main()
