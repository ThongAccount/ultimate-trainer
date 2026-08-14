"""SubQSA combine benchmark: fused CUDA kernel vs eager vs block-sparse.

Measures the O-projection-heavy path of SubQSA combine:
  1. gate MLP (Linear->SiLU->Linear)
  2. per-head sigmoid + L1 normalize
  3. 3-way blend (o_cmp/o_slc/o_win)
  4. RMSNorm
  5. Ternary O projection (dense or block-sparse)

Usage: python bench_subqsa_combine.py [--iters 50] [--warmup 10]
"""

import argparse
import time

import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kernels.subqsa_combine.subqsa_combine import (
    _subqsa_combine_eager, _HAS_SUBQSA_COMBINE,
)
from kernels.block_sparse_ternary.block_sparse_ternary import (
    block_sparse_ternary_matmul, compute_block_mask,
)

def bench(fn, iters, warmup, sync_each=False):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if sync_each:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    torch.cuda.synchronize()
    times.sort()
    return sum(times[2:-2]) / max(1, len(times) - 4)  # trim extremes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--D", type=int, default=64)  # head dim
    args = ap.parse_args()

    B, T, H, D = args.B, args.T, args.H, args.D
    D_out = H * D  # O projection dim
    BN = 64

    dev = "cuda"
    dtype = torch.float32

    # Inputs
    x = torch.randn(B, T, D_out, device=dev, dtype=dtype)
    o_cmp = torch.randn(B, H, T, D, device=dev, dtype=dtype)
    o_slc = torch.randn(B, H, T, D, device=dev, dtype=dtype)
    o_win = torch.randn(B, H, T, D, device=dev, dtype=dtype)
    gate_w1 = torch.randn(64, D_out, device=dev, dtype=dtype) * 0.1
    gate_w2 = torch.randn(3 * H, 64, device=dev, dtype=dtype) * 0.1
    out_norm_weight = torch.randn(D_out, device=dev, dtype=dtype)
    o_proj_weight = torch.randn(D_out, D_out, device=dev, dtype=dtype) * 0.05
    gamma = 0.1

    # Block mask (dense-ish: top 3 of 4 N-tiles)
    # NOTE: num_k_tiles must match the kernel's BK (default 32), not BN.
    BK = 32
    num_n_tiles = (D_out + BN - 1) // BN
    num_k_tiles = (D_out + BK - 1) // BK
    top_idx = torch.tensor([[0, 1, 2]], device=dev)
    block_mask = compute_block_mask(top_idx, 3, BN, num_n_tiles, num_k_tiles)

    print(f"SubQSA combine bench: B={B} T={T} H={H} D={D} D_out={D_out}")
    print(f"fused CUDA kernel: {'YES' if _HAS_SUBQSA_COMBINE else 'NO (eager only)'}")
    print(f"block-sparse path: {'YES' if block_mask is not None else 'NO'}")
    print(f"iters={args.iters} warmup={args.warmup}")
    print()

    # Reference output (block_mask=None dense eager) for parity check
    ref = _subqsa_combine_eager(
        x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
        out_norm_weight, o_proj_weight, gamma, block_mask=None,
    )
    ref_sparse = _subqsa_combine_eager(
        x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
        out_norm_weight, o_proj_weight, gamma, block_mask=block_mask,
    )

    results = {}

    # 1. Eager dense (reference)
    fn = lambda: _subqsa_combine_eager(
        x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
        out_norm_weight, o_proj_weight, gamma, block_mask=None,
    )
    t = bench(fn, args.iters, args.warmup)
    y = fn()
    err = (y - ref).abs().max().item()
    results["eager_dense"] = t
    print(f"eager dense:    {t*1e3:8.2f} ms  (parity err {err:.2e})")

    # 2. Eager block-sparse
    if _HAS_SUBQSA_COMBINE or True:
        fn = lambda: _subqsa_combine_eager(
            x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
            out_norm_weight, o_proj_weight, gamma, block_mask=block_mask,
        )
        t = bench(fn, args.iters, args.warmup)
        y = fn()
        err = (y - ref_sparse).abs().max().item()
        results["eager_sparse"] = t
        print(f"eager sparse:   {t*1e3:8.2f} ms  (parity err {err:.2e})")

    # 3. Fused CUDA kernel (dense only — block_mask=None)
    if _HAS_SUBQSA_COMBINE:
        fn = lambda: _subqsa_combine_eager.__self__ if False else _fused_call(
            x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
            out_norm_weight, o_proj_weight, gamma,
        )
        t = bench(fn, args.iters, args.warmup)
        y = fn()
        err = (y - ref).abs().max().item()
        results["fused_dense"] = t
        print(f"fused CUDA:     {t*1e3:8.2f} ms  (parity err {err:.2e})")

    print()
    print("RAW:", results)


def _fused_call(x, o_cmp, o_slc, o_win, gate_w1, gate_w2, out_norm_weight, o_proj_weight, gamma):
    from kernels.subqsa_combine.subqsa_combine import SubQSACombineFn
    return SubQSACombineFn.apply(
        x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
        out_norm_weight, o_proj_weight, gamma, None,
    )


if __name__ == "__main__":
    main()