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
    ap.add_argument("--fast", action="store_true", help="quick self-test: small dims, blocking launches, hard parity assert")
    args = ap.parse_args()

    if args.fast:
        # Small-dims self-test with blocking launches: surfaces OOB fast.
        import os
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        args.B, args.T, args.H, args.D, args.iters, args.warmup = 2, 64, 2, 32, 2, 1

    B, T, H, D = args.B, args.T, args.H, args.D
    D_out = H * D  # O projection dim
    BN = 16  # must match kernel TILE / block_sparse_ternary_matmul default

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

    # Block mask (dense-ish: top 3 of 8 N-tiles)
    # NOTE: num_k_tiles must match the kernel's BK (default 16).
    BK = 16
    num_n_tiles = (D_out + BN - 1) // BN
    num_k_tiles = (D_out + BK - 1) // BK
    top_idx = torch.tensor([[0, 1, 2]], device=dev)
    block_mask = compute_block_mask(top_idx, 3, BN, num_n_tiles, num_k_tiles)

    print(f"SubQSA combine bench: B={B} T={T} H={H} D={D} D_out={D_out}", flush=True)
    print(f"fused CUDA kernel: {'YES' if _HAS_SUBQSA_COMBINE else 'NO (eager only)'}", flush=True)
    print(f"block-sparse path: {'YES' if block_mask is not None else 'NO'}", flush=True)
    print(f"iters={args.iters} warmup={args.warmup}", flush=True)
    print()

    # Reference output (block_mask=None dense eager) for parity check
    ref = _subqsa_combine_eager(
        x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
        out_norm_weight, o_proj_weight, gamma, block_mask=None,
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
    print(f"eager dense:    {t*1e3:8.2f} ms  (parity err {err:.2e})", flush=True)

    # 2. Block-sparse matmul (CUDA kernel, kernel-safe tiles only: shared mem is 32x32)
    # NOTE: _subqsa_combine_eager's block_mask path uses defaults (BM=64,BN=64,BK=32)
    #       which OOB the fixed 32x32 shared buffers — instrument here directly.
    # Reference: sparse output computed via masked dense eager.
    ref_sparse = _subqsa_combine_eager(
        x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
        out_norm_weight, o_proj_weight, gamma, block_mask=None,
    )
    w_q = torch.clamp(torch.round(o_proj_weight / gamma), -1, 1) * gamma
    tile_n = (D_out + BN - 1) // BN  # N-tiles with BN=16
    for tn in range(tile_n):
        for tk in range(num_k_tiles):
            bit = tn * num_k_tiles + tk
            if not (block_mask[bit // 64] & (1 << (bit % 64))):
                ref_sparse[:, :, tn * BN:(tn + 1) * BN] = 0.0

    o_flat = torch.randn(B * T, D_out, device=dev, dtype=dtype)
    fn = lambda: block_sparse_ternary_matmul(
        o_flat, o_proj_weight, gamma, block_mask,
    )
    t = bench(fn, args.iters, args.warmup)
    y = fn()
    # kernel scales by gamma internally; ref must match
    ref_mm = (o_flat @ w_q.t()).float()
    for tn in range(tile_n):
        for tk in range(num_k_tiles):
            bit = tn * num_k_tiles + tk
            if not (block_mask[bit // 64] & (1 << (bit % 64))):
                ref_mm[:, tn * BN:(tn + 1) * BN] = 0.0
    err = (y.float() - ref_mm).abs().max().item()
    results["eager_sparse"] = t
    print(f"sparse CUDA:    {t*1e3:8.2f} ms  (parity err {err:.2e})", flush=True)

    # 3. Fused CUDA kernel (dense only — block_mask=None)
    if _HAS_SUBQSA_COMBINE:
        fn = lambda: _fused_call(
            x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
            out_norm_weight, o_proj_weight, gamma,
        )
        t = bench(fn, args.iters, args.warmup)
        y = fn()
        err = (y - ref).abs().max().item()
        if err > 1e-3:
            d = (y - ref).abs()
            idx = d.argmax().item()
            bb = idx // (T * D_out); rr = (idx % (T * D_out)) // D_out; cc = idx % D_out
            print(f"    FUSED max err at b={bb} t={rr} o={cc}: y={y.flatten()[idx].item():.5f} ref={ref.flatten()[idx].item():.5f}", flush=True)
            print(f"    diff>0.1 count: {(d > 0.1).sum().item()} / {d.numel()}", flush=True)
        results["fused_dense"] = t
        print(f"fused CUDA:     {t*1e3:8.2f} ms  (parity err {err:.2e})", flush=True)

    if args.fast:
        # Hard correctness gate (blocking mode + small dims)
        if _HAS_SUBQSA_COMBINE:
            y_fused = _fused_call(
                x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
                out_norm_weight, o_proj_weight, gamma,
            )
            assert (y_fused - ref).abs().max().item() < 1e-3, "fused kernel parity FAILED"
        y_sparse = block_sparse_ternary_matmul(o_flat, o_proj_weight, gamma, block_mask)
        ref_mm = (o_flat @ (torch.clamp(torch.round(o_proj_weight / gamma), -1, 1) * gamma).t()).float()
        for tn in range((D_out + BN - 1) // BN):
            for tk in range(num_k_tiles):
                bit = tn * num_k_tiles + tk
                if not (block_mask[bit // 64] & (1 << (bit % 64))):
                    ref_mm[:, tn * BN:(tn + 1) * BN] = 0.0
        assert (y_sparse.float() - ref_mm).abs().max().item() < 1e-3, "sparse kernel parity FAILED"
        print("SELFTEST: fused + sparse parity PASSED", flush=True)

    print()
    print("RAW:", results, flush=True)


def _fused_call(x, o_cmp, o_slc, o_win, gate_w1, gate_w2, out_norm_weight, o_proj_weight, gamma):
    from kernels.subqsa_combine.subqsa_combine import SubQSACombineFn
    return SubQSACombineFn.apply(
        x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
        out_norm_weight, o_proj_weight, gamma, None,
    )


if __name__ == "__main__":
    main()