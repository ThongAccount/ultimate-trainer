"""dX gate + batch-scale probe (hyperopt audit 2026-11-02, items 1 + "bigger GEMM batch").

Q1 (gate): at the REAL step M=16352 (x = data[:, :SEQ-1], train_gigatoken.py:173),
  backward_dx_tc's `B % 64 == 0` gate (custom_ops.py) fails, so the custom arm
  runs the 32x32 dX kernel. Is the 64x64 dX kernel faster at M=16352?
  A/B both raw kernels directly, bit-exact parity first. Shapes: fc1/fc2/head.

Q2 (batch scale): do larger GEMM M (more rows per multiplication batch) amortize
  better? solo_median of fwd tc_64 / dX 64 / update v2_64 at M in
  {16352, 16384, 32768}. M=32768 corresponds to B=64 x SEQ=512.

Usage: python tests/probe_dx_gate.py            # full
       python tests/probe_dx_gate.py --parity-only
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from bench_protocol import ab_median, solo_median, report_line

SHAPES = [("fc1", 1024, 4096), ("fc2", 4096, 1024), ("head", 1024, 50272)]
M_GATE = [16352, 16384]          # real step M, and the padded reference
M_SCALE = [16352, 16384, 32768]
THRESHOLD = 32


def make_state(name, K, N, M, dev):
    torch.manual_seed(0)
    W = torch.randint(0, 4, (N, (K + 15) // 16), device=dev, dtype=torch.int32)
    W = torch.where(W == 3, 0, W)  # ternary codes: no 3
    C = torch.zeros(N, K, dtype=torch.int16, device=dev)
    FC = torch.zeros(N, K, dtype=torch.int16, device=dev)
    X = (torch.randn(M, K, device=dev) * 0.1).half()
    dY = (torch.randn(M, N, device=dev) * 0.1).half()
    return W, C, FC, X, dY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity-only", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    dev = "cuda"

    from kernels.packed_ternary import pack_update as pu
    from kernels.packed_ternary.pack_forward import packed_ternary_forward_tc_64
    pu._load_dx_tc()
    pu._load_dx_tc_32()
    pu._load_up_tc_v2()
    assert pu._dx_tc_fn is not None and pu._dx_tc_32_fn is not None, "dX kernels not loaded"
    dx64, dx32 = pu._dx_tc_fn, pu._dx_tc_32_fn
    up64 = pu._up_tc_v2_64_fn
    assert up64 is not None, "update v2_64 not loaded"

    results = {"gate": {}, "scale": {}}

    # ── Q1: dX 64x64 vs 32x32 at real and padded M ─────────────────────────
    for name, K, N in SHAPES:
        for M in M_GATE:
            W, C, FC, X, dY = make_state(name, K, N, M, dev)
            y64 = dx64(W, dY, K)
            y32 = dx32(W, dY, K)
            same = torch.equal(y64, y32)
            print(f"{name}/M={M} dx64==dx32: {same}", flush=True)
            if not args.parity_only:
                assert same, f"parity FAILED {name}/M={M}"
                res = ab_median(lambda: dx32(W, dY, K), lambda: dx64(W, dY, K),
                                null_arm=(name == "fc1" and M == M_GATE[0]))
                print(report_line(f"{name}/M={M} dx32-vs-dx64", res), flush=True)
                results["gate"][f"{name}/M={M}"] = {
                    "parity": same, "a_ms": res.a_ms, "b_ms": res.b_ms,
                    "delta_pct": res.delta_pct, "sigma_null_pct": res.sigma_null_pct}
            else:
                results["gate"][f"{name}/M={M}"] = {"parity": same}
        torch.cuda.empty_cache()

    # ── Q2: batch scaling of the production kernels ────────────────────────
    if not args.parity_only:
        for name, K, N in SHAPES:
            for M in M_SCALE:
                W, C, FC, X, dY = make_state(name, K, N, M, dev)
                W0, C0 = W.clone(), C.clone()

                def prep():
                    W.copy_(W0); C.copy_(C0)

                fwd = solo_median(lambda: packed_ternary_forward_tc_64(W, X))
                dx = solo_median(lambda: dx64(W, dY, K))
                up = solo_median(lambda: up64(W, C, FC, X, dY, THRESHOLD, 1e-3, 0),
                                prepare=prep)
                print(f"scale {name}/M={M}: fwd={fwd.b_ms:.2f}ms dx64={dx.b_ms:.2f}ms "
                      f"up64={up.b_ms:.2f}ms", flush=True)
                results["scale"][f"{name}/M={M}"] = {
                    "fwd_ms": fwd.b_ms, "dx64_ms": dx.b_ms, "up64_ms": up.b_ms}
                torch.cuda.empty_cache()

    print("\nRESULTS JSON:", json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
