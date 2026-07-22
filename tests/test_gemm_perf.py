"""Benchmark: Packed ternary forward GEMM — latency and throughput.

Measures kernel launch time across matrix shapes and computes TFLOPS.
Provides the baseline for Phase 2B optimisation.

Usage:
    uv run python3 tests/test_gemm_perf.py
"""

from __future__ import annotations

import math
import sys
import os
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernels.packed_ternary import pack_tensor
from kernels.packed_ternary.pack_forward import (
    has_forward_kernel,
    packed_ternary_forward,
    has_forward_kernel_v2,
    packed_ternary_forward_v2,
)

if not torch.cuda.is_available():
    print("CUDA not available — benchmark requires GPU")
    sys.exit(0)

if not has_forward_kernel():
    print("Packed ternary forward kernel not loaded")
    sys.exit(0)

HAS_V2 = has_forward_kernel_v2()

# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark shapes
# ═══════════════════════════════════════════════════════════════════════════════

SHAPES = [
    # (batch, in_features, out_features)  — description
    (1, 128, 128),
    (1, 256, 256),
    (1, 512, 512),
    (1, 1024, 1024),
    (1, 4096, 4096),
    (4, 128, 128),
    (4, 256, 256),
    (4, 512, 512),
    (4, 1024, 1024),
    (4, 4096, 4096),
    (8, 256, 256),
    (8, 512, 512),
    (8, 1024, 1024),
    (16, 1024, 1024),
    (32, 1024, 1024),
]

WARMUP = 5
ITERS = 20

# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark
# ═══════════════════════════════════════════════════════════════════════════════


def benchmark_shape(kernel_fn, name: str, batch: int, in_f: int, out_f: int) -> dict:
    """Run *kernel_fn* on one shape and return timing stats."""

    # Prepare data
    torch.manual_seed(0)
    W_fp32 = torch.randn(out_f, in_f)
    W_packed = pack_tensor(W_fp32).cuda()
    X = torch.randn(batch, in_f, dtype=torch.float16, device="cuda")

    # Warmup
    for _ in range(WARMUP):
        _ = kernel_fn(W_packed, X)
    torch.cuda.synchronize()

    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms = []
    for _ in range(ITERS):
        start.record()
        _ = kernel_fn(W_packed, X)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    times_ms = sorted(times_ms)
    median = times_ms[len(times_ms) // 2]
    best = times_ms[0]
    worst = times_ms[-1]

    # Throughput
    macs = batch * out_f * in_f * 2  # multiply + add per MAC
    tflops = (macs / 1e12) / (median / 1e3)

    return {
        "batch": batch,
        "in_features": in_f,
        "out_features": out_f,
        "macs": macs,
        "median_ms": median,
        "best_ms": best,
        "worst_ms": worst,
        "gflops": tflops * 1e3,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Run
# ═══════════════════════════════════════════════════════════════════════════════

print(f"{'ver':>3} {'batch':>5} {'in_f':>6} {'out_f':>6} {'MACs':>12} {'median(ms)':>11} {'best(ms)':>9} {'GFLOPS':>8}")
print("─" * 75)

kernels = [("v1", packed_ternary_forward)]
if HAS_V2:
    kernels.append(("v2", packed_ternary_forward_v2))

all_results = []
for ver, kernel_fn in kernels:
    for batch, in_f, out_f in SHAPES:
        s = benchmark_shape(kernel_fn, ver, batch, in_f, out_f)
        all_results.append(s)
        print(
            f"{ver:>3} {s['batch']:>5} {s['in_features']:>6} {s['out_features']:>6} "
            f"{s['macs']:>12} {s['median_ms']:>9.3f}  {s['best_ms']:>7.3f}  "
            f"{s['gflops']:>6.1f}"
        )

print("─" * 75)
avg_v1 = sum(r["gflops"] for r in all_results if r.get("ver", "v1") == "v1") / max(1, len(SHAPES))
avg_v2 = sum(r["gflops"] for r in all_results if r.get("ver", "") == "v2") / max(1, len(SHAPES) if HAS_V2 else 1)
print(f"{'Average GFLOPS v1:':>55} {avg_v1:.1f}")
if HAS_V2:
    print(f"{'Average GFLOPS v2:':>55} {avg_v2:.1f}")
    if avg_v1 > 0:
        print(f"{'Speedup v2/v1:':>55} {avg_v2/avg_v1:.2f}x")
