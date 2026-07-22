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
    has_v3,
    packed_ternary_forward_v3,
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

    # FLOPs for ternary: each weight apply is 1 add/sub (no multiply needed).
    # Standard dense GEMM convention would be 2*M*N*K (multiply+add),
    # but ternary discards the multiply → M*N*K is the real operation count.
    flops = batch * out_f * in_f
    tflops = (flops / 1e12) / (median / 1e3)

    return {
        "ver": name,
        "batch": batch,
        "in_features": in_f,
        "out_features": out_f,
        "flops": flops,
        "median_ms": median,
        "best_ms": best,
        "gflops": tflops * 1e3,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Run
# ═══════════════════════════════════════════════════════════════════════════════

print(f"{'ver':>3} {'batch':>5} {'in_f':>6} {'out_f':>6} {'FLOPs':>12} {'median(ms)':>11} {'best(ms)':>9} {'GFLOPS':>8}")
print("─" * 75)

kernels = [("v1", packed_ternary_forward)]
if HAS_V2:
    kernels.append(("v2", packed_ternary_forward_v2))
if has_v3():
    kernels.append(("v3", packed_ternary_forward_v3))
if has_v4():
    kernels.append(("v4", packed_ternary_forward_v4))

all_results = []
for ver, kernel_fn in kernels:
    for batch, in_f, out_f in SHAPES:
        s = benchmark_shape(kernel_fn, ver, batch, in_f, out_f)
        all_results.append(s)
        print(
            f"{ver:>3} {s['batch']:>5} {s['in_features']:>6} {s['out_features']:>6} "
            f"{s['flops']:>12} {s['median_ms']:>9.3f}  {s['best_ms']:>7.3f}  "
            f"{s['gflops']:>6.1f}"
        )

print("─" * 75)
versions = {r["ver"] for r in all_results}
for ver in sorted(versions):
    ver_results = [r for r in all_results if r["ver"] == ver]
    avg = sum(r["gflops"] for r in ver_results) / max(1, len(ver_results))
    print(f"{'Average GFLOPS ' + ver + ':':>55} {avg:.1f}")

