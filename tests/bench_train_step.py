"""Benchmark: one training step — forward + backward + weight update.

Compares approaches:
    1. Discrete (auto-dispatch): TC forward + auto backward/update
    2. Discrete (scalar forced): all scalar kernels (old behavior)
    3. Standard AdamW: nn.Linear + F.linear + AdamW
    4. Fused AdamW: nn.Linear + F.linear + torch.compile + FusedAdamW

Measures GPU kernel time (median over 20 runs) for each step.
"""

from __future__ import annotations

import sys, os, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernels.packed_ternary import PackedTernaryLinear, compute_stride_words

if not torch.cuda.is_available():
    print("CUDA not available")
    sys.exit(0)

# ═══════════════════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════════════════

SHAPES = [
    # (batch, in_features, out_features) — description
    (1, 1024, 1024),
    (4, 1024, 1024),
    (8, 1024, 1024),
    (16, 1024, 1024),
    (32, 1024, 1024),
    (1, 4096, 4096),
    (4, 4096, 4096),
    (8, 4096, 4096),
    (16, 4096, 4096),
    (32, 4096, 4096),
]

WARMUP = 5
ITERS = 20


# ═══════════════════════════════════════════════════════════════════════════════
#  Discrete stack benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def bench_discrete(B, K, N):
    """One train step: PackedTernaryLinear forward + backward + update.

    Auto-dispatches to WMMA Tensor Cores for all phases when batch >= 16,
    scalar kernels for smaller batches.
    """

    # Use the module which includes auto-dispatch for forward
    layer = PackedTernaryLinear(K, N, threshold=64).cuda()

    # Warmup
    for _ in range(WARMUP):
        x = torch.randn(B, K, dtype=torch.float16, device="cuda")
        y = layer(x)
        loss = y.mean()
        loss.backward()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms = []
    for _ in range(ITERS):
        x = torch.randn(B, K, dtype=torch.float16, device="cuda")
        start.record()
        y = layer(x)
        loss = y.mean()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    times_ms.sort()
    median = times_ms[len(times_ms) // 2]
    flops = B * N * K  # ternary: add/sub per weight ~1 FLOP
    gflops = (flops / 1e9) / (median / 1e3)
    return {"median_ms": median, "gflops": gflops, "ver": "discrete"}


# ═══════════════════════════════════════════════════════════════════════════════
#  Standard AdamW benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def bench_adamw(B, K, N):
    """One train step: nn.Linear + F.linear + AdamW."""

    layer = nn.Linear(K, N, bias=False).cuda().half()
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-4, fused=False)

    for _ in range(WARMUP):
        x = torch.randn(B, K, dtype=torch.float16, device="cuda")
        y = F.linear(x, layer.weight)
        loss = y.mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms = []
    for _ in range(ITERS):
        x = torch.randn(B, K, dtype=torch.float16, device="cuda")
        start.record()
        y = F.linear(x, layer.weight)
        loss = y.mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    times_ms.sort()
    median = times_ms[len(times_ms) // 2]
    flops = 2 * B * N * K  # standard dense: multiply+add
    gflops = (flops / 1e9) / (median / 1e3)
    return {"median_ms": median, "gflops": gflops, "ver": "adamw"}


# ═══════════════════════════════════════════════════════════════════════════════
#  Fused AdamW benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def bench_fused_adamw(B, K, N):
    """One train step: nn.Linear + F.linear + FusedAdamW."""

    layer = nn.Linear(K, N, bias=False).cuda().half()
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-4, fused=True)

    for _ in range(WARMUP):
        x = torch.randn(B, K, dtype=torch.float16, device="cuda")
        y = F.linear(x, layer.weight)
        loss = y.mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms = []
    for _ in range(ITERS):
        x = torch.randn(B, K, dtype=torch.float16, device="cuda")
        start.record()
        y = F.linear(x, layer.weight)
        loss = y.mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    times_ms.sort()
    median = times_ms[len(times_ms) // 2]
    flops = 2 * B * N * K
    gflops = (flops / 1e9) / (median / 1e3)
    return {"median_ms": median, "gflops": gflops, "ver": "fused_adamw"}


# ═══════════════════════════════════════════════════════════════════════════════
#  Run
# ═══════════════════════════════════════════════════════════════════════════════

print(f"{'ver':>12} {'batch':>5} {'in_f':>6} {'out_f':>6} {'median(ms)':>11} {'GFLOPS':>8}")
print("─" * 60)

benchmarks = [
    ("discrete", bench_discrete),
    ("adamw", bench_adamw),
    ("fused_adamw", bench_fused_adamw),
]

all_results = []
for ver, fn in benchmarks:
    for B, K, N in SHAPES:
        s = fn(B, K, N)
        all_results.append(s)
        print(
            f"{s['ver']:>12} {B:>5} {K:>6} {N:>6} "
            f"{s['median_ms']:>9.3f}  {s['gflops']:>6.1f}"
        )

print("─" * 60)

# Summary
for ver in ["discrete", "adamw", "fused_adamw"]:
    vals = [r for r in all_results if r["ver"] == ver]
    avg = sum(r["gflops"] for r in vals) / len(vals)
    print(f"{'Average GFLOPS ' + ver + ':':>42} {avg:.1f}")
