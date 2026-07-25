"""Profile each phase of one discrete train step to find bottlenecks.

Measures:
  1. Forward (ternary GEMM)
  2. Loss computation
  3. Backward dX (ternary GEMM)
  4. Update (dense GEMM + counter logic)
  5. Python/autograd overhead

Uses CUDA events for GPU timing and time.perf_counter for CPU overhead.
"""

import sys, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kernels.packed_ternary import PackedTernaryLinear
from kernels.packed_ternary.pack_update import backward_dx, update

if not torch.cuda.is_available():
    print("CUDA not available"); sys.exit(0)

B, K, N = 32, 4096, 4096
WARMUP, ITERS = 10, 50

layer = PackedTernaryLinear(K, N, threshold=8).cuda()

# ── 1. Full autograd step (baseline) ──────────────────────────────

def full_autograd_step():
    x = torch.randn(B, K, dtype=torch.float16, device="cuda")
    y = layer(x)
    loss = y.mean()
    loss.backward()

for _ in range(WARMUP):
    full_autograd_step()
torch.cuda.synchronize()

e0 = torch.cuda.Event(enable_timing=True)
e1 = torch.cuda.Event(enable_timing=True)
times_full = []
for _ in range(ITERS):
    e0.record()
    full_autograd_step()
    e1.record()
    torch.cuda.synchronize()
    times_full.append(e0.elapsed_time(e1))
full_ms = sorted(times_full)[len(times_full)//2]

# ── 2. Forward only ───────────────────────────────────────────────

x_ref = torch.randn(B, K, dtype=torch.float16, device="cuda")
for _ in range(WARMUP):
    _ = layer(x_ref)
torch.cuda.synchronize()

times_fwd = []
for _ in range(ITERS):
    e0.record()
    y = layer(x_ref)
    e1.record()
    torch.cuda.synchronize()
    times_fwd.append(e0.elapsed_time(e1))
fwd_ms = sorted(times_fwd)[len(times_fwd)//2]

# ── 3. Backward dX only (direct kernel call) ──────────────────────

y_ref = torch.randn(B, N, dtype=torch.float16, device="cuda")
for _ in range(WARMUP):
    backward_dx(layer.W_packed, y_ref, K)
torch.cuda.synchronize()

times_bwd = []
for _ in range(ITERS):
    e0.record()
    dx = backward_dx(layer.W_packed, y_ref, K)
    e1.record()
    torch.cuda.synchronize()
    times_bwd.append(e0.elapsed_time(e1))
bwd_ms = sorted(times_bwd)[len(times_bwd)//2]

# ── 4. Update only (direct kernel call) ───────────────────────────

x_ref2 = torch.randn(B, K, dtype=torch.float16, device="cuda")
y_ref2 = torch.randn(B, N, dtype=torch.float16, device="cuda")
counter_snap = layer.counter.clone()

for _ in range(WARMUP):
    update(layer.W_packed, layer.counter, x_ref2, y_ref2, 8)
    layer.counter.copy_(counter_snap)
torch.cuda.synchronize()

times_upd = []
for _ in range(ITERS):
    layer.counter.copy_(counter_snap)
    e0.record()
    update(layer.W_packed, layer.counter, x_ref2, y_ref2, 8)
    e1.record()
    torch.cuda.synchronize()
    times_upd.append(e0.elapsed_time(e1))
upd_ms = sorted(times_upd)[len(times_upd)//2]

# ── 5. CPU overhead (Python + autograd dispatch) ──────────────────

overhead_ms = full_ms - fwd_ms - bwd_ms - upd_ms

# ── Report ─────────────────────────────────────────────────────────

print(f"\n{'═'*60}")
print(f"  Profile: discrete train step ({B}×{K}×{N})")
print(f"{'═'*60}")
print(f"  {'Phase':<30} {'ms':>8} {'%':>8}")
print(f"  {'─'*48}")
print(f"  {'Forward (ternary TC)':<30} {fwd_ms:>8.3f} {100*fwd_ms/full_ms:>7.1f}%")
print(f"  {'Backward dX (ternary TC)':<30} {bwd_ms:>8.3f} {100*bwd_ms/full_ms:>7.1f}%")
print(f"  {'Update (dense TC + counter)':<30} {upd_ms:>8.3f} {100*upd_ms/full_ms:>7.1f}%")
print(f"  {'Python/autograd overhead':<30} {overhead_ms:>8.3f} {100*overhead_ms/full_ms:>7.1f}%")
print(f"  {'─'*48}")
print(f"  {'TOTAL (autograd step)':<30} {full_ms:>8.3f} {'100.0':>7}%")
print(f"{'═'*60}")

# ── Theoretical minimum ───────────────────────────────────────────

# Memory traffic
x_bytes = 2 * B * K * 2        # X read ×2 (fwd + update)
dy_bytes = 2 * B * N * 2       # dY read ×2 (bwd + update)
w_bytes = 3 * K * N // 16      # W packed read ×3 (fwd + bwd + update) — 2-bit
ctr_bytes = 2 * K * N * 2      # counter read+write (int16)
y_bytes = B * N * 2            # Y write
dx_bytes = B * K * 2           # dX write
w_write = K * N // 16          # W packed write (flipped weights)
total_mem = x_bytes + dy_bytes + w_bytes + ctr_bytes + y_bytes + dx_bytes + w_write

bw_gb_s = 300  # T4 HBM bandwidth
min_mem_ms = total_mem / (bw_gb_s * 1e9) * 1000

# Compute ops
fwd_ops = B * N * K            # ternary add/sub
bwd_ops = B * N * K            # ternary add/sub
upd_ops = 2 * K * N * B        # dense FP16 MAC
total_ops = fwd_ops + bwd_ops + upd_ops

tc_tflops = 65  # T4 FP16 TC peak
min_compute_ms = total_ops / (tc_tflops * 1e12) * 1000

print(f"\n  Theoretical minimum:")
print(f"    Memory-bound: {min_mem_ms:.3f} ms ({total_mem/1e6:.1f} MB traffic)")
print(f"    Compute-bound: {min_compute_ms:.3f} ms ({total_ops/1e9:.1f} GFLOP)")
print(f"    Actual: {full_ms:.3f} ms")
print(f"    Gap vs memory min: {full_ms/min_mem_ms:.0f}×")
print(f"    Gap vs compute min: {full_ms/min_compute_ms:.0f}×")
print(f"{'═'*60}\n")
