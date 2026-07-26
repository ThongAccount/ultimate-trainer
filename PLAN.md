# PLAN — Packed Ternary Discrete Optimizer

## Current State (2026-07-25)

### What Works ✅
- **Convergence**: 99.8% loss reduction (617 → 1.06, 200 steps, 16→32→8 MLP)
- **TC kernels**: All 3 phases (forward, backward, update) use WMMA when dim ≥ 16
- **Update v2**: Vectorized counter (int32 pairs), skip-zero-grad, branchless
- **half2 unpack**: Forward + backward kernels use half2 vectorized SMEM stores
- **Fused backward_update**: One Python call for dX + counter flip
- **CUDA core GEMM**: `gemm_forward_packed.cu` — no unpack, ready to benchmark
- **11/11 kernels compile clean** on CUDA 13.3

### Benchmark (T4, 4096×4096, batch=32)

| Method | GFLOPS | Time | Memory/param |
|--------|--------|------|--------------|
| Discrete (TC) | 68-73 | 7.3-7.8ms | 18 bits |
| AdamW | 240-266 | 4.0-4.5ms | 96 bits |
| FusedAdamW | 243-488 | 2.2-4.4ms | 96 bits |

### Profile (kernel-level, 4096×4096, batch=32)

| Phase | Time | % |
|-------|------|---|
| Forward (TC) | 0.87ms | 14% |
| Backward (TC) | 0.94ms | 15% |
| Update (TC v2) | 1.00ms | 16% |
| Python overhead | 0.6-4.0ms | 55% (Colab noise) |
| **Kernel total** | **2.82ms** | |

### Theoretical Limits

| Metric | Value |
|--------|-------|
| Memory-bound min | 0.243ms (72.9 MB traffic) |
| Compute-bound min | 0.033ms (2.1 GFLOP) |
| Current kernel time | 2.82ms |
| Gap vs memory min | **12×** |

---

## What Was Done This Session

| # | Commit | Change | Impact |
|---|--------|--------|--------|
| 1 | `4b6a1c4` | TC dimension guards (all GEMM dims ≥ 16) | Fixed NaN/convergence |
| 2 | `84b4cf6` | Strip diagnostic warnings | Clean benchmarks |
| 3 | `670b48f` | Phase-level profiler | Identified bottlenecks |
| 4 | `4a166f1` | Update kernel v2 (vectorized counter) | -10% kernel time |
| 5 | `9fe831a` | Revert clone() removal | Fixed Python overhead regression |
| 6 | `45e0669` | Fused backward_update() | -1 Python call per step |
| 7 | `6abfe97` | half2 vectorized unpack (fwd + bwd) | -20% unpack overhead |
| 8 | `a9f24f5` | Remove redundant .contiguous() | -0.05ms Python |
| 9 | `69293d8` | CUDA core ternary GEMM (no unpack) | Ready to benchmark |
| 10 | `2df90ec` | Header fix for CUDA 12.9/13.3 | Compilation compatibility |

---

## What's Next — Closing the 12× Gap

### Priority 1: Benchmark the CUDA core GEMM (~2× kernel speedup)
The `gemm_forward_packed.cu` kernel processes packed uint32 directly — no unpack to FP16, 32× less W traffic. Expected: 0.5ms/forward vs 0.87ms/TC.

**Action**: Wire into auto-dispatch, benchmark vs TC, measure convergence.

### Priority 2: Eliminate Python overhead (~0.6ms → 0.1ms)
The autograd.Function dispatch adds 0.6-4.0ms (varies with Colab CPU). Options:
- **CUDAGraph**: Capture the entire train step, replay with zero Python overhead
- **Manual forward**: Skip autograd entirely, call kernels directly
- **torch.compile**: JIT-compile the step (blocked by custom autograd.Function)

**Action**: Write a `train_step_graph()` that captures fwd+bwd+update in a CUDAGraph.

### Priority 3: Fuse all 3 kernels into one mega-kernel (~0.5ms saved)
One kernel that does forward + backward + update. Shares X, dY, W in SMEM. Eliminates 2 kernel launches and 2× W unpack.

**Action**: Write `gemm_fused.cu` that processes all 3 phases in one launch.

### Priority 4: Double-buffered SMEM (~0.3ms saved)
Overlap next tile's memory load with current tile's WMMA compute. Hides global memory latency.

**Action**: Modify forward/backward kernels to use ping-pong SMEM buffers.

### Priority 5: Reduce counter traffic (~0.2ms saved)
Counter is 32MB read + 32MB write = 64MB per step. Options:
- **int8 counters**: Halves traffic (limits threshold to 127)
- **Sparse counter update**: Only write back changed counters
- **Counter in SMEM**: Keep hot counters on-chip (won't fit full 32MB)

**Action**: Try int8 counters with threshold=8.

### Priority 6: Larger model convergence
Current test: 16→32→8 MLP (2K params). Need to prove it scales:
- **Stage 1**: 768→2048→768 MLP
- **Stage 2**: Single transformer block
- **Stage 3**: Small GPT (6 layers, 128-dim)

**Action**: Write `test_convergence_large.py`.

---

## Architecture Notes

### Kernel Inventory (11 .cu files)
```
kernels/packed_ternary/
├── gemm_forward.cu / v2 / v3 / v4     (scalar forward variants)
├── gemm_forward_tc.cu                  (WMMA forward, 4-tile, half2 unpack)
├── gemm_forward_packed.cu              (CUDA core, no unpack — NEW)
├── gemm_backward_dx.cu                 (scalar backward)
├── gemm_backward_dx_tc.cu             (WMMA backward, half2 unpack)
├── gemm_update.cu                      (scalar update)
├── gemm_update_tc.cu                   (WMMA update v1)
├── gemm_update_tc_v2.cu               (WMMA update v2, vectorized counter)
└── packed_ternary.cuh                  (shared: pack/unpack LUT, state machine)
```

### Auto-dispatch Rules
- **Forward**: TC when B≥16, N≥16, K≥16 → v2 → v1
- **Backward**: TC when B≥16, N_out≥16, N_in≥16 → scalar
- **Update**: TC v2 when B≥16, N_out≥16, N_in≥16 → TC v1 → scalar
- **Packed**: Not in auto-dispatch (needs benchmarking)

### Memory Budget
| Component | Bits/param | Notes |
|-----------|-----------|-------|
| Weight | 2 | Packed ternary {-1,0,+1} |
| Counter | 16 | int16 (could be int8) |
| **Total** | **18** | vs 96 for AdamW (32 FP32 + 32 m + 32 v) |

### Sign Convention (verified correct)
```cuda
// Gradient descent: positive dW → decrease weight
if (grad > 0.0f)  cnt--;
else if (grad < 0.0f) cnt++;
```
Both scalar and TC kernels use this. The sign was NEVER the problem — it was always the NaN from OOB reads.
