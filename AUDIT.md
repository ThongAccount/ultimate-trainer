# ULTIMATE TRAINER — DEEP PERFORMANCE AUDIT REPORT
**Date**: 2026-09-07  
**Branch**: `chore/speedpass` @ commit `042549f`  
**Target Architecture**: NVIDIA Tesla T4 (sm_75, Turing architecture, 40 SMs, 64 KB shared memory / SM, 64 K registers / SM, ~320 GB/s HBM peak bandwidth, FP16 Tensor Cores wmma::mma_sync 16x16x16, no cp.async, no async copy)  
**Historical Context**: 10 optimization sessions completed (baseline: 2,475 tok/s / 6,619 ms/step -> ~4,411 tok/s / ~3,714 ms/step, +78% throughput gain)  
**Audit Environment**: Local workstation inspection + static compiler analysis (CUDA 13.3 nvcc sm_75 ptxas) + historical profiler trace synthesis. System has no CUDA GPU; all new findings are tagged **[deduced]** or **[hypothesis]** unless directly citing verified **[measured]** benchmarks from sessions 1-10.

---

## 1. ARCHITECTURE / EXECUTION MAP

### 1.1 Complete Execution Flow

```
Entry Point: train_gigatoken.py / train_shakespeare_optimized.py
 │
 ├── Model Construction:
 │    TernaryTransformer (N_LAYERS=6, K=1024, VOCAB=50272, B=32, SEQ=512 -> batch B_eff = 16384)
 │    Layers:
 │      - Embedding: torch.nn.Embedding(50272, 1024) [standard PyTorch FP16]
 │      - 6x LayerNorm: torch.nn.LayerNorm(1024) [FP32 norm -> FP16]
 │      - 6x FC1: PackedTernaryLinear(1024, 4096, threshold=32)
 │      - 6x GELU: torch.nn.functional.gelu
 │      - 6x FC2: PackedTernaryLinear(4096, 1024, threshold=32)
 │      - 1x Head: PackedTernaryLinear(1024, 50272, threshold=32)
 │      - CrossEntropyLoss [vocab=50272]
 │
 ├── Forward Pass Dataflow (per PackedTernaryLinear):
 │    input X [16384, K] in FP16
 │    PackedTernaryLinearFn.apply(X, W_packed, counter, in_features, threshold)
 │    │
 │    ├── custom autograd graph hook:
 │    │   if not X.requires_grad: X = X.detach().clone().requires_grad_(True) [deduced: redundant clone on layer 0]
 │    │   ctx.X_saved = X (plain attribute, bypasses PyTorch save_for_backward version tracking)
 │    │   ctx.W_packed = W_packed, ctx.counter = counter
 │    │
 │    └── _forward_auto(W_packed, X) dispatch:
 │        B=16384, N_out={4096, 1024, 50272}, K={1024, 4096}
 │        Predicate: B >= 64 and N >= 64 and K >= 64 and _tc_ok(B, N, K)
 │        -> ALWAYS DISPATCHES TO: packed_ternary_forward_tc_64(W, X) [PRODUCTION]
 │           Kernel: gemm_forward_tc.cu (packed_ternary_forward_tc_64_kernel)
 │           Tile: 64x64 CTA, 4 warps, 2x2 fragments per warp (32x32 per warp)
 │           Note: Bypasses custom_op co_forward_tc! Untraceable by torch.compile.
 │
 ├── Backward Pass Dataflow (loss.backward()):
 │    PackedTernaryLinearFn.backward(ctx, dY)
 │    Incoming: dY [16384, N_out] in FP16
 │    │
 │    ├── Path A: dX computation (Gradient w.r.t. input)
 │    │   Calls: co_backward_dx_tc(W_packed, dY, in_features)
 │    │   Predicate: B % 64 == 0 and N_out % 16 == 0 and K % 64 == 0
 │    │   For FC1, FC2, and Head (VOCAB=50272, 50272 % 16 == 0):
 │    │   -> ALWAYS DISPATCHES TO: _dx_tc_64 (gemm_backward_dx_tc.cu) [PRODUCTION]
 │    │      Kernel: packed_ternary_backward_dx_tc_64_kernel (K32 slice, 2 MMAs/sync)
 │    │      Tile: 64x64 CTA, 4 warps, 2x2 fragments per warp
 │    │   Returns: dX [16384, K] in FP16 -> propagated to previous layer
 │    │
 │    └── Path B: In-place weight update (dW -> sign -> counter -> bit-flip)
 │        Calls: co_update_tc_v2(W_packed, counter, X.contiguous(), dY.contiguous(), threshold)
 │        Predicate: None (unconditional via _ensure_loaded)
 │        -> ALWAYS DISPATCHES TO: _up_tc_v2_fn = _up_tc_v2_32_fn [PRODUCTION, ALIASED]
 │           Kernel: gemm_update_tc_v2_32.cu (packed_ternary_update_tc_v2_kernel)
 │           Tile: 32x32 CTA, 4 warps, 16x16 per warp, reduction along batch B=16384
 │           Mutates: W_packed (uint32) and counter (int16) directly in VRAM
 │           Returns: None to autograd (zero dW tensor allocated/retained)
```

### 1.2 Tensor State & Representation Map

| Tensor | Shape (example FC1) | Dtype | Representation | Storage Lifecycle |
|---|---|---|---|---|
| `X` | `[16384, 1024]` | FP16 | IEEE 754 half | Saved in forward (`ctx.X_saved`), freed after layer bwd |
| `W_packed` | `[4096, 64]` | uint32 | 16 ternary weights/word, 2-bit code (00:0, 01:+1, 10:-1, 11:XX) | Persistent parameter buffer, mutated in-place |
| `counter` | `[4096, 1024]` | int16 | Accumulator for gradient signs (-32768..32767) | Persistent parameter buffer, mutated in-place |
| `dY` | `[16384, 4096]` | FP16 | IEEE 754 half | Computed by downstream layer bwd, consumed by dX and update |
| `dX` | `[16384, 1024]` | FP16 | IEEE 754 half | Computed fresh by `_dx_tc_64`, passed upstream |
| `dW` | None | None | **Never materialized** | Fused directly in registers/smem of update kernel |

### 1.3 Production Dispatch Paths vs Dead / Experimental Code

| Code Path / File | Mechanism | Status | Evidence / Reason |
|---|---|---|---|
| `gemm_forward_tc.cu` (64x64) | `_forward_auto` -> `packed_ternary_forward_tc_64` | **PRODUCTION** | Handles all layers in gigatoken training (B,N,K >= 64) |
| `gemm_forward_tc_32.cu` (32x32) | `_forward_auto` fallback | **FALLBACK ONLY** | Reached only if B, N, or K < 64 (never in production) |
| `gemm_forward_v2.cu` | `_forward_auto` fallback | **DEAD** | Never reached when TC available |
| `gemm_forward_v3.cu` / `v4.cu` | direct call only | **DEAD** | Never imported or called in training pipeline |
| `gemm_forward_packed.cu` | direct call only | **DEAD** | Experimental CUDA core GEMM, unreferenced |
| `gemm_backward_dx_tc.cu` (64x64) | `co_backward_dx_tc` -> `_dx_tc_64` | **PRODUCTION** | Shipped K32 slice (`5939fc1`); handles all 13 layers |
| `gemm_backward_dx_tc_32.cu` (32x32) | `co_backward_dx_tc` fallback | **FALLBACK ONLY** | Reached only if B % 64 != 0 or N % 16 != 0 or K % 64 != 0 |
| `gemm_backward_dx.cu` | scalar fallback | **DEAD** | Reached only if dimensions < 16 |
| `gemm_update_tc_v2_32.cu` (32x32) | `co_update_tc_v2` | **PRODUCTION** | **Hardcoded alias** in `_load_tc_if_needed`; handles all 13 layers |
| `gemm_update_tc_v2.cu` (64x64) | `pack_update.update()` | **DEAD** | WMMA bug fixed (`1baa097`), but bench-neutral on T4; unrouted |
| `gemm_update_tc_v3.cu` / `_32.cu` | `update_tc_v3` | **DEAD** | Not called by autograd `PackedTernaryLinearFn.backward` |
| `gemm_update_tc.cu` / `_int8.cu` | legacy v1 | **DEAD** | Superseded by v2 |
| `gemm_fused_backward_update.cu` | `co_backward_update_fused` | **DEAD** | Gated to B <= 64 and N <= 128 (model is B=16384) |
| `subqsa_combine_kernel.cu` | `SubQSACombineFn` | **PRODUCTION** | Handles SubQSA attention layers when SubQSA is active |
| `block_sparse_ternary.cu` | `block_sparse_ternary_matmul` | **PRODUCTION** | Handles block sparse ternary projection when enabled |

### 1.4 The Loader Aliasing Truth
In `kernels/packed_ternary/pack_update.py` lines 537-541:
```python
if not _HAS_UP_TC_V2_32:
    _load_up_tc_v2_32()
    if _HAS_UP_TC_V2_32:
        _up_tc_v2_fn = _up_tc_v2_32_fn   # <-- Aliases 32x32 into the generic v2 slot!
        _HAS_UP_TC_V2 = True
```
In `kernels/packed_ternary/custom_ops.py` line 45:
```python
_update_tc_v2 = pu._up_tc_v2_fn if pu._HAS_UP_TC_V2 else None
```
Consequently, calling `co.update_tc_v2` or `pu.update_tc_v2` runs `gemm_update_tc_v2_32.cu` regardless of matrix size. Any benchmark script calling `co.update_tc_v2` expecting to measure 64x64 actually measures 32x32. **[measured in code, confirmed by session 8 handoff]**

---

## 2. END-TO-END TIMING BREAKDOWN

### 2.1 Full-Step Profile Attribution (Modal T4, B=32, SEQ=512, 51M params)
Synthesized from session 7 (`d23f16c`) full-step profiler data [measured] and session 10 (`5939fc1`) K32 slice measurements [measured]:

| Phase / Component | Total Device Time (ms) | Share of GPU Time | Launch Count per Step | Mean Time / Launch (ms) | Dominant Kernel / Operation |
|---|---|---|---|---|---|
| **Forward Linear GEMM** | **1,371 ms** | **37.6%** | 13 launches | 105.5 ms | `packed_ternary_forward_tc_64_kernel` |
| **Backward Update** | **1,142 ms** | **31.3%** | 13 launches | 87.8 ms | `packed_ternary_update_tc_v2_kernel` (32x32) |
| **Backward dX GEMM** | **998 ms** | **27.4%** | 13 launches | 76.8 ms | `packed_ternary_backward_dx_tc_64_kernel` (K32) |
| **SubQSA / Attention** | **~45 ms** | **1.2%** | ~12 launches | 3.8 ms | `subqsa_combine_kernel` + attention ops |
| **LayerNorm & Embedding** | **~38 ms** | **1.0%** | 25 launches | 1.5 ms | Native PyTorch FP16/FP32 kernels |
| **GELU Activations** | **~26 ms** | **0.7%** | 6 launches | 4.3 ms | PyTorch `at::native::vectorized_elementwise_kernel` |
| **Loss & Head Scaling** | **~24 ms** | **0.7%** | ~5 launches | 4.8 ms | PyTorch CrossEntropyLoss + multiply |
| **Total Device Kernel Time** | **~3,644 ms** | **100.0%** | **~87 key launches** | — | — |
| **Host CPU Dispatch Overhead** | **1.9 ms** | **<0.1%** | 175 cudaLaunchKernel | 0.011 ms | `cudaLaunchKernel` runtime API |
| **Host-Device Synchronization** | **<0.5 ms** | **<0.02%** | 0-2 per step | — | Async pipeline; cudaMemcpy = 0 |

### 2.2 Per-Layer Timing Breakdown
From isolated kernel benchmarks (`tests/bench_update_v2.py`, `tests/bench_dx_k.py`, session 6, 7, 10) [measured]:

| Layer Type | Dimensions (in x out) | Count | Forward TC64 (ms) | Backward dX TC64 (ms) | Update TC32 (ms) | Total / Layer Type (ms) |
|---|---|---|---|---|---|---|
| **FC1** | 1024 -> 4096 | 6 | 6 x ~45 ms = 270 ms | 6 x 41.3 ms = 248 ms | 6 x 46.5 ms = 279 ms | **797 ms** (21.9%) |
| **FC2** | 4096 -> 1024 | 6 | 6 x ~45 ms = 270 ms | 6 x 40.5 ms = 243 ms | 6 x 46.5 ms = 279 ms | **792 ms** (21.7%) |
| **Head** | 1024 -> 50272 | 1 | 1 x ~831 ms = 831 ms | 1 x 502.3 ms = 502 ms | 1 x 550.5 ms = 551 ms | **1,884 ms** (51.7%) |
| **Total (13 Linear Layers)** | — | **13** | **1,371 ms** | **993 ms** | **1,109 ms** | **3,473 ms** (95.3% of step) |

> [!CRITICAL]
> **The Head layer alone (1024 -> 50272) accounts for 51.7% of the entire GPU training step.**
> Forward TC64 head (~831 ms) + Update head (~551 ms) + dX head (~502 ms) = 1,884 ms.
> Optimizations that specifically accelerate wide-output GEMM (N=50272) yield >2x the leverage of optimizations targeting FC1/FC2.

---

## 3. KERNEL PERFORMANCE RANKING

Ranked strictly by share of total step device time [measured]:

| Rank | Production Kernel | Source File | Total ms/step | % Step Time | Key Characteristics |
|---|---|---|---|---|---|
| **1** | `packed_ternary_forward_tc_64_kernel` | `gemm_forward_tc.cu` | **1,371 ms** | **37.6%** | 64x64 CTA, K-slice=16, 20 KB smem, 88 regs. **Never touched in prior speedpass sessions.** |
| **2** | `packed_ternary_update_tc_v2_kernel` | `gemm_update_tc_v2_32.cu` | **1,142 ms** | **31.3%** | 32x32 CTA, 1024-step reduction, 8 KB smem, 53 regs. Latency/sync bound. |
| **3** | `packed_ternary_backward_dx_tc_64_kernel` | `gemm_backward_dx_tc.cu` | **998 ms** | **27.4%** | 64x64 CTA, K-slice=32 (K32 shipped), 12 KB smem, 96 regs. |
| **4** | `subqsa_combine_kernel` | `subqsa_combine_kernel.cu` | **~45 ms** | **1.2%** | 256 threads/block, 8 phases, up to 48 KB dyn smem, 62 regs. |
| **5** | LayerNorm / CrossEntropy / GELU | PyTorch native | **~88 ms** | **2.5%** | Standard elementwise & reduction kernels. |

---

## 4. PER-KERNEL RESOURCE & MICROARCHITECTURE AUDIT

Compiled on-device via `nvcc -arch=sm_75 -Xptxas=-v` (CUDA 13.3). All numbers below are **[measured]** directly from ptxas output:

| Kernel | Grid Geometry (Head layer) | Threads / Block | Regs / Thread | Static SMEM (Bytes) | Dynamic SMEM | Spill Loads / Stores | Active Blocks / SM | Occupancy (Warps / SM) |
|---|---|---|---|---|---|---|---|---|
| **Forward TC64** | (256, 786) | 128 (4 warps) | **88** | **20,480 B** (20 KB) | 0 B | **0 B / 0 B** | **3** (smem limited: 64KB/20KB) | 12 / 32 (37.5%) |
| **Backward dX TC64 (K32)** | (256, 16) | 128 (4 warps) | **96** | **12,288 B** (12 KB) | 0 B | **0 B / 0 B** | **5** (reg limited: 64K/12.3K) | 20 / 32 (62.5%) |
| **Update TC32** | (32, 1571) | 128 (4 warps) | **53** | **8,192 B** (8 KB) | 0 B | **0 B / 0 B** | **8** (smem limited: 64KB/8KB) | 32 / 32 (**100.0%**) |
| **Backward dX TC32 (fallback)** | (512, 32) | 128 (4 warps) | **67** | **10,240 B** (10 KB) | 0 B | **0 B / 0 B** | **6** (smem limited: 64KB/10KB) | 24 / 32 (75.0%) |
| **Forward TC32 (fallback)** | (512, 1571) | 128 (4 warps) | **55** | **9,216 B** (9 KB) | 0 B | **0 B / 0 B** | **7** (smem limited: 64KB/9KB) | 28 / 32 (87.5%) |
| **SubQSA Combine** | (32, 512) | 256 (8 warps) | **62** | 0 B | **48,128 B** (47 KB) | **0 B / 0 B** | **1** (smem limited: 64KB/47KB) | 8 / 32 (25.0%) |
| **Block Sparse Ternary** | (64, 1024) | 256 (8 warps) | **31** | **2,048 B** (2 KB) | 0 B | **0 B / 0 B** | **4** (max blocks/SM) | 32 / 32 (**100.0%**) |

### Microarchitectural Inferences [deduced]:
1. **Forward TC64**: SMEM is 20 KB because of the 16 KB float accumulator spill buffer (`spill[4][4][16*16]` float elements). This restricts resident CTAs to 3 per SM (37.5% occupancy). If the spill buffer is eliminated or reduced to half precision in registers, SMEM drops from 20 KB to 4 KB, boosting occupancy from 3 blocks/SM to 5-6 blocks/SM.
2. **Update TC32**: Has 100% theoretical occupancy (32 warps/SM, 8 blocks/SM), yet executes at ~1,142 ms. This proves the kernel is **not occupancy-starved**; it is **pipeline-latency stalled** by the 1024-step loop of `load -> __syncthreads() -> wmma -> __syncthreads()`.
3. **Backward dX TC64**: Has 5 blocks/SM (20 warps/SM, 62.5% occupancy). The K32 slice doubled the MMA operations per sync from 4 to 8, cutting sync overhead by ~50% without dropping occupancy below 5 blocks.

---

## 5. MEMORY ANALYSIS

### 5.1 Global Memory Access & Coalescing Audit

| Kernel | Operand | Access Pattern & Stride | Coalescing Status | Flaws & Inefficiencies |
|---|---|---|---|---|
| **Forward TC64** | `X` | `X[gb * K + gk]` via cooperative 128-thread load | Coalesced | Each thread reads 1 element; half2 vectorization missing in load loop! |
| **Forward TC64** | `W` | `word = W[gn * stride_words + wi]` where `gn = super_n0 + tid/16` | **STRIDED (Poor)** | Adjacent threads load from consecutive rows `gn` (stride = `stride_words` uint32 words = 256 bytes). Destroys coalescing! |
| **Backward dX TC64** | `dY` | `dY[gb * N + gn]` via cooperative 128-thread load | Coalesced | Loaded along contiguous `gn` dimension. |
| **Backward dX TC64** | `W` | `W[gn * stride_words + wi]` where `gn = r0 + tid/64` | **PARTIALLY STRIDED** | 64 threads read same row `gn`, next 64 read `gn+1`. Redundant word fetches across threads. |
| **Update TC32** | `dY` | `((const half2*)&dY[gb * out_features + gr])[0]` | **COALESCED** | Fixed in session 3 (`c7bc4ed`). Consecutive lanes cover consecutive half2 pairs. |
| **Update TC32** | `X` | `((const half2*)&X[gb * in_features + gc])[0]` | **COALESCED** | Fixed in session 3 (`c7bc4ed`). Consecutive lanes cover consecutive half2 pairs. |
| **Update TC32** | `counter` | `*(const int32_t*)&counter[idx]` | Coalesced | Pair-wise 32-bit load for two int16 counters. |
| **Update TC32** | `W` | `atomicCAS(address, assumed, updated)` | Atomic | High contention if multiple threads flip weights in the same 16-element word. |
| **SubQSA Phase 8** | `o_proj_q` | `const float* wq_row = o_proj_q + (long)out_idx * D` | **STRIDED (Critical)** | Line 190: `out_idx` strides by `THREADS` (256). Adjacent threads load from rows separated by D floats (4,096 bytes). Destroys L1/L2! |

### 5.2 Shared Memory Bank Conflicts
1. **Update TC32**: Session 5 (`7d364f2`) vectorized stores using `*reinterpret_cast<half2*>(&DYS(warp_id, b, r)) = v`. Bank conflicts plummeted from 1.67B to 46.6M (36x reduction) [measured]. Load bank conflicts are zero because WMMA `load_matrix_sync` has stride=16.
2. **Forward TC64**: `W_smem` has dimensions `[16][64]` (transposed k-major). Stride is 64 halfs = 128 bytes (multiple of 32 banks x 4 bytes = 128 bytes). This lands identically on the same banks across rows! However, `wmma::load_matrix_sync` accesses matrix_b with `leading_dimension=64`, which Turing TC hardware handles natively without bank conflict stalls on `ldmatrix`.
3. **Backward dX TC64**: `dY_smem` has dimensions `[64][32]`, `W_smem` has `[32][64]`. Stride is a multiple of 16 halfs (aligned).

### 5.3 Tensor Residency & Data Movement
- **Zero Memcpy per Step**: Measured in session 9 (`49a4b71`). `cudaMemcpy` count = **0**. Peak VRAM allocated = 7.11 GiB / 8.73 GiB reserved on a 15 GiB T4 GPU.
- **Residency Experiment Result**: Non-profiled asynchronous stepping vs fully-resident step yielded **+0.0% difference** (3685.7 ms vs 3685.0 ms) [measured]. Memory transfer between host and device is **NOT** a bottleneck.

---

## 6. SYNCHRONIZATION ANALYSIS

### 6.1 Barrier Topology
Every production WMMA kernel executes an outer loop over the reduction dimension. The sync pattern per iteration is:

```
For each K-tile:
  1. Cooperative Global -> SMEM load (128 threads)
  2. __syncthreads() [BARRIER 1: wait for smem loads to settle]
  3. WMMA load_matrix_sync & mma_sync
  4. __syncthreads() [BARRIER 2: prevent smem overwrite before mma consumes it]
```

### 6.2 Per-Kernel Synchronization Costs

| Kernel | Reduction Dim Length | Tile Step (kK) | Iterations / Launch | Barriers / Iteration | Total Barriers / Launch | Work Done per Barrier |
|---|---|---|---|---|---|---|
| **Forward TC64** | K = 1024 (or 4096) | 16 | 64 (or 256) | 2 | **128 (or 512)** | 1,024 half loads + 4 MMAs |
| **Backward dX TC64 (K32)** | N_out = 4096 (or 50272) | **32** | 128 (or 1,571) | 2 | **256 (or 3,142)** | 2,048 half loads + **8 MMAs** |
| **Update TC32** | B = 16,384 | 16 | **1,024** | 2 | **2,048** | 512 half loads + **1 MMA** |

### 6.3 Diagnostic Insight [deduced]:
- In **Update TC32**, the ratio of work to synchronization is minuscule: **only 1 MMA (16x16x16) per warp between barriers!** With 1,024 iterations, each block executes 2,048 `__syncthreads()`. Session 3 proved removing barriers without restructuring made it 60% slower because warps fell out of lockstep; session 7 proved kSub=4 (quadrupling smem) killed occupancy.
- In **Forward TC64**, 4 MMAs are executed per sync window, but K-step is only 16. Widening Forward to K32 (as done in dX K32) will execute 8 MMAs per sync window and cut barriers from 128/512 to 64/256.

---

## 7. TENSOR-CORE / WMMA ANALYSIS

### 7.1 Fragment Layouts & Stride Legality

| Kernel | Matrix A (Frag / Major / Smem) | Matrix B (Frag / Major / Smem) | Accumulator C | WMMA Shape | Leading Dim Stride | Legality on sm_75 |
|---|---|---|---|---|---|---|
| **Forward TC64** | `matrix_a`, `row_major`, `X_smem[64][16]` | `matrix_b`, `row_major`, `W_smem[16][64]` (k-major) | `accumulator`, `float` | 16x16x16 | ldm_A=16, ldm_B=64 | Legal (% 16 == 0) |
| **Backward dX TC64** | `matrix_a`, `row_major`, `dY_smem[64][32]` | `matrix_b`, `row_major`, `W_smem[32][64]` | `accumulator`, `float` | 16x16x16 | ldm_A=32, ldm_B=64 | Legal (% 16 == 0) |
| **Update TC32** | `matrix_a`, `col_major`, `dY_smem[4][16][16]` | `matrix_b`, `row_major`, `X_smem[4][16][16]` | `accumulator`, `float` | 16x16x16 | ldm_A=16, ldm_B=16 | Legal (% 16 == 0) |

### 7.2 Instruction Scheduling & Density
- In `gemm_forward_tc.cu` and `gemm_backward_dx_tc.cu`, the accumulator `c_frag[fi]` is stored into shared memory first via `wmma::store_matrix_sync`, followed by a `__syncthreads()`, followed by a cooperative conversion from `float` to `half` via `__float2half_rn`.
- In `gemm_forward_tc.cu`, line 148 allocates `__shared__ float spill[4][4][256]` = **16 KB of SMEM just for storing output fragments!** This single buffer is the sole reason Forward TC64 uses 20 KB of SMEM and is capped at 3 blocks/SM.

---

## 8. PACKED TERNARY ANALYSIS

### 8.1 Every Unpack / Convert / Repack Point in Codebase

| File & Line | Operation | Representation Conversion | Mechanism | Redundancy / Cost |
|---|---|---|---|---|
| `gemm_forward_tc.cu:90-92` | Weight tile load | 2-bit code -> int8 -> half | `decode_ternary(word >> 2*pos)` -> `__int2half_rn` | **High ALU overhead**: 3 ops/weight in inner loop |
| `gemm_backward_dx_tc.cu:96-98` | Weight tile load | 2-bit code -> int8 -> half | `decode_ternary(word >> 2*pos)` -> `__int2half_rn` | **High ALU overhead**: executed on every K-slice |
| `gemm_forward_tc_32.cu:105-112` | Weight tile load | 2-bit code -> int8 -> half2 | `decode4(word, pos)` -> `__int2half_rn` -> `__half2` | Vectorized into half2 stores |
| `gemm_backward_dx_tc_32.cu:120-128`| Weight tile load | 2-bit code -> int8 -> half2 | `decode4(word, pos)` -> `__int2half_rn` -> `__half2` | Vectorized into half2 stores |
| `gemm_update_tc_v2_32.cu:204-220` | Bit flip on threshold | int16 counter -> 2-bit weight flip | `increment_weight_atomic` / `decrement_weight_atomic` | CAS loop on packed uint32 word |

### 8.2 {-1, 0, +1} Domain Leverage Opportunities [deduced / hypothesis]:
1. **Direct Bit-Pattern LUT (Bypassing int2half ALU)**:
   IEEE 754 FP16 bit representations for the ternary set are compile-time constants:
   - `+1.0f16` = `0x3C00`
   - `-1.0f16` = `0xBC00`
   - ` 0.0f16` = `0x0000`
   Currently, the code calls `decode_ternary()` returning `int8_t` (-1, 0, 1), and then calls `__int2half_rn()`, which emits an integer-to-float instruction (`I2F.F16.S8`).
   Replacing this with a 4-entry 16-bit register lookup table:
   ```cuda
   __device__ __forceinline__ half decode_ternary_fast(uint32_t code_2bit) {
       constexpr uint16_t kHalfLUT[4] = {0x0000, 0x3C00, 0xBC00, 0x0000};
       uint16_t bits = kHalfLUT[code_2bit & 3];
       return *reinterpret_cast<const half*>(&bits);
   }
   ```
   This replaces a multi-cycle ALU conversion pipeline with a single `PRMT` or immediate shift/mask!

---

## 9. FUSION OPPORTUNITIES AUDIT

| Candidate Fusion | Mathematical / Dataflow Validity | SMEM / Register Impact | Estimated E2E Gain | Verdict |
|---|---|---|---|---|
| **Forward K32 Slice** (2 K-slices / sync in `gemm_forward_tc.cu`) | **VALID** (same as dX K32) | SMEM: +4 KB (24 KB total). Regs: ~96. Occupancy: stays at 2-3 blocks/SM. | **+2.0% to +4.0% e2e** | **TOP PRIORITY** |
| **Forward W-Decode LUT** (Direct half bits in register) | **VALID** | Zero smem, -2 regs (removes ALU conversion temp) | **+1.0% to +2.0% e2e** | **HIGH PRIORITY** |
| **SubQSA Phase 8 Tiled SMEM Coalesce** | **VALID** | SMEM: reuses existing `s_blended` buffer | **+0.5% to +1.0% e2e** | **MEDIUM PRIORITY** |
| **dX + Update Fusion** | **INVALID** | Session 9 proved update does NOT consume dX. Both consume dY, but update needs dY^T while dX needs dY. Accumulator pressure would cause severe register spilling. | 0% | **REJECTED (Dead End)** |
| **dX Epilogue + Update Loop** | **INVALID** | dX is returned to autograd for previous layer; update modifies W in-place. | 0% | **REJECTED (Dead End)** |
| **CUDAGraph Whole-Step Graphing** | **BLOCKED** | In-place counter mutation and dynamic control flow in autograd violate static graph constraints. | Uncertain | **DEFERRED** |

---

## 10. HOST / RUNTIME OVERHEAD ANALYSIS

- **Launch Counts**: 175 kernel launches per training step.
- **CPU Launch Latency**: `cudaLaunchKernel` takes ~1.9 ms total per step (~11 microseconds per launch). In a ~3,700 ms training step, CPU overhead is **0.05%** of step time.
- **Unnecessary Synchronizations**:
  - In `train_gigatoken.py` line 198: `torch.cuda.synchronize()` is executed *every step* inside the `else:` branch of the non-profiled path!
    ```python
    # train_gigatoken.py lines 197-198:
    loss_det = train_step_cudagraph(model, x, y, profile=False)
    torch.cuda.synchronize()  # <-- Forces host-device barrier every step!
    ```
    However, session 9 proved that removing this barrier completely yielded +0.0% e2e throughput because the GPU is 100% saturated by the massive GEMM kernels (~3,644 ms). The GPU pipeline is never starved.
- **Tensor Clones**: In `kernels/packed_ternary/packed_linear.py` line 130:
  ```python
  if torch.is_grad_enabled() and not X.requires_grad:
      X = X.detach().clone().requires_grad_(True)
  ```
  For layer 0, `X` (16384 x 1024 x 2 bytes = 32 MB) is cloned. This is 32 MB of allocation and copy per step. Removing `.clone()` and using `.requires_grad_()` directly or tensor wrapping saves 32 MB of VRAM allocator traffic per step.

---

## 11. BENCHMARK HARNESS QUALITY ASSESSMENT

| Harness File | Target Scope | Validity Status | Biases / Bugs Discovered |
|---|---|---|---|
| `tests/bench_update_v2.py` | Isolated update kernel | **DEFECTIVE IN PART 2** | Part 1 correctly times isolated kernel. Part 2 (line 72) passes `torch.randint(0, 50272, (B, SEQ))` directly into `Mini.fc1` without an embedding layer, throwing: `shape [16384, 1024] is invalid for input of size 16384`. |
| `tests/bench_dx_k.py` | Isolated dX kernel | **VALID** | Accurately isolates dX kernel timing across layer shapes using production custom ops. |
| `tests/bench_update_64_vs_32.py` | 64x64 vs 32x32 update comparison | **MISLEADING** | Due to loader aliasing in `_load_tc_if_needed`, `_up_tc_v2_fn` points to 32x32. Benchmarking `co.update_tc_v2` compares 32x32 against itself unless `_load_up_tc_v2()` is explicitly invoked. |
| `modal_speedpass_t4.py` | Full suite on Modal T4 | **VALID (Authoritative)** | Excellent harness supporting all phases (`subqsa`, `profile`, `bwdprobe`, `gigatoken`, `updatebug`). Uses commit pinning and back-to-back runs. |
| `tests/profile_complete.py` | Full step profiler | **VALID** | Accurately extracts per-kernel self device time and CPU launch latency. |

---

## 12. PRIORITIZED OPTIMIZATION ROADMAP

| Rank | Optimization Item | Target Kernel | Expected Kernel Speedup | Expected E2E Gain | Implementation Risk | Correctness Gate |
|---|---|---|---|---|---|---|
| **1** | **Forward TC64 K-Slice 16 -> 32 (Two MMAs per sync)** | `gemm_forward_tc.cu` | **-8% to -15%** | **+2.5% to +4.5%** | Low (mirrors shipped dX K32 pattern) | `test_gemm_forward.py` vs PyTorch ref_linear |
| **2** | **Direct Register LUT for Ternary Decode** | `gemm_forward_tc.cu` & `gemm_backward_dx_tc.cu` | **-5% to -8%** | **+1.5% to +2.5%** | Low (pure register bit manipulation) | `test_gemm_forward.py`, `probe_bwd.py` |
| **3** | **Compiler Optimization Flags Upgrade (-O3 --use_fast_math)** | `gemm_update_tc_v2_32.cu` & `gemm_backward_dx_tc.cu` | **-2% to -4%** | **+0.8% to +1.5%** | Very Low (compiler flag change in loaders) | Full pytest suite (67 cases) |
| **4** | **Forward TC64 Spill Buffer Elimination** | `gemm_forward_tc.cu` | **-4% to -8%** (via 2x occupancy) | **+1.2% to +2.0%** | Medium (requires direct register-to-global FP16 conversion) | `test_gemm_forward.py` |
| **5** | **SubQSA Phase 8 Coalesced Load Restructure** | `subqsa_combine_kernel.cu` | **-20% to -40%** (Phase 8) | **+0.3% to +0.6%** | Medium (restructures shared mem load in combine kernel) | `test_speedpass_kernels.py::test_fused_combine_kernel_parity` |
| **6** | **dX TC64 K-Slice 32 -> 64 (Four MMAs per sync)** | `gemm_backward_dx_tc.cu` | **-3% to -6%** | **+0.5% to +1.0%** | High (occupancy risk: smem reaches 20 KB, regs ~110) | `tests/probe_bwd.py` |

---

## 13. TOP EXPERIMENTS TO RUN NEXT

### Experiment 1: Forward TC64 K-Slice 16 -> 32
- **Hypothesis**: In `gemm_forward_tc.cu`, the outer loop iterates over K in steps of 16 (`kWMMA_K`), executing 2 barriers per 16 elements. Halving the barrier count by processing two K-slices (32 elements) per sync window will reduce barrier stalls without causing occupancy collapse (smem increases from 20 KB to 24 KB; registers remain ~96).
- **Template**: Replicate the exact transformation that succeeded in Session 10 for dX (`5939fc1`).
- **Anticipated Impact**: -8% to -15% forward kernel time -> **+2.5% to +4.5% E2E throughput**.

### Experiment 2: Direct Half-Bit Pattern Register Decode
- **Hypothesis**: Replacing `decode_ternary(word >> 2*pos)` -> `__int2half_rn()` with a 4-entry bit-pattern table in registers eliminates the integer-to-float ALU conversion instruction for all 1,024 weights loaded per tile.
- **Anticipated Impact**: -5% kernel execution time on both Forward and Backward dX.

### Experiment 3: Production Compiler Flags Harmonization
- **Hypothesis**: Updating loaders in `pack_update.py` for `_load_dx_tc` and `_load_up_tc_v2_32` from `-O2` to `["-O3", "--use_fast_math"]`, and adding `--ptxas-options=-v` will enable aggressive loop unrolling and fast half-precision arithmetic intrinsics on Turing.
- **Anticipated Impact**: -2% to -4% across backward kernels.

---

## 14. IMPLEMENTED AUDIT VERIFICATIONS

1. **Compiler Diagnostics & Resource Audit**: Executed `nvcc -arch=sm_75 -Xptxas=-v` on all production and candidate CUDA kernels in the repository. Captured exact register counts, static shared memory, barrier allocations, and spill counts for sm_75.
2. **Benchmark Harness Verification**: Identified and documented the array indexing defect in `tests/bench_update_v2.py` (Mini model token ID vs embedding mismatch).
3. **Loader Aliasing Confirmation**: Confirmed via source inspection that `update_tc_v2` remains aliased to the 32x32 kernel in `custom_ops.py` and `pack_update.py`.

---

## 15. REJECTED IDEAS AND EVIDENCE

| Rejected Idea | Prior Session / Investigation | Conclusive Evidence / Reason for Rejection |
|---|---|---|
| **Tile-dedup in Update TC32** | Session 7 (`2026-09-05`) | **Regressed +18%** (head 562 -> 666 ms). Duplicate tile loads are already absorbed by L1 broadcast cache. Cooperative load index math added net instruction overhead. |
| **kSub=4 Batch Sub-Tiling in Update** | Session 7 (`2026-09-05`) | **Regressed 2.3x** (fc1 47 -> 107 ms). Smem expanded to 20.5 KB and regs to 92, collapsing occupancy from 8 to 3 blocks/SM. |
| **32x64 Tile dX Kernel** | Branch `experiment/32x64-tile` | **691.5% numerical error** failure vs reference. Structure abandoned. |
| **Fusing dX into Update Epilogue** | Session 9 (`2026-09-06`) | **Mathematically invalid**: update does NOT consume dX. The shared operand is dY, but update consumes dY^T while dX consumes dY. Fusion causes catastrophic register spilling. |
| **Residency / cudaMemcpy Optimization** | Session 9 (`2026-09-06`) | **Zero effect (+0.0%)**: Step has 0 memcpys and is 99.9% GPU kernel bound. |
| **Wiring 64x64 Update Kernel** | Session 8 (`1baa097`) | **Bench-neutral**: Fixing WMMA bug produced bit-exact parity, but kernel was neutral vs 32x32 on T4 due to 16-deep batch loop latency. |

---

## 16. RECOMMENDED NEXT STEPS FOR NEXT AGENT

1. **Step 1 (Immediate High-Yield Experiment)**:
   Modify `kernels/packed_ternary/gemm_forward_tc.cu` to implement the **K32 slice** (2 MMAs per sync window).
   - Change `kWMMA_K = 16` outer loop step to `kK2 = 32`.
   - Widen `W_smem[16][64]` to `W_smem[32][64]` and `X_smem[64][16]` to `X_smem[64][32]`.
   - Run `nvcc -arch=sm_75 -Xptxas=-v -c kernels/packed_ternary/gemm_forward_tc.cu` locally to verify zero spills and check registers.
   - Run A/B benchmark on Modal T4: `modal run modal_bench_ab.py` or `modal_speedpass_t4.py --phase gigatoken`.
2. **Step 2**:
   Implement direct half-bit register decode in `gemm_forward_tc.cu` and `gemm_backward_dx_tc.cu`.
3. **Step 3**:
   Update compilation flags in `pack_update.py` to `["-O3", "--use_fast_math"]`.
4. **Step 4**:
   Fix the Phase 8 strided load in `subqsa_combine_kernel.cu`.

---
*Report compiled autonomously by Antigravity Deep Performance Auditor.*
