---
tags: [project/ultimate-ai-model, cuda, kernel-fusion, design]
date: 2026-07-27
status: active
parent: docs
aliases: [fused-ternary-backward-update]
---

# Fused Ternary Backward + Update Kernel Design

## What It Does

The fused kernel (`gemm_fused_backward_update.cu`) combines **backward dX** and **weight update** into a single CUDA C++ launch, closing the performance gap between packed-ternary training and the AdamW baseline.

## Architecture

```
Grid:  (ceil(N_out / 32), ceil(N_in / 32))
Block: 128 threads (4 warps)
```

Each CTA owns a **32×32 tile** of the weight matrix `W[n,k]`.

### Global Tensors

| Tensor | Shape | Access | Description |
|--------|-------|--------|-------------|
| `dY`   | `[B, N]` | read | Upstream gradient w.r.t. output |
| `X`    | `[B, K]` | read | Input activations (FP16) |
| `W_read` | `[N, stride]` | read | Packed ternary, snapshot for dX |
| `dX`   | `[B, K]` | write/atomicAdd | Gradient w.r.t. input (pre-zeroed) |
| `W_mut` | `[N, stride]` | write/atomicCAS | Same alloc, mutated in-place |
| `counter` | `[N, K]` | write/atomicCAS | Int16 discrete optimizer state |

### Shared Memory (10 KB)

| Buffer | Shape | Size | Purpose |
|--------|-------|------|---------|
| `W_smem` | `[32, 32]` half | 2 KB | Decoded W tile, loaded once per CTA |
| `dY_smem` | `[4, 16, 16]` half | 2 KB | dY batch subtile |
| `X_smem` | `[4, 16, 16]` half | 2 KB | X batch subtile |
| `spill` | `[4, 16, 16]` float | 4 KB | Reused: dX WMMA→global bridge + dW counter input |

### Kernel Flow

```
Phase 0:  Load packed W → W_smem (decoded FP16, once per CTA)
  ├── 128 threads cooperatively decode uint32_t → {-1,0,+1} → half
  └── __syncthreads()

Loop over batch (B, step=16):
  ├── Load dY[Bt×32] → dY_smem     (half2 vectorized, block fill)
  ├── Load X[Bt×32] → X_smem       (half2 vectorized, block fill)
  ├── __syncthreads()
  │
  ├── Phase 1 (dX):
  │   ├── WMMA: row_major(dY) × row_major(W) → acc_frag
  │   ├── Store acc_frag → spill SMEM
  │   ├── __syncthreads()
  │   └── atomicAdd dX partial to global dX[B, K]
  │       (multiple CTAs contribute to the same dX element)
  │
  └── Phase 2 (dW):
      └── Manual reduction over batch dimension
          └── Each thread accumulate into 8 private registers
```

```
Phase 3 (post-batch):
  ├── Spill dW registers → SMEM
  ├── __syncthreads()
  └── For each int16 counter pair:
      ├── Skip if both gradients == 0
      ├── Vectorized int32 counter load
      ├── Sign-based update: grad>0 → decrement, grad<0 → increment
      ├── If |counter| > threshold → atomicCAS bit flip + reset
      └── Vectorized int32 counter store
```

## Key Design Decisions

### 1. Separate `W_read` and `W_mut` Pointers

The kernel takes **two** pointers to the same allocation. `W_read` is used for the dX computation (Phase 1) and `W_mut` is mutated via atomicCAS (Phase 3). This ensures no thread reads a weight that another thread has already flipped.

### 2. WMMA for dX, Manual Reduction for dW

WMMA computes `dX[b,k] = Σₙ dY[b,n] × W[n,k]` — a standard row×row GEMM that maps perfectly to Tensor Cores.

The dW computation `dW[n,k] = Σᵦ dY[b,n] × X[b,k]` would require col_major dY × row_major X, which computes `C[b,k]` not `C[n,k]`. A manual reduction over the batch dimension is used instead, accumulating into per-thread registers.

### 3. atomicAdd for dX

Each CTA owns a 32×32 W tile, so `ceil(N/32) × ceil(K/32)` CTAs each contribute a dX partial. atomicAdd on `half` (available since sm_75) resolves these correctly without a separate reduction pass. The caller must zero-initialise the `dX` buffer.

### 4. Fixed ±1 Semantics (v2-compatible)

Unlike v3's magnitude-scaled update, the fused kernel uses the canonical gradient descent direction: `dW > 0 → decrement, dW < 0 → increment`. This matches the scalar and TC v2 semantics for bit-for-bit parity.

### 5. SMEM Reuse

The `spill[4][16][16]` float buffer is used for two purposes across phases:
- **Phase 1→global**: dX WMMA fragment → atomics
- **Phase 3**: dW register file → counter update

This saves 4 KB of SMEM compared to dedicated buffers.

## Compilation

```bash
# Standalone:
nvcc -std=c++17 -O3 --use_fast_math -arch=sm_75 \
     -c gemm_fused_backward_update.cu

# Via PyTorch load_inline (production path):
#   pack_update.py:_load_fused() does this automatically
```

Requires CUDA 11.0+ (WMMA) and sm_75+ (atomicAdd half).

## Fallback Path

When any GEMM dimension is < 16, `backward_update_fused()` falls back to the existing sequential `backward_update()`, which dispatches to separate dX + update kernels.

## Correctness Verification

See `tests/test_gemm_fused.py`:
- Single-step dX vs backward_dx (atol 1e-3)
- 100-step exact match vs sequential pipeline (W/counter bit-identity)
- Gradient direction
- Bit flip behaviour
- Small dimension fallback
- Odd shape dispatch
- Autograd integration via PackedTernaryLinear
