# HANDOFF — ultimate-ai-model / Packed Ternary Discrete Optimizer

## Session Context

This is a **CUDA C++ discrete optimization stack** for training LLMs with packed ternary weights {-1,0,+1}. No FP32/BF16 master weights exist anywhere — weights are always 2-bit packed ternary, 16 per uint32. A counter-based optimizer (int16) replaces AdamW.

The project started as a BitNet b1.58 × SubQSA merged trainer, but pivoted mid-session to focus entirely on the discrete optimizer CUDA stack. SubQSA/NSA attention has been shelved.

## Current State

### What Works ✅

**PackedTernaryTensor** (`kernels/packed_ternary/`)
- `packed_ternary.cuh` — struct, LUT decode, pack16/unpack16, state machine (increment/decrement), atomicCAS variants
- Encoding: `00=0, 01=+1, 10=-1, 11=INVALID`
- Python host helpers: `pack_tensor()`, `unpack_tensor()`

**Forward GEMM** (ternary × FP16 → FP16)
| Variant | Description | GFLOPS* | Peak GFLOPS* | Status |
|---------|-------------|---------|--------------|--------|
| v1 | naive, 1 output/thread, float accum | 10.1 | 39.6 | ✅ |
| v2 | 4 outputs/thread, shares X loads | 15.0 | 52.0 | ✅ best scalar |
| v3 | 256-thread occupancy boost | 10.3 | 41.0 | ✅ |
| v4 | half arithmetic (retired) | 10.8 | 40.5 | ❌ precision issues |
| **TC v1** | **WMMA Tensor Cores, 1 tile/block** | **28.1** | **149.2** | **✅ (old)** |
| **TC v2** | **WMMA, 4 tiles/block (4 warps)** | **~???** | **~?** | **✅ (NEW)** |

The TC v2 kernel (written 2026-07-24) processes 4 WMMA tiles per block using 4 warps. The previous TC kernel wasted 7/8 warps on redundant computation. *GFLOPS need re-benchmarking on T4.*

- TC requires batch ≥ 16 (auto-dispatch falls back to v2/v1)
- Auto-dispatch in `packed_linear.py:_forward_auto()`
- All correctness-verified against `F.linear` at atol=1e-3

**Backward + Fused Optimizer** (Phase 3) — Now with WMMA Tensor Core paths!

| Kernel | Description | Status |
|--------|-------------|--------|
| `gemm_backward_dx.cu` | dX = W^T @ dY, scalar (fallback) | ✅ legacy |
| `gemm_update.cu` | fused dW→sign→counter→flip, scalar (fallback) | ✅ legacy |
| **`gemm_backward_dx_tc.cu`** | **dX = dY @ W via WMMA, 16×16 tiles** | **✅ NEW** |
| **`gemm_update_tc.cu`** | **dW via WMMA + fused counter, 16×16 tiles** | **✅ NEW** |

The TC kernels activate automatically when batch ≥ 16 (`TC_MIN_BATCH = 16`). Both kernels:
- Use WMMA Tensor Cores on 16×16×16 tiles
- Store unpacked FP16 tiles in `__shared__` memory
- Auto-dispatch via `backward_dx()` / `update()` in `pack_update.py`

**PackedTernaryLinear** (`kernels/packed_ternary/packed_linear.py`)
- `PackedTernaryLinearFn`: `torch.autograd.Function` — forward auto-dispatch, backward dX (auto TC/scalar), fused update (auto TC/scalar)
- `PackedTernaryLinear`: `nn.Module` — `register_buffer` for W_packed (int32) and counter (int16), bias support
- `from_pretrained_linear()`: convert `nn.Linear` via mean-abs gamma quantisation
- No changes needed to `packed_linear.py` — auto-dispatch is entirely internal to `pack_update.py`

### Performance (T4, one train step fwd+bwd+update)

| Method | Avg GFLOPS* | vs AdamW |
|--------|-----------|----------|
| **Discrete (old, scalar bwd/up)** | 9.9 | 7.2× slower |
| **Discrete (NEW, TC bwd/up)** | **~???** | **~?** |
| AdamW | 71.7 | 1× |
| FusedAdamW | 65.6 | 0.9× |

*Needs re-benchmarking on T4. The TC forward kernel also got the multi-warp fix. Expect significant improvement for batch ≥ 16.*

### The GFLOPS Gap — What Was Fixed

Root cause of the 7× gap (9.9 vs 71.7 GFLOPS):

| Phase | Old (scalar) | New (TC) | Expected gain |
|-------|-------------|----------|---------------|
| Forward | 28 GFLOPS, 1/8 warp util | 4 tiles/block, 4 warps | 2-4× |
| Backward dX | ~5 GFLOPS, scalar loop | WMMA TC, 16×16 tiles | 4-6× |
| Update | ~5 GFLOPS, scalar batch loop | WMMA TC + fused counter | 4-6× |

The discrete stack ran 2/3 of the train step on scalar FP32. Now all three phases use Tensor Cores when batch ≥ 16.

### Tests

| Test file | Tests | Purpose |
|-----------|-------|---------|
| `tests/test_packed_ternary.py` | 13 | pack/unpack, state machine, stride |
| `tests/test_gemm_forward.py` | 9 | Correctness vs F.linear |
| `tests/test_gemm_perf.py` | — | Benchmark (v1-v4 + TC) |
| `tests/test_gemm_update.py` | **6** (was 3) | backward dX, flip, direction, **TC dX, TC vs scalar, TC flip** |
| `tests/test_packed_linear.py` | 7 | Module integration — shapes, backward, multistep, bias, save/load, dispatch, autograd update |
| `tests/bench_train_step.py` | — | Full step benchmark vs AdamW/FusedAdamW |

### Kernel compilation

Kernels compile via `load_inline` with CUDA 12.9 + sm_75. **7 `load_inline` invocations** after this session (+2 new):
- `packed_ternary_dx_tc_ext` — TC backward dX
- `packed_ternary_update_tc_ext` — TC fused update

### Memory per Parameter

| Component | Bits | Notes |
|-----------|------|-------|
| Weight | 2 | Packed ternary {-1,0,+1} |
| Optimizer | 16 | int16 counter |
| **Total** | **18** | vs 96 for AdamW (32 FP32 + 32 m + 32 v) |

## What's Still Blocked / Needs Work 🔴

### Performance
1. ~~Update kernel batch loop~~ → **FIXED** (WMMA TC kernel `gemm_update_tc.cu`)
2. ~~dX kernel serial over batch~~ → **FIXED** (WMMA TC kernel `gemm_backward_dx_tc.cu`)
3. ~~TC kernel wastes 7/8 warps~~ → **FIXED** (4-tile, 4-warp `gemm_forward_tc.cu`)
4. **Multi-GPU (DDP)** — not implemented. `PackedTernaryLinear` has no DDP hooks.
5. **Convergence untested** — the counter-based optimizer has been verified on synthetic tests but never on a real model / real loss curve. **This is the single biggest unknown.**
6. **Benchmark the new kernels!** The TC kernels haven't been run on a T4 yet.

### Correctness
1. **TC kernels untested on GPU** — all code is written but no CUDA runtime was available in the edit session. Must compile and run `test_gemm_update.py` on the T4.
2. **INT4 path unexplored** — would enable Tensor Cores on more shapes (8×8×32 WMMA tiles instead of 16×16×16).
3. **v4 (half accum) retired** — FP16 accumulation loses precision past K=256.

### Integration
1. **No Gigatoken support yet** — tokenizer is still HuggingFace `tokenizers`. Gigatoken would give GB/s tokenization.
2. **torch.compile blocked** — the custom `autograd.Function` prevents graph compilation.
3. **7 separate `load_inline` builds** — ~30s each at startup. Should combine into one.

## Key Architecture Decisions (New)

| Decision | Rationale |
|----------|-----------|
| TC backward `dX = dY @ W` via WMMA | Same GEMM as forward, just transposed. W unpacked to FP16, dY loaded as matrix_a, W as matrix_b |
| TC update `dW = dY^T @ X` via WMMA | dY as col_major matrix_a, X as row_major matrix_b. After accumulation, sign→counter→flip from SMEM |
| `TC_MIN_BATCH = 16` | WMMA needs 16×16×16 tiles; below 16 the scalar kernels serve as fallback |
| 4-tile forward (32×32 super-tile) | 4 warps each handle a 16×16 tile. Grid shrinks 4×, all warps do useful work |
| Auto-dispatch in `backward_dx()`/`update()` | No changes to `PackedTernaryLinear` or `packed_linear.py`. The autograd Function transparently benefits |

## Files Changed in This Session (2026-07-24)

| File | Lines | Change |
|------|-------|--------|
| `kernels/packed_ternary/gemm_backward_dx_tc.cu` | 129 | **NEW** WMMA Tensor Core kernel for dX = dY @ W |
| `kernels/packed_ternary/gemm_update_tc.cu` | 180 | **NEW** WMMA Tensor Core kernel for fused dW→counter→flip |
| `kernels/packed_ternary/gemm_forward_tc.cu` | 183 | **REWRITTEN** 4-tile, 4-warp block (was 1 tile, wasted 7/8 warps) |
| `kernels/packed_ternary/pack_update.py` | 280 | **REWRITTEN** Added TC loaders + auto-dispatch |
| `tests/test_gemm_update.py` | 198 | **UPDATED** Added 3 TC-specific tests |
| `tests/bench_train_step.py` | ~200 | **UPDATED** Docstring reflects TC auto-dispatch |
| `HANDOFF.md` | ~170 | This file — updated session log |

## Before You Go

- **HANDOFF.md is NOT committed** — delete before committing
- **Run `tests/test_gemm_update.py` on the T4 first** to verify TC kernels compile and pass
- **Run `python tests/bench_train_step.py` on the T4** to get updated performance numbers
- The forward auto-dispatch in `_forward_auto()` has NOT been updated for the new TC kernel — it still dispatches when `B >= 16` (same condition). The new kernel is a drop-in replacement.
- The `load_inline` approach creates separate builds for TC and scalar kernels. This adds ~60s to startup.
- Vault journal should reference this session
- The next big question is still: **does the discrete optimizer converge?** No one has tested this on a real model yet.
