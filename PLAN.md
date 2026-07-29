# PLAN — Packed Ternary Discrete Optimizer

## Current State (2026-07-28)

### What Works ✅
- **Convergence**: Shakespeare LM beats AdamW (loss 2.54 vs 2.61, ppl 5.8 vs 6.1)
- **TC kernels**: All phases use WMMA, 64x64 tiles (4x fewer CTAs)
- **32x32 legacy**: Preserved for dims 16-63
- **CUDAGraph**: Captures forward+backward, 4x faster than autograd
- **Gigatoken**: GB/s tokenization, integrated
- **All P0 bugs fixed**: syncthreads deadlock, CUDAGraph gradient, alignment guards, SMEM zeroing, odd tail columns

### Benchmark (T4, 51M param model, B=32, SEQ=512)

| Phase | Old (32x32) | Expected (64x64) |
|---|---|---|
| Forward (6 layers + head) | 2.8s | **~0.7s** |
| Backward | 5.2s | **~1.3s** |
| Loss + overhead | 0.4s | 0.4s |
| **Total** | **~8s** | **~2s** |

### Theoretical Limits (4096×4096, batch=32)

| Metric | Value |
|--------|-------|
| Memory-bound min | 0.243ms (72.9 MB traffic) |
| Compute-bound min | 0.033ms (2.1 GFLOP) |
| Current best (manual+cudagraph) | 1.63ms |
| Gap vs memory min | **7×** |

---

## What Was Done in This Session

| # | Bug | Fix | Severity |
|---|-----|-----|----------|
| 1 | CUDAGraph wrong gradient | Separate target buffer | Critical |
| 2 | syncthreads deadlock | Predicated kernel body | Critical |
| 3 | TC forward missing %16 guard | _tc_ok() in dispatch | Critical |
| 4 | v2 partial-batch SMEM zeroing | Copy v3 zero-fill block | High |
| 5 | v2/v3 odd tail column skipped | Tail handler for last odd col | High |
| 6 | 32x32 tile 40x slowdown | 64x64 tiles (4x fewer CTAs) | Perf |

## Next Steps

1. **Benchmark 64x64 on T4** — Pull, clear cache, run tests
2. **Profile remaining 7x gap** — Nsight Systems or detailed CUDA events
3. **Wire gemm_forward_packed.cu** — For dims < 16 or when TC not available
4. **Register missing register_fake** — For update_tc_v2/v3
5. **Unify update semantics** — v2 vs v3 inconsistency
6. **Scale to TinyStories** — 15M param from-scratch training

## Architecture Notes

### Dispatch Priority
1. 64x64 TC (dims >= 64, %64==0)
2. 32x32 TC (dims >= 16, %16==0)
3. Scalar fallback (any dims)

### Memory Budget
| Component | Bits/param |
|-----------|-----------|
| Weight | 2 (packed ternary {-1,0,+1}) |
| Counter | 16 (int16) |
| **Total** | **18** vs 96 for AdamW |
