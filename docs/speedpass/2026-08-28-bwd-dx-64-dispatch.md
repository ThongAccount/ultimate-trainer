# 64×64 Backward-dX Dispatch — 2026-08-28 (Modal T4)

## Hypothesis

The 64×64 backward-dX kernel (`gemm_backward_dx_tc.cu`, fixed in `113b40b`)
was loaded but never dispatched: `_load_tc_if_needed()` clobbered `_dx_tc_fn`
(64×64) with `_dx_tc_32_fn`, so `custom_ops` only ever exposed 32×32 for
backward dX in training.

## Change (commits `b23359b` + `242a665`)

- `pack_update.py`: split `_load_dx_tc` (64×64) from `_load_dx_tc_32`; stop
  clobbering `_dx_tc_fn`; `_load_tc_if_needed` loads both independently.
- `custom_ops.py`: expose `_dx_tc_64` separately; `backward_dx_tc` routes to
  64×64 when B, N_out, K are all 64-multiples.
- Fixed kernel bugs (W_smem transpose, leading-dim, spill race) carried forward
  from `113b40b`.

## Validation (tests/validate_bwd_dx_64.py, Modal T4)

Correctness vs torch reference — all shapes PASS, err ≈ 32×32:

| B | N_out | K | err64 |
|---|---|---|---|
| 16384 | 4096 | 1024 | 0.0623 |
| 16384 | 1024 | 4096 | 0.0308 |
| 16384 | 1024 | 1024 | 0.0308 |
| 512 | 4096 | 1024 | 0.0313 |
| 256 | 1024 | 4096 | 0.0156 |
| 128 | 64 | 64 | 0.0039 |
| 64 | 64 | 64 | 0.0038 |

## Kernel benchmark (gigatoken fc1: B=16384, N=4096, K=1024)

- 64×64: **43.70 ms**
- 32×32: **75.78 ms**
- Speedup: **1.73×**

## End-to-end A/B (train_gigatoken, same T4 container)

| variant | ms/step | tok/s |
|---|---|---|
| HEAD (dispatch fix) | 4750.1 | 3,449 |
| BASELINE (HEAD~1) | 4749.8 | 3,449 |

**Result: within noise.** Confirms §15.3 — the kernel speedup vanishes inside
CPU/autograd dispatch overhead (~90% of backward wall is dispatch, not kernels).

## Reconciliation with §6 (prior +18% regression)

§6's regression was at B=512 (SubQSA probe, grid underfill: 128 vs 512 CTAs).
At gigatoken B=16,384 there is no underfill, so the 1.73× kernel speedup holds
and no e2e regression reproduces. The change is numerically safe and
dispatch-correct, but e2e-neutral because the real bottleneck is autograd
dispatch, not GPU kernel work.

## Conclusion

Kernel micro-optimization is saturated. The 64×64 backward-dX kernel is now
correct and dispatchable (removes the clobber NPE-class defect), but the next
piece of work must target CPU/autograd dispatch overhead (§15.3) — the
backward GPU kernels sum to ~480 ms/step while backward wall is ~3350 ms/step.