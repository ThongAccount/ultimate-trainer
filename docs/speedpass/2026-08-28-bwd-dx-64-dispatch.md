# 64×64 Backward-dX Dispatch — 2026-08-28 (Modal T4), REVISED

> Revision note: the original "e2e-neutral / dispatch-dominated" conclusion
> below was WRONG. A clean in-process A/B + CPU/CUDA profiler showed the
> dispatch fix delivers a real **1.16× backward speedup** and that backward is
> **GPU-bound (~95%), not dispatch-bound**. See §CORRECTION at the end and
> HANDOFF.md §16.

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

## CORRECTION — clean in-process A/B (disregard the file-swap A/B above)

The file-swap A/B above (4750 vs 4750) was **confounded by torch .so caching**
— swapping files in-place reuses the already-compiled 64×64 `.so`, so both
"variants" ran the same kernel. A clean in-process A/B (monkeypatch
`custom_ops._dx_tc_64`, same process, compiled kernels stable) gives:

| variant | full backward ms/step |
|---|---|
| 64×64 dispatch | 2764 |
| 32×32 forced | 3208 |

**1.16× speedup, −444 ms/step.**

## Profiler attribution (CORRECTED — overturns §15.3)

CPU+CUDA profile of one step: backward wall ≈ 3063 ms, backward GPU kernels sum
≈ 2.9 s → **~95% GPU-bound; near-zero dispatch gap**. §15.3's "~90% dispatch
overhead / kernels only 480 ms" is retracted as a measurement error.

| kernel | time | share |
|---|---|---|
| `update_tc_v2` (weight update) | 1383 ms | 45% |
| head `backward_dx` (32×32, N_out=50272) | 975 ms | 32% |
| 12 MLP `backward_dx` (64×64) | 559 ms | 18% |

## Why the fix is partial

1. `update_tc_v2` = 1383 ms (45%) is untouched by the dX dispatch change.
2. Head dX is stuck on 32×32: `VOCAB = 50272`, `50272 % 64 = 32`, so the
   `N_out % 64 == 0` routing condition forever excludes it. One head layer
   (975 ms) costs more than all 12 MLP dX layers combined.

Fix options: (a) pad/pad-boundary 64×64 kernel handling non-64-multiples, or
(b) bump `VOCAB` 50272 → 50304 (= 64×786).

## Conclusion (REVISED)

The 64×64 dispatch is correct and yields a **real 1.16× backward speedup**
(−444 ms/step ≈ 13% of backward). Next targets, in order: `update_tc_v2`
(45% of backward) and the head-layer dX (32%, blocked by VOCAB not being a
64-multiple).