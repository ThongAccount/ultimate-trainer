# Results: hyperopt-audit "all 3" run (2026-11-02)

User request: "Go for all 3 and increase multiplication batches if can."
The 3 items from `chore/hyperopt-audit` @ c117d9f. Master trajectory:
**4810.507 → 3629.323 → 3519.203 → 3395.638 ms** (−29.4% vs measured baseline
`tests/perf_baseline_t4.json`).

## Item 1 — dX `%16` gate fix: DONE, merged @ 098a385

Gate `B % 64 == 0` → `B % 16 == 0` in `backward_dx_tc`
(`kernels/packed_ternary/custom_ops.py`), so the 64×64 dX kernel fires at the
real step M=16352. Prereg `2026-11-02-dx-gate-16-prereg.md`. Validation
ut-gate16b @ 3c113de: pytest **103 passed**, perfgate step_ms
**3629.323 vs 4810.507 = −24.5%**. (First run failed pytest 101/102 — the
perfgate/dispatch fixes lived only on feat/c13-probe-batch; ported as 3eff5f8.)

## Item 2 — shipped-path P0 profiling: DONE (measured, branch feat/dxgate-shipped-profile @ b67b5a0)

Doc `2026-11-02-dxgate-shipped-results.md`, artefacts `results/dxgate_batch.d2adefec.*`:
- dX 64×64 vs 32×32 bit-exact; at M=16352/16384: fc1 −40.5%, fc2 −41.2%,
  head −45.3% (all WIN beyond max(2σ_null,1%)).
- **Batch increase rejected by measurement**: fwd GEMM time scales perfectly
  linearly (2.02× for 2× rows, fc1 36.5→74.1 ms at M 16352→32768) — bigger
  multiplication batches buy nothing on T4. head @ M=32768 OOMs (6.14 GiB
  alloc vs 14.56 GiB card) — capacity limit, not a tuning failure.
- Shipped profile (cublas arm, synthetic, 300 steps): B=0 median 589.16 ms,
  B=5 median 600.59 ms → flip counting ≈ **11 ms (~1.9%)**. The ~530→~590 ms
  climb over steps is real; `empty_cache()` at step 150 did not reset it → not
  allocator fragmentation. cuBLAS GEMMs dominate the step (~515 of ~590 ms).

## Item 3 — wire fwdswizzle-xpose then fwdspill: BOTH DONE, merged

- **xpose** merged @ 44a29a5 (bundle `kernels-fwd-xpose`, run 41399991891
  success): 3-line rasterization transpose in `gemm_forward_tc.cu`.
  Validation ut-xpose: pytest 103, perfgate PASS, fwd rows −28…−33% vs
  baseline, step 3629.3 → **3519.203 ms**. Artefacts `results/gate16.2289f955.*`.
- **fwdspill** merged @ e68b9cd (bundle `kernels-fwd-spill`, run 41409380855
  success, release 16,724,752 B): warp-private spill epilogue ported from
  `tests/fwd_spill_warpprivate.cu`, bit-exact. Prereg
  `2026-11-02-fwd-spill-prereg.md`. Validation ut-fwdspill @ 41812f32
  (`UT_PREBUILT_TAG=kernels-fwd-spill`): pytest **103 passed** (219.7 s),
  perfgate **PASS rc=0**:
  - fc1/fwd 40.23 → 25.849 ms **−35.74%**, fc2/fwd 41.44 → 26.494 ms **−36.06%**,
    head/fwd 499.84 → 341.989 ms **−31.58%** (vs measured baseline; combines
    gate16+xpose+spill);
  - dx/update rows within bar (no-regression control holds);
  - **step_ms 3395.638** (3519.2 → 3395.6 = −3.5% from spill alone;
    4810.5 → 3395.6 = **−29.4%** total).
  Artefacts `results/gate16.41812f32.*`.

## Master state

`e68b9cd` (ls-remote verified): 3eff5f8 (ported fixes), 4a6a350 (measured
baseline), 098a385 (gate16), 44a29a5 (xpose), a942088 (xpose artefacts),
e68b9cd (fwdspill). Per-branch bundle tags keep kernel-changing branches from
being shadowed by the master `.so`.

## Remaining backlog (not in scope of "all 3")

Step-climb cause on the shipped arm (not fragmentation), scale fold, fused CE,
dW autotune.
