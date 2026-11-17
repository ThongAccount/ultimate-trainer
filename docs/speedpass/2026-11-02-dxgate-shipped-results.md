# dX-gate + batch-scale + shipped-profile results (2026-11-02)

Prereg: `2026-11-02-dxgate-shipped-prereg.md`. Session `ut-dxgate-a`, tag
`d2adefec` (branch `feat/dxgate-shipped-profile`), prebuilt kernels
(`UT_PREBUILT=1`, zero nvcc). Artefacts `results/dxgate_batch.d2adefec.*`.

## Q1 — dX 64×64 vs 32×32 at the real step M=16352 (bit-exact everywhere)

| shape/M | 32×32 (A) | 64×64 (B) | delta | verdict |
|---|---|---|---|---|
| fc1 / 16352 | 75.37 ms | 44.86 ms | **−40.48%** (σ_null 1.18%, bar 2.36%) | WIN |
| fc1 / 16384 | 75.91 ms | 44.91 ms | −40.84% | WIN |
| fc2 / 16352 | 78.02 ms | 45.92 ms | −41.15% | WIN |
| fc2 / 16384 | 78.43 ms | 46.23 ms | −41.05% | WIN |
| head / 16352 | 1020.53 ms | 558.59 ms | **−45.27%** | WIN |
| head / 16384 | 1057.34 ms | 575.10 ms | −45.61% | WIN |

**Conclusion: the `B % 64 == 0` gate in `backward_dx_tc` (custom_ops.py) has
been costing the custom arm ~41–46% of dX time at the real step shape.**
Per the pre-registered wiring rule (faster beyond the bar at M=16352 on fc1
AND head), the gate relaxes to `% 16` on branch `feat/dx-gate-16`. dX in the
step ≈ 6×(51.2+50.7)+649.3 ≈ 1261 ms → expected step saving ≈ 41–46% of
that ≈ **520–580 ms (~11–12% of the 4810 ms custom step)**, to be confirmed
by whole-step A/B on the wiring branch.

## Q2 — batch scaling ("increase multiplication batches")

fwd tc_64 / dX-64 / update v2_64 at M ∈ {16352, 16384, 32768} (fc1/fc2):

| M | fc1 fwd | fc1 dX64 | fc1 up64 | fc2 fwd | fc2 dX64 | fc2 up64 |
|---|---|---|---|---|---|---|
| 16352 | 36.51 | 46.07 | 50.23 | 37.88 | 46.85 | 49.12 |
| 16384 | 36.67 | 46.24 | 50.83 | 38.21 | 46.95 | 49.61 |
| 32768 | 74.14 | 92.91 | 103.54 | 76.25 | 95.34 | 105.52 |

Perfectly linear (2.02× time for 2× rows): **bigger GEMM batches buy no
amortization on T4** — per-row cost is constant. head/M=32768 OOM'd
(6.14 GiB alloc vs 14.56 GiB card; dY+Y alone ≈ 6.6 GB fp16) — dxgate step
rc=1 from that, a capacity limit not a failure; fc1/fc2 curves answer the
question. No batch increase recommended on T4 memory.

## Shipped-path P0 profile (cublas_cublas arm, synthetic, 300 steps)

- B=0: median **589.16 ms**; B=5: median **600.59 ms** → flip counting costs
  **~11 ms (~1.9%)** on the shipped arm.
- Within-run climb is real and NOT allocator fragmentation: ~530 ms early
  (steps 2–50) → ~590–600 ms late; `empty_cache()` at step 150 spiked
  (621.9 ms) then settled back to the climbed level — the climb continues
  past it. Step 1 = 120 s (cuBLAS autotune/warmup, excluded by protocol).
- Profiler (steps 100–200): step device time is dominated by cuBLAS GEMMs
  (aten::mm ~240 ms + turing s1688gemm variants ~275 ms of ~590 ms);
  `update_tc_v2` residual ~0.13 ms/step with PT_UPDATE_BACKEND=cublas.
  → the P1 shipped-path items (scale fold ~35–45 ms, fused CE ~33 ms, dW
  autotune) remain the right targets; the climb (~10% over 300 steps) is
  still unexplained — candidates: clock throttle, cuBLAS workspace growth.
  Loss flat at 11.0 on synthetic data (expected, random labels).

## Next

1. Wire the dX gate fix: branch `feat/dx-gate-16` (one-line dispatch change,
   pure Python → prebuilt bundle stays valid), validated by pytest gate +
   perfgate whole-step (M=16352) vs the measured 4810.507 ms baseline.
2. Wire fwdswizzle-xpose then fwdspill (kernel .cu changes) via per-branch CI
   bundles (`UT_PREBUILT_TAG`), each on its own branch with UT_PREBUILT on the
   matching tag.
