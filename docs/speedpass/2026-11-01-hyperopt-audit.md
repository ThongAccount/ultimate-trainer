# Hyperoptimization audit — 2026-11-01

Full audit of what to hyperoptimize next in ultimate-trainer, after the
C13 probe batch (`feat/c13-probe-batch` @ 71f6a55, see
`2026-11-01-c13-probe-batch-results.md`). Written on `chore/hyperopt-audit`
from master `b988bfe`.

Note: `HANDOFF.md` at the repo root is gitignored (never committed after
`5b1670f`); its §25/§26 dead-end lists survive only as quotes inside the
speedpass docs, which are the durable record cited below.

## 1. Bottleneck model of the step

Two configurations exist; do not conflate them:

- **custom arm** (default env: `PT_BACKEND` unset → `custom`,
  `cublas_backend.py:19`): whole step **4810.507 ms** (10-trial median,
  `tests/perf_baseline_t4.json`, measured T4 @ ee9cefe). Per-kernel medians
  (B=16384 isolated): fc1 fwd 40.226 / dx 51.214 / update 55.724; fc2
  41.437 / 50.677 / 54.879; head 499.841 / 649.306 / 700.429.
  6 layers × (fc1+fc2) + head ⇒ kernel sum ≈ **3614 ms**; the remaining
  ≈ **1196 ms** is glue: embedding, norms, gelu, 26 `* scale` passes,
  logits materialization and cross-entropy (16352×50272 fp16 ≈ 1.6 GB out,
  plus the CE reduce), optimizer, launch overhead. Never attributed.
- **shipped arm** (`PT_BACKEND=cublas`, `PT_UPDATE_BACKEND=cublas`; the
  `cublas_cublas` arm of `colab_step_ab.py`, labeled "shipped" there):
  **573.7 ms** @ B05prod300 (vs 4666.7 ms custom_custom same VM, 8.13×).
  cuBLAS replaces the custom fwd and dX; update is cuBLAS too. Within-run
  climb 510→575 ms by step 99 (`results/official_suite.B05prod300.log`) is
  unattributed. Historical shipped reference 510.9 ms.

Head layer ≈ 51% of the per-kernel sum. Custom kernels achieve 2.4–3.4
TFLOPS ≈ 4–5% of the T4's 65 TFLOPS peak.

Key dispatch fact (verified in code): the real step feeds
`x = data[:, :SEQ-1]` (`train_gigatoken.py:173`), so the GEMM M dimension is
**16352** (32×511), not 16384. `backward_dx_tc` takes the 64×64 dX kernel only
when `B % 64 == 0` (`custom_ops.py:88`); 16352 % 64 = 32 → **the custom arm
has always run dX on the 32×32 kernel**, never getting the measured 1.73×
64×64 gain (fc1 43.70 ms vs 32×32, `2026-08-28-bwd-dx-64-dispatch.md`).

## 2. Ranked backlog

### Shipped path (target 573.7 ms) — cuBLAS arm
- **P0 (measurement only):** profile the shipped step with `torch.profiler`;
  attribute the 510→575 ms within-run climb (empty_cache intervention at
  step ~150); same-VM B=0 vs B=5 A/B to separate flip-counting (~1.6 ms by
  bytes) from drift. No kernel change needed.
- **P1:** fold the 2⁻⁵ `scale` into decode/GEMM alpha — ~35–45 ms (6–8%);
  bit-exact only outside fp16 subnormals, quantify first.
- **P1:** fuse cross-entropy + scale — ~33 ms (~6%); breaks bit-stable gate
  values → needs loss-trajectory gate + re-baseline.
- **P1:** head dW GEMM autotune (80.7 ms dW vs ~51 ms fwd for equal FLOPs)
  → up to ~50 ms; summation order changes → sign-disagreement-rate gate,
  not bit-exact.
- **P2:** resolve conflicting fc1 fwd measurements (4.15 vs 7.33 ms);
  skip layer-0 dX + embedding/LN grads (~10 ms, scope question); fuse LN
  casts (~10 ms).

### Custom arm (target 4810 ms)
- **P1 — dX `%16` gate fix (biggest single item):** relax the `B % 64 == 0`
  gate in `backward_dx_tc` to `% 16` (the 64×64 kernel has internal bounds
  guards — same argument as the update v2_64 `%16` gate). Predicted
  630–1070 ms (13–22% of step) using the measured 1.73–1.85× 64×64 ratio.
  One-line change + parity + A/B at M=16352.
- **P1 — wire fwdswizzle-xpose:** grid transposed so n varies fastest
  (3-line change, bit-identical SASS, head fwd −19.90%, fc2 −2.26%,
  fc1 −1.20%, all bit-exact). ≈ −108 ms ≈ 2.2% of the custom step.
  Apply the same rasterization to dX (head dY re-stream ~26 GB; expect
  fc-range gains, measure).
- **P1 — wire fwdspill:** warp-private spill epilogue, smem 20480→8192 B,
  occ 3→6 CTAs/SM, fc1 −14.36% / fc2 −13.97% / head −7.00%, bit-exact.
  ≈ −104 ms ≈ 2.2% of the custom step.
- Wiring order: dX gate fix first (swizzle must be measured on the 64×64 dX
  at 16352 rows), then xpose, then fwdspill — each on its own branch behind
  bit-exact parity + same-VM whole-step A/B vs the 4810.507 ms baseline.
  Run one combined xpose+fwdspill probe before assuming the wins add.

## 3. Dead ends — do NOT revisit (evidence pointers)

- fwd K32 widening +31% (session 12); fwd spill-to-4KB with CTA barriers +2%
  (E4) — fwdspill won by removing barriers, not occupancy; dX K64 +20%
  (20 KB smem); update kSub=2 +44%, kSub=4 2.3× slower, tile dedup +18%,
  64×64 update wiring bench-neutral (`1baa097`)
- fwd W decode LUT 1.06× scratch; compiler flags byte-identical SASS;
  fp16 accumulator impossible on sm_75 (float WMMA acc is HW)
- fwd/dX occupancy >5 blocks/SM: latency not occupancy-bound in that range
- residency/cudaMemcpy 0%; persistent kernel neutral (launch not the
  bottleneck); cudaGraphs ~0%
- 32×64 tile dX: 691% numerical error (tag freeze-experiment-32x64-tile)
- fusing dX into update epilogue: mathematically invalid
- split-K update, 4-wide counter CAS, VOCAB 50304: falsified
- counter-word update: parked after three failures
- **fwdwdedup +9–10%** (bit-exact, 2026-11-01, `results/probe_batch.d1485f8d.fwdwdedup.log`)
- **uppipeline**: reference-transition gate FAIL — variant drops dead_zone
  (2026-11-01); class not formally closed but that variant is out
- Phase 3 ncu profiling: permission-gated on hosted VMs

## 4. Not optimizable here

float WMMA accumulator (HW), cp.async on sm_75 under load_inline (fragile),
ncu (gated), head update (already cuBLAS in shipped path), M=16352 padding
(cosmetic).

## 5. Operational rules for the next sessions (learned 2026-10-31/11-01)

- Kernels compile in CI only: `build-kernels.yml` triggers on **master
  pushes** touching kernels/** (+ `workflow_dispatch`), publishes to release
  `kernels-latest`; VMs load it via `tools/fetch_prebuilt.py`
  (`UT_PREBUILT=1`, zero nvcc).
- **Hazard:** on a kernel-changing branch the master bundle's `.so` is
  touched newer than the branch sources → ninja would silently load the
  stale master kernel. Kernel-wiring branches MUST run with
  `UT_PREBUILT=0` (or add a `workflow_dispatch` input for a per-branch
  release tag — not yet implemented).
- One T4 at a time, no polling; pre-register (commit doc with
  hypotheses/thresholds/falsifiers) BEFORE allocation; thresholds never
  changed after measurement; rc=2 COMMIT-MISMATCH artefacts are never
  committed.
- Timing claims only via `tests/bench_protocol.py::ab_median`:
  10-trial paired interleaved median, warmup 10, `_sleep` settle, A/A null
  arm; decision bar |Δ| > max(2σ_null, 1%); bit-exact parity gate first and
  non-vacuous (seed counters past threshold).
- Scratch in /workspace/temp only; torch not importable locally — GPU
  checks run only on Colab.

## 6. Next single T4 session (≈50–60 min)

1. P0 shipped-path profiling (torch.profiler on the cublas_cublas arm,
   empty_cache intervention, B=0 vs B=5 A/B) — new script.
2. dX `%16` gate-fix A/B on the custom arm at M=16352 — branch with
   `UT_PREBUILT=0`, parity + 10-trial A/B vs the 51.214/50.677/649.306 ms
   dX baseline rows.

Both are pre-registration-first; artefacts to `results/` with provenance.
