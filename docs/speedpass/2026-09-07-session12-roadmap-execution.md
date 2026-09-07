# Session 12 (2026-09-07) — Roadmap execution: 6 experiments, 0 wins

Executed every experiment in the session-11 deep-audit roadmap, with paired
Modal T4 A/Bs, gates, per-change compile verification. **No optimization
survived measurement.** Confirms prior-session conclusion: all three main
kernels (fwd TC64, dX TC64, update TC32) sit at microarchitectural local
optima on T4.

## Experiments

### E1. Forward TC64 K-slice 16→32 — REVERTED (+31%)
Hypothesis: same K-widening that won dX at K32 transfers to forward.
Implementation: tiles 20→24KB, 88→116 regs, occ 3→2 blocks/SM.
Result: fc1 +30%, head +31% slower (3 trials, consistent). Forward has NO
smem/reg headroom — 16KB spill leaves none. dX won because it started at 8KB.

### E2. Half-bit register decode LUT — FALSIFIED (1.06x), not implemented
Hypothesis: W 2-bit decode dominates forward load loop.
Direct probe: pre-decoded fp16 W entry vs packed identical kernel
(`tests/scratch_fwd_fp16w.cu`): head 638→604 ms, 1.06x. Decode cost ≈6%
kernel-level ≈ <1% e2e. Not worth register LUT complexity.

### E3. Compiler flag harmonization (-O3 --use_fast_math) — REVERTED (0%)
Changed dX TC64 + update TC32 loaders from -O2 to -O3 --use_fast_math
(matching fwd/SubQSA). Regs/smem identical: dX 96/12KB, update 53/8KB.
Bench: head dX 516.5 vs 517.0, update 558.3 vs 558.8 — noise. Compiler
flags already at practical limit for these kernels.

### E4. Forward 16KB spill → 4KB (occupancy 3→7) — REVERTED (+2%)
Hypothesis: spill caps occupancy at 37.5%; shrink frees blocks.
Implementation: per-round spill reuse (4 rounds × sync), 64 regs / 8KB.
Result: head −2.3% slower, fc1 +1% — occupancy gain did NOT translate.
Combined with E1: occ 3→2 slow AND occ 3→7 slow ⇒ forward not occupancy-
bound. The spill itself isn't the cost; the store round-trip through smem
is inherent to WMMA float accumulators.

### E5. SubQSA Phase 8 stride-D coalesce — SKIPPED (not on benchmark path)
Verified `train_gigatoken.py` has zero SubQSA references; the speedpass
benchmark is pure packed-ternary transformer. SubQSA combine is a separate
workload; no e2e impact on this branch's benchmark. Not pursued without
a measurement path.

### E6. dX TC64 K32→K64 — REVERTED (+20%)
Hypothesis: 4 MMAs/sync halves sync count again.
Implementation: tiles 12→20KB, 96→122 regs, occ 5→3 blocks/SM.
Result: head 615→619 vs base 513.5, fc1 +17% — register/smem cost overtook
sync savings. K32 is the dX sweet spot.

## Updated bottleneck model (evidence)

- fwd TC64, dX TC64, update TC32 all ≈2.5–3 GFLOP/ms on T4.
- Sync-count widening: helps only when smem headroom exists (dX K32 ✓,
  fwd ✗, dX K64 ✗ at occ collapse).
- Occupancy is NOT the forward lever (3 vs 7 blocks: both slow).
- Decode is NOT the forward lever (1.06x).
- Compiler flags: nothing left.
- These kernels are latency-bound in the 16-deep WMMA smem access pattern
  shared by all three; no further micro-restructure with >1% e2e upside
  was found this session.

## Session outcome

| Exp | Result | Decision |
|---|---|---|
| E1 fwd K32 | +31% | REVERT |
| E2 decode | 1.06x | FALSIFIED, skip |
| E3 flags | ~0% | REVERT |
| E4 spill | +2% | REVERT |
| E5 SubQSA P8 | n/a | SKIP (off path) |
| E6 dX K64 | +20% | REVERT |

Session net e2e: **0.0%** (no shipped changes; prod unchanged at 7d07369 +
dX K32 commit). No correctness gates weakened; all probes/pytest passed.

## Recommended next (stops the roadmap)

All ranked micro-optimizations are exhausted. Remaining upside is structural,
none with measured support on this hardware:
- double-buffered smem (needs cp.async — sm_80+, T4 sm_75 can't) ✗
- algorithmic: larger K per CTA via split-K deserves ONE measured trial on
  dX (grid ×2, atomic pre-reduce) — but +20% from occ loss warns against.
- accept current throughput; move to correctness/convergence work.
Per rules: stop conditions met — remaining ideas below meaningful threshold
or require disproportionate complexity.