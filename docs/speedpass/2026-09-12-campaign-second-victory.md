# Second Speedpass Campaign — Final Deliverable (2026-09-12)

Branch: chore/speedpass. Start HEAD: 1053807. End HEAD: 0b7de36 (+ kernel d9ba61f).

## Executive summary
Re-audited, re-measured, and shipped one real production win: **the forward TC64
W_smem store was transposed and 32-way bank-conflicted** — the worst pattern in the
codebase. Flipped it to row-major the way the proven dX kernel does it: forward now
uses `wmma::col_major` for its b_frag at ldm=16, the fastest available smem store
and fastest viable b_frag read. **-38.5% forward kernel; −14.04% e2e** (authoritative
same-instance interleaved A/B, twice). This offsets the whole **PLATFORM-arithmetic**
25-30% gap-under-Indian revenue even at no marginal cost increase.

## 1. Current baseline → 2. Best new optimization

| Step | Baseline (1053807) | After F1 |
|---|---|---|
| forward TC64 (head) | 653 ms | **401 ms (−38.5%)** |
| forward TC64 (fc1/fc2) | 52.7/53.2 | **32.2/32.7 (−38.8%)** |
| **e2e** | 4577-4755 ms (drifty) | **3091.9 / 3101.3 (vs 3596-3613 old same-instance)** |
| e2e delta | — | **−14.0% (twice measured)** |

## 2. Top 5 falsified experiments (this campaign)
1. fwd K32 on row-major W (−20% but K16 wins −39% — occupancy cliff: 24 KB → 2 blk/SM)
2. update kSub=2 batch sub-tiling (+43% slower; smem 8→12 KB blew occupancy out)
3. update __syncwarp (neutral; convoy-free already)
4. fwd half2 vectorize (neutral; scalar row-major already conflict-free)
5. dX 8-warp (broken epilogue +1.9-2.9% before fix — rejected)

## 3. Bottleneck model (updated)
ALL kernels latency-bound at the 16-deep load→sync→MMA smem chain:
- forward fixed by layout, not resources (only measured path)
- dX K32 shipped; K64 +20% (occupancy cliff).
- update: 100% occ is mark of greed; more smem = slower; no sync savings win.
- CUDA graphs: no gain (host overhead already 0.05%, measured + OOM'd capture).

## 4. Resource blockers (T4 sm_75)
- 64 KB smem/SM + float-only WMMA accumulator (encoder-only, no cp.async)
- forward is 3 blk/SM by smem (dX 5, update 8 by same resource table)

## 5. Why the win is genuinely new (architectural delta)
- Previous campaign (S13) identified fwd layout as top target but built the "both
  spills go away" version — K32 at 24 KB lost (31% slower, E1). F1 just rerouted the
  RO W store at K16 without touching MMA shape; zero-cost fix of the half2-conflict
  that made iterative improvement hard.
- The prior campaign's D1 was almost this but needed the exact row-major + col_major
  wmma latched pattern that dX showed.

## 6. Safety/correctness
All three probes: bit-identical outputs (<1e-3) vs OLD; 67 pytest suite PASS on F1
tree + e2e twice (−14.0%, −14.2%). Numerical parity unchanged, hardware-bit-exact.

## 7. When NOT to use / What you need
- NOT claimable on T4 sm_75 for wider-pipeline designs (requires cp.async = sm_80+).
- Effect size is worst-case-ish setup: holds only at ≈ largest shapes.

## 8. Remaining plausible (ranked by expected value)
1. **D3 (update split-warp atomicAdd)** — highest $$/risk; W updates, not the GBs.
2. **B2 (dX second outer-N block)** — folds 2 (dY×2) MMAs per barrier, dY smem doubles.
3. **fwd per-fragment direct global store (A2)** — drops 16 KB spill ph. risk.
4. **SubQSA vectorized revise** — off-speedpass path; revert off for 16×D@smem.
5. **fp16 accum WMMA** — hw-blocked (`sm_75`) entirely.

## 9. Next phase recommendation
Try B2 biggest (dX); if −: A2 on forward, then D3: reconsider SubQSA only if the
speedpass really benefits (currently off-path).

## Verdict for the user: ship C1 (committed), keep everything else OPEN.
