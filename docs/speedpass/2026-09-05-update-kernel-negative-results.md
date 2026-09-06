# Session 7 (2026-09-05): update_tc_v2 load-restructure attempts — negative results

Baseline: `d23f16c`, Modal T4. Current profile of a full training step
(total 7517 ms GPU across the profiled window, per-kernel shares):

| kernel | ms/step | share |
|---|---|---|
| forward TC64 | 1371 | ~18 % of all GPU |
| **update_tc_v2 (32×32)** | **1142** | ~15 % |
| backward_dx TC64 | 1061 | ~14 % |

Isolated kernel baselines (tests/bench_update_v2.py, 20 iters, same process):

| shape | baseline ms |
|---|---|
| fc1 (1024→4096) | 46.7–46.9 |
| fc2 (4096→1024) | 47.1–48.3 |
| head (1024→50272) | 562–574 |

## Attempt A: deduplicate tile loads (4 warps → 2 smem copies)

The 4 warps in a block cover a 2×2 grid of 16×16 frags, so warps (0,2) fetch
identical dY tiles and warps (0,1) identical X tiles → each 16×16 half tile
read twice from global memory. Restructured so two warps share one copy
(block-cooperative load, frag ldm kept at 16-half stride to preserve the
measured-fastest WMMA smem access pattern).

**Result: regression.** head 562→~666 ms (+18 %), fc1/fc2 flat to slightly
worse. Reverted.

Interpretation: the duplicate loads were absorbed by L1/L2 (both warps hit the
same lines ~simultaneously; the second read is an L1 broadcast, not DRAM).
Halving *issued* traffic did not reduce *DRAM* traffic, while the more complex
cooperative load loop (index math, longer unroll) added instruction overhead.

## Attempt B: kSub=4 (batch sub-tiling, kBK=64)

Process 4 × kK=16 batch sub-tiles per `__syncthreads` window to quadruple
outstanding global loads per sync. smem 6 KB → 20.5 KB, 45 → 92 registers.

**Result: severe regression.** fc1 47→107 ms (2.3×), head 573→1280 ms (2.2×).
Reverted.

Interpretation: occupancy collapse. 20.5 KB smem + 92 regs cut resident blocks
per SM from ~10 to ~3 on T4 (64 KB smem/SM). The kernel is an occupancy-latency
design; raw in-flight bytes per warp cannot compensate for losing 70 % of
resident warps.

## Conclusions

1. `gemm_update_tc_v2_32.cu` load structure is at a **local optimum**: both
   traffic-reduction and latency-hiding restructures regress on T4.
2. The remaining ~2× on the update path is the dead **64×64 update kernel**
   (`gemm_update_tc_v2.cu`), which halves per-tile DRAM volume by having each
   16×16 frag read shared by 2× the output elements. That is P3 in HANDOFF:
   fix the WMMA col-major addressing bug, then wire dispatch.
3. Alternative: head dY is fp16 16384×50272×2B = 1.6 GB read per step per
   update; a fused `(dX, update)` traversal would read dY once, not twice.

## Artifacts this session

- `tests/profile_complete.py` run on Modal (`/tmp/profile_complete.txt`) —
  full per-kernel + CPU dispatch + phase profile at d23f16c. CPU dispatch is
  negligible (cudaLaunchKernel 1.9 ms total); backward is GPU-bound.
- `modal_validate_update.py`, `modal_profile_complete.py`, `modal_bench_ab.py`
  — local-tree-mount Modal runners (mount, no clone needed).
- A/B method: 3 trials × {base, variant} in one Modal job, report all trials,
  not best-of; run-to-run T4 noise on this kernel is <1 %.
