# Phase 2 counterword kernel — findings (stopped per brief's 3-strike rule)

## What was attempted
`kernels/packed_ternary/update_counterword.cu` — derived from `gemm_update_tc_v2_32.cu`
by string replacement: identical WMMA accumulation (same fragments, same smem,
same barriers), with ONLY the counter/flip epilogue rewritten from
per-element `atomicCAS` to one-thread-per-W-word (16 counter updates + one
word-level RMW, no atomics). Local sm_75 build clean: REG:52, SHARED:8192,
0 spills, STACK:0 (the v2 LUT-in-local-memory artifact also disappears).

## Result: PARITY FAILED — not shippable, parked

Gate: `tests/probe_update_counterword.py` (reference gate + W/counter
bit-exact vs v2_32 with counters seeded past threshold).

### Evidence (T4, B=64 K=64 N=64 threshold=4, 3 strikes)
1. fc1 full shape: W DIFF / counter DIFF, 100,754 flips — non-vacuous.
2. Isolated single-element gradient (X[:,16]=1, dY[:,0]=1 → dW[0,16]=+64):
   v2 writes exactly counter[0,16]=-1; counterword writes exactly counter[16,0]=-1.
3. Random full-shape: 1064/4096 counter diffs; only 159/1064 match a pure
   transpose hypothesis; values differ by magnitude (2↔4, 4→0), i.e. wrong
   accumulator SLOT per element, not a transposed index map.

### Diagnosis (best current understanding)
The write indices are computed correctly (`gc = super_c0 + c`,
`counter[gr*in_features + gc]`, verified by hand-trace for li=16 →
counter[0,16]). The DWF→(wq, r_in, c_in) lookup was corrected once to
mirror v2's `warp_r_off=(w%2)*kM, warp_c_off=(w/2)*kN` exactly and the JIT
cache was ruled out (extension name bumped, same result). The residual
divergence pattern (mismatched magnitudes, not a clean permutation) points
at the accumulator content for warps 1/2/3 not matching the element the
epilogue believes it is reading — most likely the WMMA fragment/ldm wiring
in the copied accumulation loop interacts with the epilogue's assumption
about DYS/XS slice indexing in a way that only shows when the REAL multi-
warp path runs (single-element gradients at tile (0,0) and (0,16) pass in
isolation because those hit warps 0 and 2 directly).

### Per the brief's rule
"If a step fails 3 times, stop and report numbers instead of rewriting
kernels." Stopping here. Numbers reported; kernel parked in `tests/`
(never wired into production dispatch).

### What this does NOT change
The Phase 0 conclusion stands and is the campaign's headline: dispatch fwd
runs at 1.60 TFLOPS (2.5% of T4 peak) while pre-decoded W + cuBLAS runs at
33.1 TFLOPS — a 20.7x gap with decode costing only 0.105 ms. The
counterword idea targets the update kernel (46.5 ms of fc1), where the
ceiling is smaller than the fwd/dX path anyway.

### Next step if revisited
Instrument `dW_float_smem` directly: dump the accumulator for a known
single-element gradient through a debug build, compare slot-by-slot against
v2's epilogue reads, THEN re-derive the mapping from measured slots rather
than from the v2 source arithmetic. Estimated 1-2 T4 cycles.
