# Update-Kernel Experiments — 2026-08-26 (Modal T4)

Causal isolation campaign for the TC32 weight-update kernel
(`kernels/packed_ternary/gemm_update_tc_v2_32.cu`, 41.6% of training CUDA time).
Harness: standalone CUDA binary (`bench.cu`), no repo code modified; production
kernel reproduced verbatim as `upd_prod`; variants templated on batch step and
barrier policy. Compiled sm_75 on-container (CUDA 12.8).

## Methodology

- **Shapes**: fc1 (B=16384, K=1024, N=4096), fc2 (K=4096, N=1024),
  head (K=1024, N=50272). Grid mapped exactly like the production launcher:
  `x = ceil(in/32)`, `y = ceil(out/32)`, block = 128 threads.
- **Parity**: two tiers per run — threshold=30000 ("no-flip expected": pure
  accumulation must be bit-exact) and threshold=32 (realistic: concurrent flips
  to a shared W-word may legally land in either order → word diffs tolerated,
  counter diffs must be ~0 for identical accumulation order... see defects §4).
- **Determinism probe**: prod run twice from identical state must be
  bit-identical (`prod-vs-prod repeat: counters_differ=0 words_differ=0`) — this
  validated the reference before any variant comparison.
- **Timing**: cudaEvents bracketing kernel-only loops, 3 warmup + 20 timed.

## Results (harness v3/v4 — all variants bit-exact vs prod)

| variant | fc1 | fc2 | head | notes |
|---|---|---|---|---|
| prod (strided loads + syncthreads) | 126 ms | 142 ms | 1592 ms | baseline |
| prodsw (prod loads + syncwarp)     | 204 ms | 245 ms | 2920 ms | barriers REMOVED |
| nobar16 (coalesced + syncwarp)     | 72 ms  | 76 ms  | 896 ms  | |
| synth32 (coalesced + syncthreads)  | 71 ms  | 76 ms  | 887 ms  | barriers KEPT |
| k32_nobar (coalesced, step 32)     | 71 ms  | 75 ms  | 868 ms  | ≤2% vs synth32 |

## Conclusions

1. **Barrier hypothesis DISPROVEN.** Removing barriers with prod's strided loads
   is 60–80% SLOWER. The barriers serialize warps into lockstep bursts that
   preserve locality; they are protective, not costly.
2. **Root cause of the 44% win: load coalescing.** Prod maps lane→element via
   `i = wtid*8 + j` (16 B stride between lanes → ~50% wasted sectors). The fix
   maps consecutive lanes to consecutive half2 pairs (4 B stride). Identical
   elements loaded in both cases → bit-exact parity.
3. **Batch-step 32 rejected**: <2% additional gain, not worth complexity.

## End-to-end transfer (production trainer A/B, commit-pinned)

- `8ebcd50` (baseline): 2510 tok/s, bwd wall 5090 ms/step
- `c7bc4ed` (coalesced): 2461 tok/s, bwd wall 4959–5029 ms/step
- Kernel saving transferred as predicted (−61…−131 ms/step vs −87 predicted)
  but is invisible end-to-end because backward is **~90% CPU-dispatch-bound**
  (bwd wall 5090 ms vs ~480 ms total backward GPU kernels).

## Harness defects caught mid-campaign (do NOT repeat)

1. v1: grid x/y swapped for non-square shapes (fc1 ran 16384 CTAs instead of
   4096); `(int16_t)100000` overflowed to −31072 making the "no-flip" tier flip.
2. v2: rewritten variant load loops loaded only 1-of-32 smem elements; apparent
   parity came from stale-smem reuse across launches. Rule: after ANY rewrite,
   re-run parity AND a determinism probe before trusting results.

## Reproduction

```
modal run modal_bench_update.py   # mounts /src, compiles bench.cu sm_75, runs 3 shapes
# local artifact: /tmp/kexp/bench.cu (ephemeral), archived in this doc's git history
```
