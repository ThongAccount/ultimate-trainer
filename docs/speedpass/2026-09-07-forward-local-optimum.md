# Session 12 — Forward TC64 falsified hypotheses (2026-09-07)

All measured on Modal T4, paired A/B, same process. Forward is the #1 kernel
(1,371 ms / 37.6% of step). Three hypotheses tested directly:

| # | Hypothesis | Change | Result | Verdict |
|---|---|---|---|---|
| 1 | K-slice 16→32 halves sync count (proven on dX) | tiles 20→24KB, 88→116 regs, occ 3→2 | fc1 +30%, head +31% | **FALSIFIED** |
| 2 | 16KB spill caps occupancy; shrink to 4KB | occ 3→7 blocks/SM | head −2.3% (slower), fc1 +1% | **FALSIFIED** |
| 3 | W 2-bit decode dominates load loop | fp16-W direct (scratch) | head 638→604 ms = 1.06x | **FALSIFIED** (<1% e2e) |

## Diagnosis

Opposite occupancy moves both regress (3→2 blocks slow, 3→7 blocks slow):
forward is NOT occupancy-bound. Sync-window widening hurts (+31%): the same
change that won dX at K32. fp16 W decode saves only 6% kernel-level: decode
NOT the cost. Forward is bound by the X/dY row-major strided smem tile access
(strips 64×16 stride-K), which all three kernels share — consistent with
dX/update sitting at the same ~2.5–3 GFLOP/ms ceiling.

## Conclusion

Forward TC64 at a local optimum for the resource/loop structures tested.
Expected e2e headroom from micro-restructure ≈ 0–1%. Exp2 (half-bit decode)
falsified by direct fp16-W probe — do not implement.

## Evidence files

- `tests/scratch_fwd_fp16w.cu` + `tests/scratch_fwd_fp16w.py` — fp16-W probe
- `modal_bench_fwd_ab.py` — A/B harness (K16 vs K32, then spill vs base)
- baseline worktree `/tmp/uam_fwdbase` @ 7d07369