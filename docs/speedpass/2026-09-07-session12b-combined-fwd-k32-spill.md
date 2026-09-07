# Session 12b — Combined fwd K32 + reduced spill: FALSIFIED (+32%)

Single-hypothesis test of the Session-12 miss: does spill reduction create the
smem headroom that K32 needs?

## Setup

- Base (worktree @ 7d07369): fwd K16, 16 KB spill → 20 KB smem, 88 regs → 3 blocks/SM
- Exp (combined): K32 loop (E1) + 4 KB spill (E4), no other changes
- Paired Modal T4 A/B, 3 trials, isolated kernels (`tests/bench_fwd_k.py`)

## Resource report (nvcc sm_75 ptxas)

| | BASE (K16) | EXP (K32+spill) |
|---|---|---|
| W_smem | 2 KB [16][64] | 4 KB [32][64] |
| X_smem | 2 KB [64][16] | 4 KB [64][32] |
| spill | 16 KB | 4 KB |
| total smem | 20 KB | **12 KB** |
| regs | 88 | 96 |
| spills | 0 | 0 |
| blocks/SM (smem) | 3 | 5 |
| blocks/SM (regs) | 5 | 5 |
| **blocks/SM (min)** | **3** | **5** |
| warps/SM | 12 | **20** |

Hypothesis resource target HIT exactly: 12 KB / 96 regs / 5 blocks — the same
profile as the winning dX K32 (12 KB / 96 regs / 5 blocks).

## Results (paired A/B, Modal T4)

| shape | BASE | EXP | Δ |
|---|---|---|---|
| fc1 (1024→4096) | 52.5 / 52.2 | 68.2 / 68.1 | **+30%** |
| fc2 (4096→1024) | 52.3 / 52.7 | 68.5 / ~68 | **+30%** |
| head (1024→50272) | 642 / 646 | 846 / ~846 | **+31.8%** |

Consistent across trials. Correctness parity: same 9/10 pass + pre-existing
`_pack_and_check` harness bug on both sides (BASE and EXP identical).

## Verdict

Hypothesis **FALSIFIED**. Resource profile identical to the dX K32 winner, yet
forward regresses +30%. Occupancy (3→5 blocks) is NOT sufficient. The failure
is memory-access behavior, specifically the fwd W smem store:

`W_smem[c][r]` is transposed. Innermost loop index c (=k) becomes the row,
so consecutive threads store 16 apart (K16) / 32 apart (K32) — smem store
scatter doubles with K-widening. The store-transpose cost scales with kK2 and
swamps the sync savings. dX K32 doesn't write transposed (W stays row-major),
so its K-widening wins.

Combined vs E1-alone (880 ms head): spill only relieved 4 % of the +37 %
E1 overhead → ~32 % net. Spill reduction is not the missing ingredient.

## Implication

fwd K32 would need the W decode to write a NON-transposed smem tile
(`W_smem[r][c]`, row-major k fastest) + b_frag ldm=matching, i.e. a layout
change, not a resource change. Out of scope for this single test; not pursued
here.

## Files

- exp worktree at `/tmp/uam_fwdbase` (BASE), main tree reverted clean
- harness `modal_bench_fwd_ab.py` (pointed at fwd bench)
- log `/tmp/fwd_comb_ab.log`