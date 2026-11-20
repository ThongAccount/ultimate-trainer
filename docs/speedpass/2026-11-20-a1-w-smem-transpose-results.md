# Results: A1 — W smem store transpose (2026-11-20) — ALREADY DONE

Branch `feat/a1-w-smem-transpose` not created — A1 was already applied at
commit `d9ba61f` (2026-09-12): "perf: fwd TC64 W_smem row-major (kill 32-way
store bank conflict) — -38% isolated, correct".

## What was already done

Commit `d9ba61f` flipped W tile from transposed `W_smem[k][n]` (c=k
innermost, stride 64, 32-way store bank conflict) to row-major `W_smem[n][k]`
+ matrix_b col_major ldm=16 (same trick the proven dX kernel uses). W store
now inner-stride 1, 2-way at worst.

Measured on Modal T4: fc1 52.7→32.2 ms, fc2 53.2→32.7 ms, head 653→401 ms
(−38.5%), parity bit-identical to old.

## Current state on master (8717664)

`gemm_forward_tc.cu:63-66`:
```cpp
// W_smem is [n][k] row-major (64×16): W[n0:64, k0:k0+16] stored untransposed.
// b_frag loads from it as col_major (leading dim kWMMA_K=16) — the same
// trick dX uses. Consecutive lanes write consecutive k (stride 1), killing
// the 32-way store bank conflict of the old transposed W_smem[c][r].
__shared__ half W_smem[kSuperN][kWMMA_K];  // [64][16]
```

A1 is already shipped. No further work needed.

## Decision

A1 is DONE (already in master @ d9ba61f). Remove from the candidate list.