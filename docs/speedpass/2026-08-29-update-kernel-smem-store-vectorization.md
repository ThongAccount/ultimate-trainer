# Update-Kernel Shared-Store Vectorization — 2026-08-29 (Modal T4)

Single change to `kernels/packed_ternary/gemm_update_tc_v2_32.cu` that removes a
2-way shared-memory bank conflict in the TC32 weight-update kernel.

## Hypothesis → proof

The kernel's dY/X tile load loops write `half` values to shared memory as two
separate 16-bit stores:

```c
DYS(warp_id, b, r)     = v.x;
DYS(warp_id, b, r + 1) = v.y;
```

Adjacent halves land in the same 4-byte shared bank, so every warp-level store
instruction replays as a **2-way conflict**. ncu confirmed before the change:

| metric | value |
|---|---|
| shared-store bank conflicts | 1,669,866,713 |
| shared-store wavefronts | 3,318,864,250 (≈2× stores) |
| shared-load conflicts | 1.7M (≈0-way, already clean) |

So `wmma::load_matrix_sync` loads were *never* the bottleneck — a prior padded
`kStride` attempt at load de-aliasing changed nothing ("excessive wavefronts"
stayed byte-identical). The real target was the **stores**.

## Fix

Vectorize each half-pair into a single 32-bit `half2` store. `r`/`c` are always
even (`i = q*2`), so the destination is 4-byte aligned and the pointer cast is
legal:

```c
*reinterpret_cast<half2*>(&DYS(warp_id, b, r)) = v;
*reinterpret_cast<half2*>(&XS(warp_id, b, c))  = v;
```

32 lanes now hit 32 distinct banks (conflict-free). Scalar fallback path for
unaligned/boundary cases is unchanged.

## Results

Correctness (bit-exact vs torch reference and cross-checks): **PASS**
- `update_tc_v2` counter: 0/16384 errors (max diff 0)
- backward_dx, TC-vs-scalar, flip-direction tests: all `max_diff=0.0000`

ncu (T4, kernel only):

| metric | before | after |
|---|---|---|
| shared-store bank conflicts | 1.67B | **46.6M** (36×) |
| shared-store wavefronts | 3.32B | **1.70B** |

Isolated kernel timing (B=16384, 20 iters, bypass autograd):

| shape | in | out | before | after | Δ |
|---|---|---|---|---|---|
| fc1 | 1024 | 4096 | 50.09 ms | 46.24 ms | **-7.7%** |
| fc2 | 4096 | 1024 | 56.36 ms | 46.28 ms | **-17.9%** |
| head | 1024 | 50272 | 594.64 ms | 550.48 ms | **-7.4%** |

## Notes

- The bench harness's "full step" section throws `shape '[16384, 1024]' is
  invalid for input of size 16384` — a pre-existing harness bug (it feeds token
  IDs where an embedding tensor is expected), unrelated to this change.
- Prior `kSmemPad` auto-padding experiment was **reverted**: it targeted
  `load_matrix_sync` (already 0-way) and inflated smem 8→10 KB (occupancy risk)
  for zero benefit.