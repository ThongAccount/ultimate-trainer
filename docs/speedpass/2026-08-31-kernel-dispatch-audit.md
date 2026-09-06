# Kernel Dispatch Audit — 2026-08-31

Full static pass over every packed-ternary kernel, its loader, its dispatch site,
and the predicate gating it. This supersedes any prior dispatch claims; it is
derived from the code at `d23f16c`, not from docs.

## 1. Model shapes (gigatoken, B=32×SEQ, SEQ=512)

| layer | in_features (K) | out_features (N) | batch (B) |
|---|---|---|---|
| fc1 (×6) | 1024 | 4096 | 16384 |
| fc2 (×6) | 4096 | 1024 | 16384 |
| head (×1) | 1024 | 50272 | 16384 |

All B, K are `%64==0`. N={4096,1024,50272}: 4096/1024 are `%64==0`;
50272 `%16==0` but `%64==32`.

## 2. Kernel inventory → loader → dispatch

### Forward

| kernel file | fn (module) | tile | reached by |
|---|---|---|---|
| `gemm_forward_tc.cu` | `_forward_fn_tc_64` / `packed_ternary_forward_tc_64` | 64×64 | `_forward_auto` |
| `gemm_forward_tc_32.cu` | `_forward_fn_tc` / `co_forward_tc` | 32×32 | `_forward_auto` |
| `gemm_forward_v2.cu` | `packed_ternary_forward_v2` | scalar | `_forward_auto` |
| `gemm_forward.cu` | `packed_ternary_forward` | scalar v1 | `_forward_auto` |
| v3/v4/packed | test-only | — | direct calls only |

`_forward_auto` (packed_linear.py:55):
1. `B,N,K >= 64 and _tc_ok(B,N,K)` → **64×64** (`has_tc_64`)
2. `_tc_ok(B,N,K)` and `has_tc()` → **32×32** (`co_forward_tc`)
3. `N>=4 and v2` → v2
4. v1

Model routing: all layers qualify for 64×64 (50272%16=0). ✓ production = 64×64.

### Backward dX

| kernel | fn | tile | gate |
|---|---|---|---|
| `gemm_backward_dx_tc.cu` | `_dx_tc_fn` → `_dx_tc_64` | 64×64 | `B%64==0 and N_out%16==0 and K%64==0` |
| `gemm_backward_dx_tc_32.cu` | `_dx_tc_32_fn` → `_dx_tc` | 32×32 | fallback |
| `gemm_backward_dx.cu` | `_dx_fn` | scalar | dims<16 |

`co_backward_dx_tc` (custom_ops.py:73):
`B%64==0 and N_out%16==0 and K%64==0` → 64×64, else 32×32.

d23f16c relaxed N_out gate 64→16; head (50272) now routes 64×64. ✓

### Update (weight dW → counter → flip)

| kernel | fn | tile | reached in production? |
|---|---|---|---|
| `gemm_update_tc_v2_32.cu` | `_up_tc_v2_32_fn` | 32×32 | **YES — always** |
| `gemm_update_tc_v3_32.cu` | `_up_tc_v3_32_fn` | 32×32 | only via `backward_update` fallback |
| `gemm_update_tc_v2.cu` | `_up_tc_v2_fn` (real) | **64×64** | **NO — dead + buggy** |
| `gemm_update_tc_v3.cu` | `_up_tc_v3_fn` (real) | 64×64 | **NO — dead** |
| `gemm_update_tc.cu` | `_up_tc_fn` | 32×32 v1 | NO (superseded) |
| `gemm_update.cu` | `_up_fn` | scalar | dims<16 |

### Fused

`gemm_fused_backward_update.cu` → `co_backward_update_fused` — only when
`B<=64 and N_out<=128 and in_features<=128` (packed_linear.py:155). Model B=16384 ⇒
**never used**.

## 3. The production update dispatch is pinned to 32×32

`custom_ops._ensure_loaded()` (custom_ops.py:40) calls `_load_tc_if_needed()`,
which does:

```python
_load_up_tc_v2_32()               # loads gemm_update_tc_v2_32.cu (32×32)
_up_tc_v2_fn = _up_tc_v2_32_fn    # aliases 32×32 into the "v2" slot
_HAS_UP_TC_V2 = True
```

`_ensure_loaded` then sets `_update_tc_v2 = pu._up_tc_v2_fn` = 32×32. `co_update_tc_v2`
(custom_ops.py:86) has **no gate** — it always dispatches to this 32×32 fn.

The trainer autograd calls `co_update_tc_v2` directly (packed_linear.py:163), so the
production update kernel is **32×32 for every layer, including head (out=50272)**.

The real 64×64 update kernel (`gemm_update_tc_v2.cu`) is reachable only via
`_load_up_tc_v2()` from `pack_update.update()`'s 64×64 branch — which is never hit
by training, and the kernel itself has a documented WMMA addressing bug (§14.2 in
HANDOFF).

## 4. Predicate truth table

| op | GEMM | tile dims | reduction dim | true gate (for %16/%64 WMMA) |
|---|---|---|---|---|
| forward 64×64 | X@Wᵀ | M=B,N=N_out | K=in (%16-step) | B,N≥64 & %16; K≥64 & %16 tail-safe |
| backward dX 64×64 | dY@W | M=B,N=K(in) | N_out (%16-step) | B,K tile %64 (tail-safe), N_out %16 |
| update 32×32 | dYᵀ@X | M=N_out,N=N_in (32) | B (%16-step) | N_out,N_in %32; B any (tail-safe) |

Note for update the reduction dim is **batch (B=16384)**; the 32×32 kernel loops
1024× (kK=16). The update kernel's `pack_update.update()` 64×64 gate is
`_tc_ok_64(B) and _tc_ok_64(N_out) and _tc_ok_64(N_in)` — the `B` requirement is
spurious (B is reduction, only needs tail handling), but irrelevant because the
kernel is dead.

## 5. Conclusions

1. Forward and backward dX are now correctly on 64×64 for all model layers.
2. **Update is the #1 remaining cost and is pinned to 32×32 for all layers.**
3. The 64×64 update kernel exists but is (a) unreachable from production and
   (b) semantically buggy (WMMA col_major addressing, HANDOFF §14.2).
4. The highest-value fix is getting update onto 64×64 tiles — either by fixing the
   existing 64×64 kernel + wiring dispatch, or increasing the 32×32 kernel's
   batch step kK 16→32 (§14.4 recommendation).