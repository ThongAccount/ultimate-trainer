# Session 6 (2026-08-29): Head-layer dX onto 64×64 — +9.8% e2e tok/s

## Finding

The 64×64 backward-dX kernel (`gemm_backward_dx_tc.cu`) tiles **B (batch)
× K (in_features)** at 64 and reduces over **N (out_features) in 16-steps**
(`kWMMA_K = 16`, with `tile_r = min(16, N-r0)` tail zero-pad). It therefore
requires **N ≡ 0 (mod 16)**, not N ≡ 0 (mod 64).

But the dispatch gate in `custom_ops.backward_dx_tc` was:

```python
if B % 64 == 0 and N_out % 64 == 0 and K % 64 == 0:  # N_out = dY.size(1) = out_features
```

The head layer has `N_out = VOCAB = 50272`; `50272 % 64 = 32`, so the gate
permanently routed head dX to the 32×32 kernel — despite the 64×64 kernel
being able to handle N=50272 (`50272 % 16 == 0`, exactly 3142 reduction steps,
zero tail waste).

## Fix

One line: `N_out % 64 == 0` → `N_out % 16 == 0`.

```python
if _dx_tc_64 is not None and B % 64 == 0 and N_out % 16 == 0 and K % 64 == 0:
```

## Verification (Modal T4)

### Correctness — bit-exact

`tests/validate_head_64_dispatch.py`:
- Head dispatch now routes to 64×64 (`max|out - d64| = 0.00e+00`).
- 64×64 == 32×32 **bit-exact** on head (16384, 50272, 1024), fc1
  (16384, 4096, 1024), and N=50000 tail-stress (128, 50000, 256).
- err64 vs torch fp32 ref: 0.13 on head (refmax 489) — within tolerance.

`tests/probe_head_64.py` (isolated correctness on non-64-multiple N, incl.
N=50000 non-16-multiple tail):
- All shapes PASS vs torch ref.

### Isolated kernel benchmark (B=16384, n=20)

| shape | 64×64 | 32×32 | speedup |
|---|---|---|---|
| head (16384, 50272, 1024) | 557.7 ms | 991.2 ms | 1.78× |

### Full-backward A/B

`tests/ab_head_only.py` (isolates ONLY head layer; MLP layers stay on 64×64):
- head→32: 4147 ms/step → head→64: 3809 ms/step, **−338 ms/step (1.089×)**.

### e2e tok/s (interleaved, single process to cancel T4 drift)

`tests/ab_e2e_head.py`:

| | ms/step | tok/s |
|---|---|---|
| head→32 (old) | 4118.0 | 3979 |
| head→64 (new) | 3714.1 | 4411 |

**+9.8% e2e tok/s**, −404 ms/step. Meets the ≥5% e2e criterion.

### Correctness suite

57 passed (`test_packed_linear`, `test_gemm_update`, `test_kernels`,
`test_packed_ternary`), 564 s.

## Files

- `kernels/packed_ternary/custom_ops.py` — dispatch gate fix (1 line).
- Harness: `tests/probe_head_64.py`, `tests/validate_head_64_dispatch.py`,
  `tests/ab_head_only.py`, `tests/ab_e2e_head.py`.
- Runners: `modal_probe_head_64.py`, `modal_validate_head_64.py`,
  `modal_ab_head_only.py`, `modal_ab_e2e_head.py`, `modal_pytest_suite.py`.

## Caveat

Modal T4 run-to-run absolute timings drift substantially (same config measured
2279 ms vs 3809 ms across two runs) — likely thermal/clock. All A/B comparisons
here are **within a single process** to cancel that drift; the e2e number is
interleaved A,B,A,B.