# Results: A2 — direct float store per-frag (2026-11-20) — NEGATIVE, not shipped

Branch `feat/a2-direct-float-store` @ af0104d, prereg
`2026-11-20-a2-direct-float-store-prereg.md`, session ut-a2.
Artefact `results/perfgate.json` (rc=0, all gates pass).

## Verdicts vs the pre-registered predictions

| Item | Prediction | Measured | Verdict |
|------|-----------|----------|---------|
| fwd A2 | −3..−8% fc1/fc2/head kernel time | fc1 −0.5%, fc2 −4.8%, head −1.8% vs master | within prediction range but step regressed |
| dX A2 | −3..−8% fc1/fc2/head kernel time | fc1 +1.8%, fc2 +3.2%, head −5.5% vs master | mixed; not a clear win |
| parity | bit-exact on all shapes | PASS (fc1, fc2, head all bit-exact) | PASS |
| pytest | 103 passed | 103 passed in 23.09s | PASS |
| step | −1..−3% vs 2976.4 ms | **3007.072 ms = +1.03% vs master** | **LOSS** |

## Key finding

The A2 change (direct float store per-frag, eliminating smem spill) is
bit-exact but does NOT improve the step time. The fwd kernels are
marginally faster (fc1 −0.5%, fc2 −4.8%, head −1.8%), but the overall
step time regressed +1.03% (3007.072 vs 2976.4 ms). The dX kernels are
mixed (fc1 +1.8%, fc2 +3.2%, head −5.5%).

The direct register→global writes may not coalesce as well as the
smem-mediated writes. The fragment layout (thread t holds 8 elements
scattered across the 16×16 tile) doesn't match the coalescing pattern of
the smem-mediated path.

**Decision: A2 stays on the list as measured-negative. Not merged.**

## Artefacts

- `results/perfgate.json` — full perfgate output (provenance, kernels,
  step_ms, deltas, rc=0)
- pytest 103 passed in 23.09s
- parity bit-exact on fc1/fc2/head

## Per-kernel numbers (A2 variant)

| Kernel | fc1 | fc2 | head |
|--------|-----|-----|------|
| fwd_ms | 25.721 | 25.212 | 335.785 |
| dx_ms | 36.663 | 37.087 | 490.542 |
| update_ms | 47.338 | 45.635 | 591.962 |

step_ms: 3007.072

## Comparison vs current master (2976.4 ms)

| Metric | Master | A2 | Delta |
|--------|--------|-----|-------|
| step_ms | 2976.4 | 3007.072 | **+1.03%** |

A2 is slightly slower. Not an improvement.