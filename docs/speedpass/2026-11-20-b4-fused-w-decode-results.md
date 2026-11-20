# Results: B4 — fwd fuse W decode into dY load (2026-11-20) — NEGATIVE, not shipped

Branch `feat/b4-fused-w-decode` @ 7c70e23, prereg
`2026-11-20-b4-fused-w-decode-prereg.md`, session ut-b4.
Artefact `results/perfgate.json` (rc=0, all gates pass).

## Verdicts vs the pre-registered predictions

| Item | Prediction | Measured | Verdict |
|------|-----------|----------|---------|
| fwd B4 | −3..−6% fc1/fc2/head kernel time | fc1 −6.4%, fc2 −5.3%, head −8.3% vs master | within prediction range |
| parity | bit-exact on all shapes | PASS (fc1, fc2, head all bit-exact) | PASS |
| pytest | 103 passed | 103 passed in 273.20s | PASS |
| step | −1..−2% vs 2976.4 ms | **3142.713 ms = +5.6% vs master** | **LOSS** |

## Key finding

The B4 change (fwd fuse W decode into dY load, vectorized decode + half2
store) is bit-exact and the fwd kernels are faster (fc1 −6.4%, fc2 −5.3%,
head −8.3%). But the overall step time regressed +5.6% (3142.713 vs
2976.4 ms). The vectorized decode increased register pressure and reduced
occupancy, which hurt the overall step time.

**Decision: B4 stays on the list as measured-negative. Not merged.**

## Per-kernel numbers (B4 variant)

| Kernel | fc1 | fc2 | head |
|--------|-----|-----|------|
| fwd_ms | 24.199 | 25.090 | 313.694 |
| dx_ms | 38.719 | 38.565 | 594.279 |
| update_ms | 53.367 | 53.074 | 704.249 |

step_ms: 3142.713

## Comparison vs current master (2976.4 ms)

| Metric | Master | B4 | Delta |
|--------|--------|-----|-------|
| step_ms | 2976.4 | 3142.713 | **+5.6%** |
| fc1/fwd | 25.849 | 24.199 | −6.4% |
| fc2/fwd | 26.494 | 25.090 | −5.3% |
| head/fwd | 341.989 | 313.694 | −8.3% |

B4 is slower overall despite faster fwd kernels. Not an improvement.