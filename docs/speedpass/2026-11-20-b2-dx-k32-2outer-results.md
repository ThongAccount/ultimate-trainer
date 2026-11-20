# Results: B2 — dX K32 + 2 outer loops (2026-11-20) — NEGATIVE, not shipped

Branch `feat/b2-dx-k32-2outer` @ b27ce85, prereg
`2026-11-20-b2-dx-k32-2outer-prereg.md`, session ut-b2.
Artefact `results/perfgate.json` (rc=1, valid negative).

## Verdicts vs the pre-registered predictions

| Item | Prediction | Measured | Verdict |
|------|-----------|----------|---------|
| dX B2 | −3..−8% head dX kernel time | head −6.76% (605.384 vs 649.31 ms) | within prediction range |
| parity | bit-exact on all shapes | PASS (fc1, fc2, head all bit-exact) | PASS |
| pytest | 103 passed | 103 passed in 275.59s | PASS |
| step | −1..−3% vs 2976.4 ms | **3517.857 ms = +18.2% vs master** | **LOSS** |

## Key finding

The B2 change (dX K32 + 2 outer loops, 4 MMAs/sync at 12 KB smem) is
bit-exact and the dX head kernel is −6.76% faster. But the overall step
time regressed +18.2% (3517.857 vs 2976.4 ms). The 2-outer-loop change
increased register pressure and reduced occupancy, which hurt the overall
step time.

The head/update kernel is +5.54% slower (739.2 vs 700.43 ms), likely
because the B2 change's register pressure spills into the update kernel's
execution.

**Decision: B2 stays on the list as measured-negative. Not merged.**

## Per-kernel numbers (B2 variant)

| Kernel | fc1 | fc2 | head |
|--------|-----|-----|------|
| fwd_ms | 26.829 | 27.901 | 356.297 |
| dx_ms | 38.456 | 38.448 | 605.384 |
| update_ms | 53.788 | 52.108 | 739.200 |

step_ms: 3517.857

## Comparison vs current master (2976.4 ms)

| Metric | Master | B2 | Delta |
|--------|--------|-----|-------|
| step_ms | 2976.4 | 3517.857 | **+18.2%** |
| head/dx | 649.31 | 605.384 | −6.76% |
| head/update | 700.43 | 739.200 | **+5.54%** |

B2 is slower overall. Not an improvement.