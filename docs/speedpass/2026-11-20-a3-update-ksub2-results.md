# Results: A3 — update kSub=2 (2026-11-20) — NEGATIVE, not shipped

Branch `feat/a3-update-ksub2` @ 1024ed7, prereg
`2026-11-20-a3-update-ksub2-prereg.md`, session ut-a3.
Artefact `results/perfgate.json` (rc=1, valid negative).

## Verdicts vs the pre-registered predictions

| Item | Prediction | Measured | Verdict |
|------|-----------|----------|---------|
| update A3 | −1..−3% fc1/fc2/head kernel time | fc1 +0.56%, fc2 −0.67%, head +6.57% | mixed; not a clear win |
| parity | bit-exact on all shapes | PASS (fc1, fc2, head all bit-exact) | PASS |
| pytest | 103 passed | 103 passed in 271.93s | PASS |
| step | −0.5..−1% vs 2976.4 ms | **3672.027 ms = +23.4% vs master** | **LOSS** |

## Key finding

The A3 change (update kSub=2, 2 blocks/loop, smem 8→12 KB) is bit-exact
but the update kernels are mixed (fc1 +0.56%, fc2 −0.67%, head +6.57%).
The overall step time regressed +23.4% (3672.027 vs 2976.4 ms). The
2-block change increased register pressure and reduced occupancy, which
hurt the overall step time.

**Decision: A3 stays on the list as measured-negative. Not merged.**

## Per-kernel numbers (A3 variant)

| Kernel | fc1 | fc2 | head |
|--------|-----|-----|------|
| fwd_ms | 27.581 | 28.418 | 368.449 |
| dx_ms | 39.313 | 38.986 | 603.643 |
| update_ms | 56.035 | 54.510 | 746.440 |

step_ms: 3672.027

## Comparison vs current master (2976.4 ms)

| Metric | Master | A3 | Delta |
|--------|--------|-----|-------|
| step_ms | 2976.4 | 3672.027 | **+23.4%** |
| fc1/update | 55.72 | 56.035 | +0.56% |
| fc2/update | 54.88 | 54.510 | −0.67% |
| head/update | 700.43 | 746.440 | **+6.57%** |

A3 is slower overall. Not an improvement.