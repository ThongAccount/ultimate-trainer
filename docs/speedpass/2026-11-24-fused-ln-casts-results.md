# Fused LayerNorm casts: NOISE (within measurement noise)

Branch: `feat/fused-ln-casts` @ `79aaeee`
Prereg: `docs/speedpass/2026-11-24-fused-ln-casts-prereg.md`
Date: 2026-11-24

## Verdict: NOISE — improvement within measurement noise, not a clear win

The fused LayerNorm + cast via `torch.compile` shows a small improvement
that is below the 1% decision threshold. The `torch.compile` overhead
(compilation time + guard evaluation) partially offsets the cast fusion
savings.

## Measurements

Same-VM comparison on Colab T4, flip_budget=0, 20 steps each:

| Branch | Step ms (median) | Step ms (mean) | Std |
|--------|-----------------|----------------|-----|
| master @ `164ea6f` | **467.636** | 467.464 | 3.257 |
| fused-ln-casts @ `79aaeee` | **465.697** | 465.439 | 3.680 |

**Delta: −1.939 ms (−0.41%)** — below the 1% threshold.

## What passed

- pytest: 103/103 passed on the correct branch
- Loss trajectory: bit-identical (11.0078125 on both branches)

## Decision

Per prereg decision rule: |Δ| = 0.41% < max(2σ_null, 1%) → NOISE.
Not a clear win. The candidate is **closed** — the `torch.compile` overhead
offsets the cast fusion savings. Not merged.

## Artefacts

- Branch `feat/fused-ln-casts` retains the implementation for reference
  but is **not merged**.
