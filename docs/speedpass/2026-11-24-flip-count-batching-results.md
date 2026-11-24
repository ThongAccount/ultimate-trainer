# Flip-count batching: measured-negative

Branch: `feat/flip-count-batching` @ `43571ef`
Prereg: `docs/speedpass/2026-11-24-flip-count-batching-prereg.md`
Date: 2026-11-24

## Verdict: MEASURED-NEGATIVE — slower by +2.46%

The deferred flip_count read optimization is **slower** than the current
per-step flip_count read. The branch divergence from the deferred check
costs more than the memory traffic savings.

## Measurements

Same-VM comparison on Colab T4, flip_budget=5, 20 steps each:

| Branch | Step ms (median) | Step ms (mean) | Std |
|--------|-----------------|----------------|-----|
| master @ `e34d482` | **482.576** | 481.213 | 5.489 |
| flip-count variant @ `43571ef` | **494.423** | 494.744 | 6.168 |

**Delta: +11.847 ms (+2.46%)** — the variant is slower.

## What passed

- Smoke: 13/13 tests passed
- pytest: 103/103 passed (all existing correctness gates)
- Loss trajectory: bit-identical (11.0078125 on both branches)

## Root cause

The deferred flip_count read adds a branch (`if cnt > threshold || cnt < -threshold`)
that most threads evaluate as false (most weights don't flip on any given step).
The branch divergence and the restructured control flow cost more than the
flip_count memory read savings. On T4 (sm_75), the update epilogue kernel is
memory-bandwidth-bound, and the extra branch instructions add overhead without
reducing the dominant memory traffic (dW read, counter read/write, W atomic).

## Decision

Per prereg decision rule: |Δ| = 2.46% > max(2σ_null, 1%) → measured-negative.
Candidate is **closed**. Not merged.

## Artefacts

- `results/perfgate.json` — perfgate results from Colab T4
- Branch `feat/flip-count-batching` retains the implementation for reference
  but is **not merged**.

## Next item

Item 3: fused LayerNorm casts (~10 ms target) and skip-unused-grads (~10 ms target).
