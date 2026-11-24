# Skip unused gradients: WIN — merged

Branch: `feat/skip-unused-grads` @ `718ea0c`
Prereg: `docs/speedpass/2026-11-24-skip-unused-grads-prereg.md`
Date: 2026-11-24

## Verdict: WIN — −3.07% step time, merged to master

Skipping unused embedding and LayerNorm gradients saves **14.352 ms/step**
(−3.07%). The improvement exceeds the 1% decision threshold. Loss trajectory
is bit-identical.

## Measurements

Same-VM comparison on Colab T4, flip_budget=0, 20 steps each:

| Branch | Step ms (median) | Step ms (mean) | Std |
|--------|-----------------|----------------|-----|
| master @ `164ea6f` | **467.636** | 467.464 | 3.257 |
| skip-unused-grads @ `718ea0c` | **453.284** | 454.047 | 3.292 |

**Delta: −14.352 ms (−3.07%)** — exceeds the 1% threshold.

## What passed

- pytest: 103/103 passed on the correct branch
- Loss trajectory: bit-identical (11.0078125 on both branches)

## Mechanism

1. **Embedding**: `requires_grad = False` on the embedding weight. The
   embedding is a lookup table whose gradient is sparse and unused (no
   optimizer.step()).
2. **LayerNorm**: `requires_grad = False` on all norm weights and biases.
   LayerNorm gradients are dense and unused.
3. **Embedding output detach**: `.detach()` on the embedding output before
   the first layer. This prevents the layer-0 dX from propagating back to
   the embedding.

## Decision

Per prereg decision rule: |Δ| = 3.07% > max(2σ_null, 1%) → WIN.
Merged to master.

## Artefacts

- `results/skip_grads.*` on the VM
- Merged to master.
