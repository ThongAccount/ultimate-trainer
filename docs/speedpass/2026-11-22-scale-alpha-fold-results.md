# Scale-alpha fold: measured-negative (parity FAIL)

Branch: `feat/scale-alpha-fold` @ `3443179`
Prereg: `docs/speedpass/2026-11-22-scale-alpha-fold-prereg.md`
Date: 2026-11-24

## Verdict: PARITY FAIL — not bit-exact

The scale-alpha fold mechanism fails the pre-registered bit-exact parity gate
at the first shape tested (`16352x1024x4096`, fc1).

## Root cause

The two approaches are mathematically equivalent but **not bit-exact**:

1. **Scale-alpha fold (variant)**: `W16[i] = decode(W[i]) * scale`, then
   `Y = X @ W16^T`. The scale is applied to each weight element before
   accumulation.
2. **Separate scale (reference)**: `Y = (X @ decode(W)^T) * scale`. The scale
   is applied to the final output after accumulation.

For `scale = K^-0.5` (a non-power-of-2 for most K), these produce different
rounding behaviour in fp16. The accumulation of pre-scaled values vs.
post-scaling the accumulated result diverges in the low bits.

This is a fundamental numerical property, not a bug in the implementation.
The scale-alpha fold **cannot** be made bit-exact with the separate scale
application for arbitrary scale values.

## What passed

- Smoke: 13/13 tests passed
- pytest: 103/103 passed (all existing correctness gates)

## What failed

- `scale_alpha_parity`: FAIL at `16352x1024x4096` (first shape)
  - `torch.equal(y, y_ref)` returned False
  - The outputs are numerically close but not bit-identical

## Decision

Per prereg: "FAIL if not bit-exact" → this candidate is **closed**.
The scale-alpha fold mechanism is not viable for the shipped arm because
it changes the numerics. The separate `* scale` passes must remain.

## Artefacts

- `results/scale_alpha.0f4d8de6.log` — parity failure log from Colab T4
- Branch `feat/scale-alpha-fold` retains the implementation for reference
  but is **not merged**.

## Next item

Item 2: flip-count batching/deferral (~11 ms target).
