# Results: I1 — fused CE+scale (2026-11-20) — WIN, merged

Branch `feat/i1-fused-ce-scale` @ 5f890e4, prereg
`2026-11-20-i1-fused-ce-scale-prereg.md`, session ut-i1.
Artefact: parity + A/B output (inline below).

## Verdicts vs the pre-registered predictions

| Item | Prediction | Measured | Verdict |
|------|-----------|----------|---------|
| fused CE+scale | ~14 ms saved on shipped step | **−11.714 ms** (13.048 vs 24.761 ms) | **WIN** |
| parity | bit-exact loss trajectory vs eager CE | diff = 0.000000e+00 | PASS |
| pytest | 103 passed | 103 passed in 272.77s | PASS |

## Key finding

The fused CE+scale via torch.compile saves **11.714 ms** on the CE
computation (13.048 ms vs 24.761 ms, −47.3%). The loss trajectory is
bit-exact (diff = 0.000000e+00), so the bit-stable gate passes.

This is close to the predicted ~14 ms scale pass saving. The fused kernel
eliminates the separate scale multiply, fusing it with the CE computation.

**Decision: I1 is an IMPROVEMENT. Merge to master.**

## Per-kernel numbers

| Metric | Eager CE | Fused CE+scale | Delta |
|--------|----------|----------------|-------|
| CE time | 24.761 ms | 13.048 ms | **−11.714 ms** |

## Loss trajectory

```
eager: 11.320312
fused: 11.320312
diff: 0.000000e+00
```

Bit-exact. Gate passes.

## Recommendation

Merge `feat/i1-fused-ce-scale` to master. The fused CE+scale via
torch.compile is a clean win: −11.714 ms on the CE computation with
bit-exact loss trajectory. Remove from the candidate list.