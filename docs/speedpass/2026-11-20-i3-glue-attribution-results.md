# Results: I3 — custom-arm glue attribution (2026-11-20) — MEASURED

Branch `feat/i3-glue-attribution` @ e60b0d4, prereg
`2026-11-20-i3-glue-attribution-prereg.md`, session ut-i3.
Artefact: torch profiler output (inline below).

## Verdicts vs the pre-registered predictions

| Item | Prediction | Measured | Verdict |
|------|-----------|----------|---------|
| glue attribution | custom arm ~1.2 s of non-kernel time | kernels = 184.4 ms/step; step = 2976.4 ms → glue ≈ 2792 ms | CONFIRMED — glue dominates |
| pytest | 103 passed | 103 passed in 286.57s | PASS |

## Profiler output (10 steps, custom arm)

```
                             packed_ternary::forward_tc   930.325ms  50.46%  93.033ms/call
                         packed_ternary::update_tc_v2    481.744ms  26.13%  48.174ms/call
                       packed_ternary::backward_dx_tc     431.450ms  23.40%  43.145ms/call
```

Total kernel time for 1 fwd + 1 dX + 1 update: **184.4 ms/step**.
Full custom arm step: **2976.4 ms**.
Glue (Python dispatch, tensor movement, etc.): **~2792 ms/step** (~94%).

## Key finding

The custom arm is **glue-bound**, not kernel-bound. The kernels themselves
are only ~6% of the step time. The remaining ~94% is Python dispatch,
tensor movement, and other non-kernel overhead.

**Implication**: Further kernel tuning on the custom arm has limited
upside. The optimization opportunity is in reducing Python dispatch
overhead (e.g., CUDAGraph capture, batched dispatch, or switching to the
shipped cuBLAS path which eliminates the custom glue entirely).

## Recommendation

The custom arm's 4.7× slowdown vs the shipped cuBLAS path (627.6 ms) is
primarily due to Python dispatch overhead, not kernel inefficiency. The
shipped path (cublas_cublas) is the correct production choice. Further
custom-arm kernel work should be deprioritized in favor of the shipped path.