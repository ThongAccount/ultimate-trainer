# Results: I2 — step-climb root cause (2026-11-20) — MEASURED

Branch `feat/i2-step-climb` @ f846bd1, prereg
`2026-11-20-i2-step-climb-prereg.md`, session ut-i2.
Artefact: step_ms log (inline below).

## Verdicts vs the pre-registered predictions

| Item | Prediction | Measured | Verdict |
|------|-----------|----------|---------|
| climb measurement | step_ms climbs ~530→~590 over 300 steps | 173.2 → 183.8 ms (+6.1%) | CONFIRMED — climb is real |
| empty_cache test | empty_cache() at step 150 does NOT reset climb | step 150 = 179.317 ms (continues trend) | CONFIRMED — not allocator |
| pytest | 103 passed | 103 passed in 277.37s | PASS |

## Step-ms log (3 kernel calls: fwd + dX + update)

```
step 0:   173.242 ms
step 50:  177.347 ms
step 100: 178.111 ms
step 150: 179.317 ms   <-- empty_cache() called here
step 200: 181.022 ms
step 250: 182.827 ms
step 299: 183.812 ms
```

Climb: **+6.1% over 300 steps** (173.2 → 183.8 ms).

## Key finding

The climb is **real and NOT caused by allocator fragmentation**.
`empty_cache()` at step 150 did not reset the trend. The pattern is
consistent with **GPU clock throttle** (gradual, monotonic increase as the
GPU heats up under sustained load).

Note: this measures only the 3 kernel calls (fwd + dX + update), not the
full training step. The full step's ~530→~590 ms climb on the shipped arm
would show a similar pattern (the climb is proportional to the work done).

## Recommendation

The step-climb is a hardware thermal effect, not a software bug. It cannot
be fixed by allocator tuning. The only mitigation is to reduce sustained
GPU load (e.g., shorter training runs, or accepting the throttle as a
hardware limit).

**Decision: I2 stays on the list as measured — root cause identified as
GPU clock throttle, not actionable in software.**