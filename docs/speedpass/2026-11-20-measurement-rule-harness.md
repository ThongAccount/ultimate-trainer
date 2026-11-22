# Standing measurement rule — harness encoding (2026-11-20)

Ruling context: the post-sweep debate (A2/B2/B4/A3 closed; I3's 94%-glue and
I2's clock-throttle claims both rejected as measurement artefacts) adopted a
standing rule for every future A/B verdict. This doc records where the rule
is enforced **in tooling**, not by convention.

## The rule

1. **Interleaved A/B, same session** — paired A/B/A/B so clock/allocator
   drift is common-mode and cancels.
2. **Median-over-steps** — never min-of-N.
3. **Spread reported** — every number carries its arm's dispersion
   (half-IQR as % of median). A |delta| inside the spread is noise.
4. **Clock telemetry attached** — nvidia-smi clocks/power/temp sampled
   during the measurement; a droop signature is computed in the artefact.
   Telemetry that cannot be collected is recorded `available=false` **with
   its cause**, never silently dropped.

## Where it is encoded

- `tests/bench_protocol.py`
  - `spread(times)` → p10/p90/IQR/half-IQR-% per arm.
  - `ABResult.a_spread` / `.b_spread` — attached to every result.
  - `report_line(...)` now appends `spreadA=±x% spreadB=±y%`; the
    WIN/LOSS/NOISE decision (sigma_null / MIN_DETECTABLE_PCT) is unchanged,
    so sub-threshold deltas still print **NOISE**, not a verdict.
  - `ClockSampler(csv_path)` — `start()`/`stop()` (or context manager)
    samples nvidia-smi (same query as `scripts/colab_measure_p0.py`) and
    returns a summary: samples, min/median/max SM clock, median power, max
    temp, `sustained_droop` flag (min < 90% of median).
  - `ab_median`/`solo_median` already implement (1) and (2) since C13.
- `scripts/colab_step_ab.py`
  - Driver `RESULTS:` line gains `half_iqr_ms` (spread of the step median).
  - Each arm subprocess is wrapped in a `ClockSampler`; a `CLOCKS:` line is
    printed per arm and the artefact gains a `clocks` block per arm plus a
    `measurement_rule` pointer to this doc.
- `tests/test_measurement_rule.py` — 10 unit tests, no GPU needed:
  spread math, NOISE/WIN behaviour, null-sigma threshold, telemetry parsing
  against a fake nvidia-smi (including the droop case), the unavailable-
  sampler failure path, and a compile check of the VM driver string.

## Usage for the next candidate

```python
from bench_protocol import ab_median, report_line, ClockSampler

clk = ClockSampler("results/clocks.mycandidate.csv")
clk.start()
res = ab_median(run_a, run_b, prepare_a, prepare_b, null_arm=True)
print(report_line("head", res))
print(clk.stop())   # attach this dict to the artefact
```

## What this does NOT cover

- **Whole-step arms are still sequential subprocesses** (the backend is read
  at import time). Cross-arm whole-step deltas below session drift are not
  acceptance-grade; for sub-% effects use the in-process interleaved
  protocol above, not `colab_step_ab.py` arm ratios.
- Throttle-reason bitfields are not queried (the field name differs across
  driver versions); the recorded signal is clock droop + temperature.
- The modified `colab_step_ab.py` has not run on a GPU from this branch
  (CPU-only dev box): the driver is compile-checked and the new code paths
  are unit-tested. Run one Colab smoke before the next sweep.
