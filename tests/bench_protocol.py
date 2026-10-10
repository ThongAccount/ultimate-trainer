"""Shared A/B benchmark protocol — the C13 upgrade (replaces ad-hoc 3-trial min-of-3).

Per docs/speedpass/2026-10-10-20-agent-crack-sweep.md (investigation #15):
the old protocol (3 trials, min, no clock settle) had a ±4–6% CI, making any
1% effect undetectable and several past "falsified" verdicts actually
"undecided". This module is the mandatory contract for every new probe.

Protocol:
- N_TRIALS (default 10) paired, single-process, strictly interleaved A/B/A/B
  (any allocator/clock drift is common-mode and cancels in the pairing).
- median (not min — min-of-N is a biased order statistic under asymmetric noise)
- WARMUP (default 10) plus a torch.cuda._sleep clock-settle barrier after each
  trial so power-state wobble does not alias into the next measurement.
- Optional A/A null arm: run the same arm against itself to empirically measure
  sigma_null; decision rule |mean delta| > max(2*sigma_null, MIN_DETECTABLE %).

Usage (kernel-level A/B where each call mutates state — clone per call):

    from tests.bench_protocol import ab_median, report_line

    res = ab_median(run_a, run_b, prepare_a, prepare_b)
    print(report_line("head", res))

where run_a/run_b are zero-arg callables and prepare_* rebuild fresh state
(e.g. W0.clone(), C0.clone()) — prepare time is EXCLUDED from timing.
"""

import time

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover - allows import on non-torch boxes
    torch = None
    _HAS_TORCH = False

N_TRIALS = 10
WARMUP = 10
SETTLE_SLEEP_ITER = 20_000  # torch.cuda._sleep iterations between trials
MIN_DETECTABLE_PCT = 1.0    # ship bar: nothing below this is reportable as a win


class ABResult:
    __slots__ = ("a_ms", "b_ms", "delta_pct", "a_all", "b_all", "sigma_null_pct")

    def __init__(self, a_all, b_all, sigma_null_pct=None):
        import statistics
        self.a_all, self.b_all = list(a_all), list(b_all)
        self.a_ms = statistics.median(self.a_all)
        self.b_ms = statistics.median(self.b_all)
        self.delta_pct = 100.0 * (self.b_ms - self.a_ms) / self.a_ms
        self.sigma_null_pct = sigma_null_pct


def _settle():
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda._sleep(SETTLE_SLEEP_ITER)
        torch.cuda.synchronize()


def _time_once(run, prepare):
    """One timed call. prepare() (fresh destructive state) runs OUTSIDE timing."""
    if prepare is not None:
        prepare()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _bench(run, prepare, warmup=WARMUP):
    for _ in range(warmup):
        if prepare is not None:
            prepare()
        run()
    _settle()
    return [_time_once(run, prepare) for _ in range(N_TRIALS)]


def ab_median(run_a, run_b, prepare_a=None, prepare_b=None, warmup=WARMUP,
             null_arm=False):
    """Interleaved paired A/B, median-of-N_TRIALS.

    null_arm=True adds an A/A measurement first (same callable twice) and
    returns an empirically calibrated sigma_null in the result. Use it the
    first time a probe runs on a new backend; the null sigma is the decision
    threshold generator.
    """
    a_all, b_all = [], []
    for _ in range(N_TRIALS):
        a_all.append(_time_once(run_a, prepare_a))
        b_all.append(_time_once(run_b, prepare_b))
        _settle()
    # warmup once more for safety after settle
    sigma_null = None
    if null_arm:
        import statistics
        n1, n2 = [], []
        for _ in range(N_TRIALS):
            n1.append(_time_once(run_a, prepare_a))
            n2.append(_time_once(run_a, prepare_a))
            _settle()
        diffs = [100.0 * (y - x) / x for x, y in zip(n1, n2)]
        sigma_null = statistics.pstdev(diffs)
    return ABResult(a_all, b_all, sigma_null)


def report_line(name, res):
    """One standardized line: name, A, B, delta, decision."""
    thresh = MIN_DETECTABLE_PCT
    if res.sigma_null_pct is not None:
        thresh = max(2.0 * res.sigma_null_pct, MIN_DETECTABLE_PCT)
    verdict = "WIN" if res.delta_pct <= -thresh else (
        "LOSS" if res.delta_pct >= thresh else "NOISE")
    null = (f" sigma_null={res.sigma_null_pct:.2f}%" if res.sigma_null_pct
            is not None else "")
    return (f"{name:<6} A={res.a_ms:8.2f}ms B={res.b_ms:8.2f}ms "
            f"delta={res.delta_pct:+.2f}% (thresh {thresh:.2f}%) "
            f"-> {verdict}{null}")


def regress(name, res, baseline_ms, warn_pct=2.0, fail_pct=5.0):
    """Perf-gate check: compare B against a committed baseline number."""
    d = 100.0 * (res.b_ms - baseline_ms) / baseline_ms
    if d >= fail_pct:
        return f"{name}: FAIL {d:+.2f}% vs baseline {baseline_ms:.2f}ms"
    if d >= warn_pct:
        return f"{name}: WARN {d:+.2f}% vs baseline {baseline_ms:.2f}ms"
    return f"{name}: ok {d:+.2f}% vs baseline {baseline_ms:.2f}ms"

def solo_median(run, prepare=None, warmup=WARMUP, null_arm=False):
    """Single-arm benchmark under the same discipline as ab_median.

    The perf-gate arm of the C13 protocol (colab_perfgate.py): WARMUP warmup
    calls, then N_TRIALS timed calls of ONE callable with the clock-settle
    barrier between trials; the reported number is the median (not min).
    prepare() (fresh destructive state, e.g. W.copy_(W0), C.copy_(C0) before
    a mutating update-kernel call) runs OUTSIDE timing, exactly like the
    prepare_* arms of ab_median.

    Returns ABResult with a_ms == b_ms == the median, so the existing
    regress() gate check consumes it directly (it reads res.b_ms); delta_pct
    is 0 by construction and a_all == b_all == the trial list.

    null_arm=True adds the A/A calibration (same callable measured twice per
    trial pair) and reports sigma_null_pct — run it on the first gate
    measurement of a session.
    """
    for _ in range(warmup):
        if prepare is not None:
            prepare()
        run()
    _settle()
    trials = []
    for _ in range(N_TRIALS):
        trials.append(_time_once(run, prepare))
        _settle()
    sigma_null = None
    if null_arm:
        import statistics
        n1, n2 = [], []
        for _ in range(N_TRIALS):
            n1.append(_time_once(run, prepare))
            n2.append(_time_once(run, prepare))
            _settle()
        diffs = [100.0 * (y - x) / x for x, y in zip(n1, n2)]
        sigma_null = statistics.pstdev(diffs)
    return ABResult(trials, trials, sigma_null)
