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

Standing measurement rule (2026-11-20, post-sweep ruling): every A/B verdict
is interleaved same-session (ab_median above), median-over-steps, reports the
SPREAD of each arm (spread()/report_line below), and carries clock telemetry
(ClockSampler below) so a clock/power confound is visible in the artefact
instead of being asserted after the fact (the I2 lesson). Telemetry that
cannot be collected is reported unavailable WITH its cause, never dropped.
"""

import os
import statistics
import subprocess
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


def spread(times):
    """Dispersion of one arm's timed samples, for the standing rule.

    Returns dict with p10/p90 (ms) and half_iqr_pct — half the interquartile
    range as % of the median, i.e. the typical ± wobble of the arm. A |delta|
    smaller than the arms' spread is session noise, not an effect; the
    verdict threshold (sigma_null / MIN_DETECTABLE_PCT) stays the decision
    mechanism, the spread is the reported evidence alongside it.
    """
    s = sorted(times)
    if not s:
        return {"n": 0, "p10": float("nan"), "p90": float("nan"),
                "iqr": float("nan"), "half_iqr_pct": float("nan")}
    med = statistics.median(s)
    if len(s) < 4:  # quantiles need enough points; report zero-width honestly
        return {"n": len(s), "p10": s[0], "p90": s[-1], "iqr": 0.0,
                "half_iqr_pct": 0.0}
    q4 = statistics.quantiles(s, n=4)   # Q1, Q2, Q3
    q10 = statistics.quantiles(s, n=10)  # p10 .. p90
    iqr = q4[2] - q4[0]
    return {"n": len(s), "p10": q10[0], "p90": q10[-1], "iqr": iqr,
            "half_iqr_pct": 100.0 * (iqr / 2.0) / med if med else float("nan")}


class ClockSampler:
    """nvidia-smi clock/power/temp telemetry riding along with a measurement.

    Same query as scripts/colab_measure_p0.py (known-good on the Colab T4
    image); throttle-reason fields are deliberately not queried because the
    field name differs across driver versions — clock droop + temperature is
    the evidence the rule needs.

    Usage:
        clk = ClockSampler("results/clocks.head.csv")
        clk.start()
        res = ab_median(run_a, run_b, prepare_a, prepare_b)
        summary = clk.stop()   # dict; available=False + cause if not sampled
    """

    def __init__(self, csv_path, interval_ms=500, binary="nvidia-smi"):
        self.csv_path = csv_path
        self.interval_ms = interval_ms
        self.binary = binary
        self._proc = None
        self._fh = None
        self.available = True
        self.cause = None

    def start(self):
        if self._proc is not None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True)
        try:
            self._fh = open(self.csv_path, "w")
            self._proc = subprocess.Popen(
                [self.binary,
                 "--query-gpu=timestamp,clocks.sm,clocks.max.sm,power.draw,"
                 "temperature.gpu,utilization.gpu",
                 "--format=csv", "-lms", str(self.interval_ms)],
                stdout=self._fh, stderr=subprocess.DEVNULL)
        except OSError as e:
            # Surface the cause; a missing sampler must not fail the benchmark.
            self.available = False
            self.cause = f"{type(e).__name__}: {e}"
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    @staticmethod
    def _num(field):
        # "1590 MHz" / "28.8 W" / "43 C" -> float; "[N/A]" etc -> None
        try:
            return float(field.strip().split()[0])
        except (ValueError, IndexError):
            return None

    def stop(self):
        """Terminate the sampler and return the parsed summary dict."""
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        summary = {"available": self.available, "csv": self.csv_path,
                   "samples": 0}
        if self.cause:
            summary["cause"] = self.cause
        if not self.available:
            return summary
        clocks, powers, temps = [], [], []
        try:
            with open(self.csv_path) as f:
                lines = f.read().splitlines()
        except OSError as e:
            summary["available"] = False
            summary["cause"] = f"{type(e).__name__}: {e}"
            return summary
        for line in lines[1:]:  # skip header
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            sm, pw, tp = self._num(parts[1]), self._num(parts[3]), self._num(parts[4])
            if sm is not None:
                clocks.append(sm)
            if pw is not None:
                powers.append(pw)
            if tp is not None:
                temps.append(tp)
        summary["samples"] = len(clocks)
        if clocks:
            summary.update({
                "sm_clock_mhz_min": min(clocks),
                "sm_clock_mhz_median": statistics.median(clocks),
                "sm_clock_mhz_max": max(clocks),
                "power_w_median": statistics.median(powers) if powers else None,
                "temp_c_max": max(temps) if temps else None,
                # A sustained droop is the clock-throttle signature; if the
                # min is far below the median, say so in the artefact.
                "sustained_droop": (min(clocks) < 0.9 * statistics.median(clocks)),
            })
        else:
            summary["available"] = False
            summary["cause"] = "no parseable samples in " + self.csv_path
        return summary

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


class ABResult:
    __slots__ = ("a_ms", "b_ms", "delta_pct", "a_all", "b_all", "sigma_null_pct",
                 "a_spread", "b_spread")

    def __init__(self, a_all, b_all, sigma_null_pct=None):
        self.a_all, self.b_all = list(a_all), list(b_all)
        self.a_ms = statistics.median(self.a_all)
        self.b_ms = statistics.median(self.b_all)
        self.delta_pct = 100.0 * (self.b_ms - self.a_ms) / self.a_ms
        self.sigma_null_pct = sigma_null_pct
        self.a_spread = spread(self.a_all)
        self.b_spread = spread(self.b_all)


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
            f"-> {verdict}{null}"
            f" spreadA=±{res.a_spread['half_iqr_pct']:.2f}%"
            f" spreadB=±{res.b_spread['half_iqr_pct']:.2f}%")


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
