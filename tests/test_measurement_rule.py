"""Unit tests for the standing measurement rule encoding (2026-11-20).

Pure-python: no GPU, no torch required. The ClockSampler is tested against a
fake nvidia-smi shell script (telemetry pass case) and a nonexistent binary
(failure case — the cause must be surfaced, never silently dropped).
"""
import os
import stat
import time

import pytest

from bench_protocol import ABResult, ClockSampler, report_line, spread


def test_spread_basic():
    xs = [float(x) for x in range(1, 12)]  # 1..11, median 6
    s = spread(xs)
    assert s["n"] == 11
    assert s["p10"] <= s["p90"]
    assert s["iqr"] >= 0
    assert s["half_iqr_pct"] == pytest.approx(100.0 * (s["iqr"] / 2) / 6.0)


def test_spread_tiny_sample_is_honest_zero():
    s = spread([5.0, 5.0])
    assert s["n"] == 2
    assert s["half_iqr_pct"] == 0.0


def test_spread_empty():
    assert spread([])["n"] == 0


def test_result_carries_spread_and_report_line_prints_it():
    a = [100.0, 100.5, 99.5, 100.2, 99.8, 100.1]
    b = [50.0, 50.2, 49.8, 50.1, 49.9, 50.0]
    res = ABResult(a, b)
    assert res.a_spread["n"] == 6 and res.b_spread["n"] == 6
    line = report_line("head", res)
    assert "spreadA=±" in line and "spreadB=±" in line
    assert "-> WIN" in line  # -50% delta, far past any threshold


def test_sub_threshold_delta_is_noise_not_a_verdict():
    a = [100.0] * 6
    b = [100.5] * 6  # +0.5% < MIN_DETECTABLE_PCT (1.0)
    res = ABResult(a, b)
    assert "-> NOISE" in report_line("dx", res)


def test_null_sigma_raises_threshold():
    a = [100.0] * 6
    b = [98.5] * 6  # -1.5%: would be WIN at 1% bar, but sigma_null=2 -> NOISE
    res = ABResult(a, b, sigma_null_pct=2.0)
    assert "-> NOISE" in report_line("update", res)


def test_clock_sampler_unavailable_surfaces_cause(tmp_path):
    clk = ClockSampler(str(tmp_path / "clocks.csv"),
                       binary="/nonexistent/nvidia-smi-xyz")
    clk.start()
    assert clk.available is False
    summary = clk.stop()
    assert summary["available"] is False
    assert summary["samples"] == 0
    assert "No such file" in summary["cause"]


def _fake_nvidia_smi(tmp_path, rows):
    script = tmp_path / "nvidia-smi-fake.sh"
    body = "\n".join(f'  echo "{r}"' for r in rows)
    script.write_text(
        "#!/bin/sh\n"
        'echo "timestamp, clocks.sm, clocks.max.sm, power.draw, temperature.gpu, utilization.gpu"\n'
        f"{body}\n"
        "sleep 30\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


@pytest.mark.parametrize("clocks,expect_droop", [
    ([1500, 1500, 1500], False),
    ([1500, 1500, 1000], True),   # min < 90% of median -> droop signature
])
def test_clock_sampler_parses_fake_telemetry(tmp_path, clocks, expect_droop):
    rows = [f"2026/11/20 22:00:00.000, {c} MHz, 1590 MHz, 30.5 W, 60 C, 99 %"
            for c in clocks]
    binary = _fake_nvidia_smi(tmp_path, rows)
    clk = ClockSampler(str(tmp_path / "clocks.csv"), binary=binary)
    clk.start()
    time.sleep(0.4)  # fake prints 3 rows immediately, then idles
    summary = clk.stop()
    assert summary["available"] is True, summary.get("cause")
    assert summary["samples"] == len(clocks)
    assert summary["sm_clock_mhz_min"] == min(clocks)
    assert summary["sm_clock_mhz_max"] == max(clocks)
    assert summary["sustained_droop"] is expect_droop
    assert summary["temp_c_max"] == 60.0


def test_step_ab_driver_still_compiles():
    # The driver is a source string executed on the VM; a syntax error there
    # would only surface after burning a Colab session. Compile it here.
    import scripts.colab_step_ab as step_ab
    compile(step_ab.DRIVER, "step_ab_driver.py", "exec")
