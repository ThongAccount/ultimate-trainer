"""Pure-Python tests for the perf gate logic (no torch, no GPU, runs anywhere).

Covers:
- tests/perf_baseline_t4.json schema: loads, has all required fields, and the
  per-kernel numbers validate against the same checks the gate performs
  (schema version, all 3 shapes x 3 kernels, step_ms present).
- bench_protocol.regress thresholds: synthetic medians produce exactly
  ok / WARN / FAIL strings, including boundary cases (+2.00% -> WARN,
  +5.00% -> FAIL, improvements never FAIL/WARN).
- bench_protocol.solo_median result shape: a solo ABResult can be consumed
  by regress() directly (a_ms == b_ms, delta 0).

Run:
    python3 -m pytest tests/test_perfgate_logic.py -q
    python3 tests/test_perfgate_logic.py
"""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                      # bench_protocol lives in tests/
sys.path.insert(0, os.path.dirname(HERE))      # repo root (for colab_perfgate)

from bench_protocol import ABResult, regress           # noqa: E402
import bench_protocol as bp                              # noqa: E402

BASELINE_PATH = os.path.join(HERE, "perf_baseline_t4.json")

SCHEMA_VERSION = 1
SHAPES = ("fc1", "fc2", "head")
KERNELS = ("fwd", "dx", "update")


class FakeResult:
    """Stands in for an ABResult: regress() only reads res.b_ms."""

    __slots__ = ("b_ms",)

    def __init__(self, b_ms):
        self.b_ms = b_ms


# ── regress() thresholds ────────────────────────────────────────────────

class TestRegressThresholds:
    def test_ok_when_equal(self):
        assert regress("x", FakeResult(100.0), 100.0) == \
            "x: ok +0.00% vs baseline 100.00ms"

    def test_ok_when_improved(self):
        # a 20% speedup is ok — the gate fails on regression only
        assert ": ok " in regress("x", FakeResult(80.0), 100.0)

    def test_warn_just_above_2pct(self):
        s = regress("x", FakeResult(102.5), 100.0)
        assert s.startswith("x: WARN ")
        assert "+2.50%" in s

    def test_warn_boundary_exactly_2pct(self):
        # regress uses >= — exactly at the threshold is already WARN
        assert regress("x", FakeResult(102.0), 100.0).startswith("x: WARN ")

    def test_fail_at_5pct(self):
        s = regress("x", FakeResult(105.0), 100.0)
        assert s.startswith("x: FAIL ")
        assert "+5.00%" in s

    def test_fail_above_5pct(self):
        assert regress("x", FakeResult(130.0), 100.0).startswith("x: FAIL ")

    def test_fail_boundary_exactly_5pct(self):
        # >= fail_pct -> FAIL even at exactly the threshold
        assert regress("x", FakeResult(105.0), 100.0).startswith("x: FAIL ")

    def test_below_warn_is_ok(self):
        assert regress("x", FakeResult(101.99), 100.0).startswith("x: ok ")

    def test_negative_regression_never_fails(self):
        # even a -90% "regression" (10x faster) is ok, not WARN/FAIL
        s = regress("x", FakeResult(10.0), 100.0)
        assert ": WARN" not in s and ": FAIL" not in s

    def test_custom_thresholds(self):
        # gate defaults are warn=2, fail=5; the API allows overrides
        assert regress("x", FakeResult(101.0), 100.0,
                       warn_pct=0.5, fail_pct=1.5).startswith("x: WARN ")
        assert regress("x", FakeResult(102.0), 100.0,
                       warn_pct=0.5, fail_pct=1.5).startswith("x: FAIL ")

    def test_abresult_delta(self):
        r = ABResult([10.0, 20.0, 30.0], [10.0, 20.0, 30.0])
        assert r.a_ms == r.b_ms == 20.0
        assert r.delta_pct == 0.0


# ── solo_median on a torch-less box ────────────────────────────────────

class TestSoloMedianWithoutTorch:
    def test_solo_median_pure_python(self):
        # With torch unavailable (or CPU-only), solo_median still times a
        # plain callable with the same median/statistics machinery.
        # _time_once calls torch.cuda.synchronize() — on a torch-less box
        # bp.torch is None and it would AttributeError; patch it out with a
        # no-op to keep this test pure.
        class _FakeTorch:
            class cuda:
                @staticmethod
                def synchronize():
                    pass

                @staticmethod
                def _sleep(n):
                    pass

                @staticmethod
                def is_available():
                    return False

        old = bp.torch
        bp.torch = _FakeTorch()
        try:
            calls = []
            res = bp.solo_median(lambda: calls.append(1), warmup=1)
            assert res.a_ms == res.b_ms          # solo: both arms are the same
            assert res.delta_pct == 0.0
            assert len(calls) == 1 + bp.N_TRIALS  # warmup + trials
            # and regress() consumes it directly:
            line = regress("solo", res, res.b_ms)
            assert line.startswith("solo: ok +0.00%")
        finally:
            bp.torch = old


# ── baseline JSON schema ───────────────────────────────────────────────

class TestBaselineSchema:
    def load(self):
        with open(BASELINE_PATH) as f:
            return json.load(f)

    def test_loads_and_has_schema_version(self):
        b = self.load()
        assert b["schema"] == SCHEMA_VERSION

    def test_meta_marks_provisional(self):
        b = self.load()
        assert "provisional" in b["_meta"]["status"]
        assert "regenerate with --update-baseline" in b["_meta"]["note"]

    def test_all_shapes_and_kernels_present(self):
        b = self.load()
        for shape in SHAPES:
            for kernel in KERNELS:
                v = b["kernels"][shape][f"{kernel}_ms"]
                assert isinstance(v, (int, float)), \
                    f"kernels.{shape}.{kernel}_ms missing or not numeric"
                assert v > 0, f"kernels.{shape}.{kernel}_ms must be positive"

    def test_step_ms(self):
        b = self.load()
        assert b["step_ms"] == 3091.9  # HANDOFF §23 same-instance post-F1 avg

    def test_config_matches_gate(self):
        # config block must agree with the gate's own constants so the
        # committed baseline and the runner never drift apart silently
        b = self.load()
        c = b["config"]
        assert c["batch"] == 16384
        assert c["threshold"] == 32
        assert c["n_trials"] == 10 and c["warmup"] == 10
        for shape, (inn, out) in {
            "fc1": (1024, 4096), "fc2": (4096, 1024), "head": (1024, 50272),
        }.items():
            assert c["shapes"][shape] == {"in": inn, "out": out}

    def test_known_numbers_match_handoff(self):
        b = self.load()
        k = b["kernels"]
        # fwd: HANDOFF §23 C1 post-F1 isolated (Modal T4)
        assert k["fc1"]["fwd_ms"] == pytest.approx(32.2)
        assert k["fc2"]["fwd_ms"] == pytest.approx(32.7)
        assert k["head"]["fwd_ms"] == pytest.approx(401.4)
        # dX: HANDOFF §25 C9a Colab T4 (fc2 borrowed from fc1 — never isolated)
        assert k["fc1"]["dx_ms"] == pytest.approx(41.03)
        assert k["head"]["dx_ms"] == pytest.approx(520.46)
        # update: HANDOFF §23 C3 unmodified-kernel baseline
        assert k["fc1"]["update_ms"] == pytest.approx(45.40)
        assert k["fc2"]["update_ms"] == pytest.approx(47.16)
        assert k["head"]["update_ms"] == pytest.approx(557.8)

    def test_baseline_validates_against_runner_constants(self):
        # cross-check: the gate runner's own SHAPES/THRESHOLD must match the
        # committed config exactly (imports colab_perfgate, which must be
        # torch-free at module level — it imports torch only inside main())
        import colab_perfgate as gate

        b = self.load()
        c = b["config"]
        assert gate.THRESHOLD == c["threshold"]
        assert gate.BATCH == c["batch"]
        for (name, inn, out) in gate.SHAPES:
            assert c["shapes"][name] == {"in": inn, "out": out}

    def test_synthetic_regression_on_baseline_numbers(self):
        # end-to-end logic check: synthetic medians vs the committed baseline
        # produce the expected gate verdicts, exactly as colab_perfgate does
        b = self.load()
        for shape in SHAPES:
            for kernel in KERNELS:
                base = b["kernels"][shape][f"{kernel}_ms"]
                # exact match -> ok
                assert regress("t", FakeResult(base), base).startswith("t: ok ")
                # +10% -> FAIL
                assert regress("t", FakeResult(base * 1.10), base) \
                    .startswith("t: FAIL ")
                # +3% -> WARN
                assert regress("t", FakeResult(base * 1.03), base) \
                    .startswith("t: WARN ")
                # -10% (improvement) -> ok
                assert regress("t", FakeResult(base * 0.90), base) \
                    .startswith("t: ok ")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
