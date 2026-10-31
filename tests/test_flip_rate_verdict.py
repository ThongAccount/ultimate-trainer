"""CPU tests for the flip-rate trajectory verdict (no CUDA needed).

These pin the PRE-REGISTERED criteria of tools/flip_rate_verdict.py: the
gate must pass a healthy trajectory, fail a limit cycle, and refuse to pass
a vacuous run.  The thresholds themselves were fixed before any measurement.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.flip_rate_verdict import evaluate_flip_rate


def _check(res, name):
    return next(c for c in res["checks"] if c["name"] == name)


def test_healthy_trajectory_passes():
    # noisy window at the random-walk floor, clean window essentially silent,
    # control arm holding its rate up.
    res = evaluate_flip_rate(
        main={"noisy": 0.00098, "clean": 0.00002},
        control={"noisy": 0.00098, "clean": 0.00090},
        threshold=32,
    )
    assert res["passed"], res["checks"]
    assert _check(res, "V_non_vacuity")["ok"]
    assert _check(res, "M_decay_when_noise_removed")["ok"]
    assert _check(res, "C_control_sustains_flips")["ok"]


def test_limit_cycle_fails_the_decay_criterion():
    # The weights keep flipping after the noise is gone: clean ~= noisy.
    res = evaluate_flip_rate(
        main={"noisy": 0.00098, "clean": 0.03125},
        control={"noisy": 0.00098, "clean": 0.00095},
        threshold=32,
    )
    assert not res["passed"]
    assert not _check(res, "M_decay_when_noise_removed")["ok"]
    # the other two criteria still hold — the failure is specifically the decay
    assert _check(res, "V_non_vacuity")["ok"]
    assert _check(res, "C_control_sustains_flips")["ok"]


def test_vacuous_run_is_refused():
    # Nothing ever flips in the noisy window: the decay test would be
    # measuring an absence, so the gate must fail on non-vacuity.
    res = evaluate_flip_rate(
        main={"noisy": 0.0, "clean": 0.0},
        control={"noisy": 0.00098, "clean": 0.00090},
        threshold=32,
    )
    assert not res["passed"]
    assert not _check(res, "V_non_vacuity")["ok"]


def test_control_that_also_decays_is_refused():
    # If the constant-noise control ALSO decays, the main arm's decay proves
    # nothing about the noise schedule — the gate must refuse.
    res = evaluate_flip_rate(
        main={"noisy": 0.00098, "clean": 0.00002},
        control={"noisy": 0.00098, "clean": 0.00001},
        threshold=32,
    )
    assert not res["passed"]
    assert not _check(res, "C_control_sustains_flips")["ok"]


def test_zero_denominator_is_not_a_pass():
    # main/noisy == 0 and clean == 0: ratio 0/0 must not read as "decayed".
    res = evaluate_flip_rate(
        main={"noisy": 0.0, "clean": 0.0},
        control={"noisy": 0.0, "clean": 0.0},
        threshold=32,
    )
    assert not res["passed"]
    assert not _check(res, "V_non_vacuity")["ok"]


def test_non_vacuity_floor_scales_with_threshold():
    # The floor is derived from T^2, not hard-coded: a different threshold
    # must move the floor.
    t8 = evaluate_flip_rate(
        main={"noisy": 0.00098, "clean": 0.0},
        control={"noisy": 0.00098, "clean": 0.00098},
        threshold=8,
    )
    t32 = evaluate_flip_rate(
        main={"noisy": 0.00098, "clean": 0.0},
        control={"noisy": 0.00098, "clean": 0.00098},
        threshold=32,
    )
    assert t8["non_vacuity_floor"] > t32["non_vacuity_floor"]
    # at T=8 the floor is 0.25/64 = 0.0039, so 0.00098 is below it
    assert not t8["passed"]
    assert t32["passed"]


def test_damping_passes_when_dead_zone_breaks_limit_cycle():
    # Without dead zone: clean ~= noisy (limit cycle, ratio ~ 1.0).
    # With dead zone: clean drops well below noisy — damping works.
    from tools.flip_rate_verdict import evaluate_damping
    res = evaluate_damping(
        main_clean_no_dz=0.03125,   # limit cycle: clean ~= noisy
        main_clean_dz=0.001,        # dead zone cuts flips drastically
        main_noisy=0.03125,
        dead_zone=1e-3,
    )
    assert res["passed"], res["checks"]
    assert _check(res, "D1_damping_reduces_flips")["ok"]
    assert _check(res, "D2_damped_meets_decay")["ok"]


def test_damping_fails_when_dead_zone_too_small():
    # Dead zone too small: clean still ~= noisy, no damping.
    from tools.flip_rate_verdict import evaluate_damping
    res = evaluate_damping(
        main_clean_no_dz=0.03125,
        main_clean_dz=0.025,        # barely any reduction
        main_noisy=0.03125,
        dead_zone=1e-6,
    )
    assert not res["passed"]
    assert not _check(res, "D1_damping_reduces_flips")["ok"]
    assert not _check(res, "D2_damped_meets_decay")["ok"]


def test_damping_fails_when_no_reduction():
    # dead_zone=0 means no damping: clean_dz == clean_no_dz, ratio = 1.0.
    from tools.flip_rate_verdict import evaluate_damping
    res = evaluate_damping(
        main_clean_no_dz=0.03125,
        main_clean_dz=0.03125,
        main_noisy=0.03125,
        dead_zone=0.0,
    )
    assert not res["passed"]
    assert not _check(res, "D1_damping_reduces_flips")["ok"]


def test_flip_budget_passes_when_budget_breaks_limit_cycle():
    # Without budget: clean ~= noisy (limit cycle, ratio ~ 1.0).
    # With budget: clean drops well below noisy — budget works.
    from tools.flip_rate_verdict import evaluate_flip_budget
    res = evaluate_flip_budget(
        main_clean_no_budget=0.03125,   # limit cycle: clean ~= noisy
        main_clean_budget=0.001,        # budget cuts flips drastically
        main_noisy=0.03125,
        budget=10,
    )
    assert res["passed"], res["checks"]
    assert _check(res, "F1_budget_reduces_flips")["ok"]
    assert _check(res, "F2_budgeted_meets_decay")["ok"]


def test_flip_budget_fails_when_budget_too_large():
    # Budget too large: clean still ~= noisy, no breaking of the cycle.
    from tools.flip_rate_verdict import evaluate_flip_budget
    res = evaluate_flip_budget(
        main_clean_no_budget=0.03125,
        main_clean_budget=0.025,        # barely any reduction
        main_noisy=0.03125,
        budget=50,
    )
    assert not res["passed"]
    assert not _check(res, "F1_budget_reduces_flips")["ok"]
    assert not _check(res, "F2_budgeted_meets_decay")["ok"]


def test_flip_budget_fails_when_no_reduction():
    # No effective budget: clean_budget == clean_no_budget, ratio = 1.0.
    from tools.flip_rate_verdict import evaluate_flip_budget
    res = evaluate_flip_budget(
        main_clean_no_budget=0.03125,
        main_clean_budget=0.03125,
        main_noisy=0.03125,
        budget=10**9,
    )
    assert not res["passed"]
    assert not _check(res, "F1_budget_reduces_flips")["ok"]


def test_flip_budget_echoes_budget_value():
    # budget=1 is a valid edge: it must be echoed into the result.
    from tools.flip_rate_verdict import evaluate_flip_budget
    res = evaluate_flip_budget(0.03125, 0.001, 0.03125, budget=1)
    assert res["flip_budget"] == 1
    assert res["passed"], res["checks"]


def test_flip_budget_fails_when_noisy_zero():
    # F2 denominator empty (noisy=0): 0/0 must not pass vacuously.
    from tools.flip_rate_verdict import evaluate_flip_budget
    res = evaluate_flip_budget(0.03, 0.0, 0.0, budget=10)
    assert not res["passed"]
    assert not _check(res, "F0_non_vacuity")["ok"]


def test_flip_budget_fails_when_no_budget_clean_zero():
    # F1 denominator empty (no_budget=0): 0/0 must not pass vacuously.
    from tools.flip_rate_verdict import evaluate_flip_budget
    res = evaluate_flip_budget(0.0, 0.0, 0.03, budget=10)
    assert not res["passed"]
    assert not _check(res, "F0_non_vacuity")["ok"]


def test_control_zero_noisy_with_clean_positive_fails():
    # C must not pass via the +inf ratio when the control noisy window is
    # empty — a control arm that never flips under noise is vacuous.
    res = evaluate_flip_rate(
        main={"noisy": 0.00098, "clean": 0.00002},
        control={"noisy": 0.0, "clean": 0.001},
        threshold=32,
    )
    assert not res["passed"]
    assert not _check(res, "C_control_sustains_flips")["ok"]


def test_all_check_thresholds_are_print_safe():
    # Regression for the budget-run crash: the gate driver prints
    # threshold=... and F0_non_vacuity's threshold is the STRING "> 0".
    # Every check threshold from both evaluators must survive the driver's
    # formatter (numeric -> .6g, str -> as-is); a str with :.6g raises ValueError.
    from tools.flip_rate_verdict import evaluate_flip_budget, evaluate_flip_rate
    v = evaluate_flip_rate(main={"noisy": 0.01, "clean": 0.001},
                           control={"noisy": 0.01, "clean": 0.009}, threshold=32)
    fb = evaluate_flip_budget(main_clean_no_budget=0.0155, main_clean_budget=0.0004,
                              main_noisy=0.0143, budget=10)
    for res in (v, fb):
        for c in res["checks"]:
            t = c["threshold"]
            rendered = format(t, ".6g") if isinstance(t, (int, float)) else str(t)
            assert isinstance(rendered, str) and rendered


def test_control_arm_is_exempt_from_flip_budget():
    # Pre-registered 2026-10-31: the control arm runs UNBUDGETED so criterion
    # C measures uncapped sustainment; the main arm carries the budget.
    from tools.flip_rate_verdict import arm_flip_budget
    assert arm_flip_budget("main", 5) == 5
    assert arm_flip_budget("main_base", 5) == 5
    assert arm_flip_budget("control", 5) == 0
    assert arm_flip_budget("control", 50) == 0
    # budget off: everything is 0 anyway
    assert arm_flip_budget("main", 0) == 0
    assert arm_flip_budget("control", 0) == 0
