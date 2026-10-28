"""Flip-rate trajectory verdict — pure logic, no CUDA, CPU-testable.

Pre-registered 2026-10-26, BEFORE any measurement was taken. The criteria
below are the gate; changing them after seeing numbers invalidates the gate.

─── What the gate is for ───────────────────────────────────────────────────
The ternary optimizer has no learning rate and no damping: the counter
accumulates sign(dW) and the weight flips when |counter| > threshold, after
which the counter resets to 0. A weight that keeps receiving a *consistent*
gradient therefore flips every `threshold` steps forever — a limit cycle. The
observable signature of a limit cycle is a flip rate that does NOT decay.

The gate therefore asks a question the noise schedule can answer:
when the gradient-noise share is removed, does the flip activity stop?

Two arms, same seed / data / model / shapes, differing only in the schedule:

  * main arm    — noise share falls (noisy window -> clean window)
  * control arm — noise share stays constant for the whole run

The control calibrates what "noise-driven flip activity that does not go away"
looks like, so a decaying main arm cannot pass vacuously.

─── Expected magnitudes (from the counter mechanics, not from measurements) ─
  counter: cnt += sign(dW); flip when |cnt| > T; cnt <- 0 on flip

  * limit cycle (consistent residual gradient) ....... flip rate ~ 1/T
  * pure zero-mean gradient noise (random walk) ...... flip rate ~ 1/T^2
  * settled (dW == 0, counter frozen) ................ flip rate ~ 0

For T = 32 that is 3.125% (limit cycle) vs 0.098% (noise floor) vs 0 — a
factor of T between the two live regimes, so the three are separable.

─── Criteria (all three must hold) ────────────────────────────────────────
  V  non-vacuity  main/noisy flip rate >= 0.25/T^2
                  (the noise must actually drive flips; below this the
                  decay test would be measuring nothing)
  M  decay        main/clean <= 0.10 * main/noisy
                  (removing the noise must cut the flip rate by >= 10x;
                  a limit cycle keeps main/clean near main/noisy and fails)
  C  control      control/clean >= 0.50 * control/noisy
                  (constant noise must SUSTAIN the flip rate — this is what
                  makes criterion M meaningful rather than vacuous)
"""

from __future__ import annotations

# ── Pre-registered thresholds ──────────────────────────────────────────────
# Fraction of the theoretical 1/T^2 random-walk floor that the noisy window
# must reach for the test to be non-vacuous.
NON_VACUITY_FLOOR_FRACTION = 0.25

# The clean window must be at most this fraction of the noisy window.
MAX_CLEAN_TO_NOISY = 0.10

# The control arm (constant noise) must keep at least this fraction of its
# own early-window flip rate.
MIN_CONTROL_RATIO = 0.50

# ── Pre-registered counter dead-zone damping ───────────────────────────────
# Pre-registered 2026-10-28: counter dead-zone default. |dW| below this does
# not accumulate, so a consistent residual gradient can no longer drive the
# counter to threshold every ~T steps. Sensitivity values 1e-4 / 1e-2 to be
# swept on T4 if the default does not break the limit cycle.
DEAD_ZONE_DEFAULT = 1e-3

# With dead_zone applied, the clean-window flip rate must be at most this
# fraction of the clean-window flip rate WITHOUT dead_zone — damping must
# measurably reduce flip activity beyond the noise schedule alone.
MAX_DAMPED_CLEAN_RATIO = 0.50

# With dead_zone applied, the clean-window flip rate must meet the original
# M criterion (decay when noise removed) — damping must restore the decay
# that the limit cycle destroyed.
MAX_DAMPED_CLEAN_TO_NOISY = 0.10


def _ratio(numerator: float, denominator: float) -> float:
    """Safe ratio: an empty denominator yields +inf (i.e. "no decay")."""
    if denominator <= 0.0:
        return float("inf") if numerator > 0.0 else 0.0
    return numerator / denominator


def evaluate_flip_rate(
    main: dict,
    control: dict,
    threshold: int = 32,
    non_vacuity_floor_fraction: float = NON_VACUITY_FLOOR_FRACTION,
    max_clean_to_noisy: float = MAX_CLEAN_TO_NOISY,
    min_control_ratio: float = MIN_CONTROL_RATIO,
) -> dict:
    """Evaluate the flip-rate trajectory gate.

    Args:
        main:     {"noisy": flip_rate, "clean": flip_rate} for the arm whose
                  noise share falls over the run.
        control:  {"noisy": flip_rate, "clean": flip_rate} for the arm whose
                  noise share stays constant.  Both windows are measured at
                  the same noise level in the control, so the keys describe
                  window POSITION (early / late), not noise level.
        threshold: the counter threshold T used by the model.

    Returns:
        dict with "passed", "checks" and the measured ratios.
    """
    main_noisy = float(main["noisy"])
    main_clean = float(main["clean"])
    ctrl_noisy = float(control["noisy"])
    ctrl_clean = float(control["clean"])

    floor = non_vacuity_floor_fraction / float(threshold) ** 2
    main_ratio = _ratio(main_clean, main_noisy)
    ctrl_ratio = _ratio(ctrl_clean, ctrl_noisy)

    checks = [
        {
            "name": "V_non_vacuity",
            "ok": main_noisy >= floor,
            "value": main_noisy,
            "threshold": floor,
            "detail": (
                f"main noisy-window flip rate {main_noisy:.6g} "
                f"(>= {floor:.6g} = {non_vacuity_floor_fraction}/T^2, T={threshold})"
            ),
        },
        {
            "name": "M_decay_when_noise_removed",
            "ok": main_ratio <= max_clean_to_noisy,
            "value": main_ratio,
            "threshold": max_clean_to_noisy,
            "detail": (
                f"main clean/noisy flip-rate ratio {main_ratio:.6g} "
                f"(<= {max_clean_to_noisy})"
            ),
        },
        {
            "name": "C_control_sustains_flips",
            "ok": ctrl_ratio >= min_control_ratio,
            "value": ctrl_ratio,
            "threshold": min_control_ratio,
            "detail": (
                f"control late/early flip-rate ratio {ctrl_ratio:.6g} "
                f"(>= {min_control_ratio})"
            ),
        },
    ]

    return {
        "passed": all(c["ok"] for c in checks),
        "checks": checks,
        "ratios": {
            "main_clean_over_noisy": main_ratio,
            "control_late_over_early": ctrl_ratio,
        },
        "flip_rates": {
            "main_noisy": main_noisy,
            "main_clean": main_clean,
            "control_noisy": ctrl_noisy,
            "control_clean": ctrl_clean,
        },
        "non_vacuity_floor": floor,
    }


def evaluate_damping(
    main_clean_no_dz: float,
    main_clean_dz: float,
    main_noisy: float,
    dead_zone: float = DEAD_ZONE_DEFAULT,
) -> dict:
    """Evaluate whether the counter dead zone breaks the limit cycle.

    Args:
        main_clean_no_dz: clean-window flip rate WITHOUT dead zone (limit
            cycle keeps this near main_noisy, ratio ~ 1.0).
        main_clean_dz: clean-window flip rate WITH dead zone — damping should
            push this toward 0.
        main_noisy: noisy-window flip rate (same for both, the noise schedule
            doesn't change between the two measurements).
        dead_zone: the dead-zone value used for the dz measurement.

    Returns:
        dict with "passed", "checks" and the measured ratios.
    """
    main_clean_no_dz = float(main_clean_no_dz)
    main_clean_dz = float(main_clean_dz)
    main_noisy = float(main_noisy)

    dz_ratio = _ratio(main_clean_dz, main_clean_no_dz)
    dz_to_noisy = _ratio(main_clean_dz, main_noisy)

    checks = [
        {
            "name": "D1_damping_reduces_flips",
            "ok": dz_ratio <= MAX_DAMPED_CLEAN_RATIO,
            "value": dz_ratio,
            "threshold": MAX_DAMPED_CLEAN_RATIO,
            "detail": (
                f"dead-zone clean/no-clean flip-rate ratio {dz_ratio:.6g} "
                f"(<= {MAX_DAMPED_CLEAN_RATIO})"
            ),
        },
        {
            "name": "D2_damped_meets_decay",
            "ok": dz_to_noisy <= MAX_DAMPED_CLEAN_TO_NOISY,
            "value": dz_to_noisy,
            "threshold": MAX_DAMPED_CLEAN_TO_NOISY,
            "detail": (
                f"dead-zone clean/noisy flip-rate ratio {dz_to_noisy:.6g} "
                f"(<= {MAX_DAMPED_CLEAN_TO_NOISY})"
            ),
        },
    ]

    return {
        "passed": all(c["ok"] for c in checks),
        "checks": checks,
        "ratios": {
            "damped_clean_over_no_dz": dz_ratio,
            "damped_clean_over_noisy": dz_to_noisy,
        },
        "flip_rates": {
            "main_noisy": main_noisy,
            "main_clean_no_dz": main_clean_no_dz,
            "main_clean_dz": main_clean_dz,
        },
        "dead_zone": dead_zone,
    }
