# Pre-registration: I3 — custom-arm glue attribution (2026-11-20)

Committed on `feat/i3-glue-attribution` **before any Colab allocation**.
Thresholds and decision rules below are binding; they are not changed after
measurement.

## Scope

One T4 session runs `scripts/colab_i3_profile.py`, which clones the branch at
`--expect-commit` and executes, in this fixed order:

0. `smoke` — `tests/probe_fwd_spill.py --probe fwdspill --skip-timing`
   (parity-only plumbing check; abort on failure)
1. `pytest` — correctness gate, `colab_pytest.TEST_MODULES`
2. `profile` — torch profiler on the custom arm (custom_custom, custom_fwd,
   custom_dx, custom_update), 10 steps, CUDA + CPU activities

No production file is modified. This is a measurement-only item.

## Hypothesis

The custom arm (custom_custom) has ~1.2 s of "glue" (Python dispatch,
tensor movement, etc.) that isn't in the shipped cuBLAS path. Profile where
it goes.

## Pre-registered predictions

| Item | Prediction | Falsifier / close rule |
|------|-----------|----------------------|
| glue attribution | custom arm ~1.2 s of non-kernel time | report top-10 ops by self_cuda time |
| pytest | 103 passed | any failure = FAIL |

## Decision rule

This is a measurement-only item. No A/B comparison. Report the top-10 ops
by self_cuda time and the total glue time (step time minus kernel time).

## Artefacts

`results/i3.<tag>.{json,log}` plus profiler output on the VM, downloaded with
`colab download` before `colab stop`. rc: 0 = all gates pass, 1 = a gate
failed (valid result), 2 = commit-mismatch refusal (invalid; never commit
rc=2 artefacts).