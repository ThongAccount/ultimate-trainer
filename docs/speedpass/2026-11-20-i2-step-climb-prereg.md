# Pre-registration: I2 — step-climb root cause (2026-11-20)

Committed on `feat/i2-step-climb` **before any Colab allocation**.
Thresholds and decision rules below are binding; they are not changed after
measurement.

## Scope

One T4 session runs `scripts/colab_i2_climb.py`, which clones the branch at
`--expect-commit` and executes, in this fixed order:

0. `smoke` — `tests/probe_fwd_spill.py --probe fwdspill --skip-timing`
   (parity-only plumbing check; abort on failure)
1. `pytest` — correctness gate, `colab_pytest.TEST_MODULES`
2. `climb` — shipped arm (cublas_cublas) 300 steps, log step_ms at steps
   0, 50, 100, 150, 200, 250, 299; test empty_cache() at step 150

No production file is modified. This is a measurement-only item.

## Hypothesis

The ~530→~590 ms climb over 300 steps on the shipped arm is real;
`empty_cache()` at step 150 did not reset it → not allocator fragmentation.
Root cause unknown. Candidates: GPU clock throttle, memory fragmentation,
kernel JIT recompilation, or data-dependent kernel selection.

## Pre-registered predictions

| Item | Prediction | Falsifier / close rule |
|------|-----------|----------------------|
| climb measurement | step_ms climbs ~530→~590 over 300 steps | report step_ms at each checkpoint |
| empty_cache test | empty_cache() at step 150 does NOT reset climb | if it resets, root cause = allocator |
| pytest | 103 passed | any failure = FAIL |

## Decision rule

This is a measurement-only item. No A/B comparison. Report step_ms at
each checkpoint and whether empty_cache() resets the climb.

## Artefacts

`results/i2.<tag>.{json,log}` plus step_ms log on the VM, downloaded with
`colab download` before `colab stop`. rc: 0 = all gates pass, 1 = a gate
failed (valid result), 2 = commit-mismatch refusal (invalid; never commit
rc=2 artefacts).