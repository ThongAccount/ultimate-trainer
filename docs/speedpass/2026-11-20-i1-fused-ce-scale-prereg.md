# Pre-registration: I1 — fused CE+scale (2026-11-20)

Committed on `feat/i1-fused-ce-scale` **before any Colab allocation**.
Thresholds and decision rules below are binding; they are not changed after
measurement.

## Scope

One T4 session runs `scripts/colab_i1_validate.py`, which clones the branch at
`--expect-commit` and executes, in this fixed order:

0. `smoke` — `tests/probe_fwd_spill.py --probe fwdspill --skip-timing`
   (parity-only plumbing check; abort on failure)
1. `pytest` — correctness gate, `colab_pytest.TEST_MODULES`
2. `i1_parity` — bit-exact parity of the fused CE+scale variant vs eager CE
   (logits [16352,50272] fp16; loss trajectory gate)
3. `i1_ab` — same-VM whole-step A/B, 10-trial paired median vs measured
   shipped baseline 627.6 ms

No production file is modified outside the I1 change. Every probe compiles
its variant side-by-side with the production kernel in one process and
**must pass a bit-exact parity gate** before timing; a probe that fails
parity is a FAIL, not a result.

## Hypothesis

I1: A custom fused CE+scale kernel could save the ~14 ms scale pass on the
shipped step. Measured: eager CE 24.88 ms, scale→CE 38.72 ms, torch.compile
CE 21.70 ms. A fused kernel could at best save the ~14 ms scale pass but
breaks the bit-stable gate → needs a loss-trajectory gate + re-baseline.

## Pre-registered predictions

| Item | Prediction | Falsifier / close rule |
|------|-----------|----------------------|
| fused CE+scale | ~14 ms saved on shipped step | win/loss judged per bench_protocol rule below |
| parity | bit-exact loss trajectory vs eager CE | FAIL if loss diverges |
| pytest | 103 passed | any failure = FAIL |

## Decision rule (bench_protocol, mandatory)

All timings use `tests/bench_protocol.py::ab_median`: 10-trial paired
interleaved median, warmup 10, `torch.cuda._sleep` clock settle, A/A null arm
on the first shape for sigma_null. A variant counts as a win/loss only when
|Δ| > max(2*sigma_null, 1%). The old 3-trial min-of-3 protocol is retired.

## What this batch does NOT claim

- I1 is a shipped-arm-only change; custom arm is untouched.
- Single-seed timing numbers support no convergence claim.
- The fused CE+scale kernel must preserve bit-exact loss trajectory.

## Artefacts

`results/i1.<tag>.{json,log}` plus per-step logs on the VM, downloaded with
`colab download` before `colab stop`. rc: 0 = all gates pass, 1 = a gate
failed (valid result), 2 = commit-mismatch refusal (invalid; never commit
rc=2 artefacts).