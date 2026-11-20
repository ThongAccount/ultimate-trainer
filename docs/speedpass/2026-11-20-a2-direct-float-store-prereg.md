# Pre-registration: A2 — direct float store per-frag (2026-11-20)

Committed on `feat/a2-direct-float-store` **before any Colab allocation**.
Thresholds and decision rules below are binding; they are not changed after
measurement.

## Scope

One T4 session runs `scripts/colab_a2_validate.py`, which clones the branch at
`--expect-commit` and executes, in this fixed order:

0. `smoke` — `tests/probe_fwd_spill.py --probe fwdspill --skip-timing`
   (parity-only plumbing check; abort on failure)
1. `pytest` — correctness gate, `colab_pytest.TEST_MODULES`
2. `a2_parity` — bit-exact parity of the A2 variant vs production kernel
   (fc1, fc2, head; counters past threshold so flips actually occur)
3. `a2_ab` — same-VM whole-step A/B, 10-trial paired median vs measured
   custom baseline 2976.4 ms (`tests/perf_baseline_t4.json`)

No production file is modified outside the A2 change. Every probe compiles
its variant side-by-side with the production kernel in one process and
**must pass a bit-exact parity gate** before timing; a probe that fails
parity is a FAIL, not a result.

## Hypothesis

A2: `wmma::store_matrix_sync` to smem spill → in-register float convert +
direct global write. Eliminates the smem slot entirely for fwd and dX
kernels. The update kernel is NOT in scope (its spill feeds counter/threshold
logic, not float→half conversion).

## Pre-registered predictions

| Item | Prediction | Falsifier / close rule |
|------|-----------|----------------------|
| fwd A2 | −3..−8% fc1/fc2/head kernel time; step −1..−3% | win/loss judged per bench_protocol rule below |
| dX A2 | −3..−8% fc1/fc2/head kernel time; step −1..−3% | same decision rule |
| parity | bit-exact on all shapes (fc1, fc2, head) | FAIL if not bit-exact |
| pytest | 103 passed | any failure = FAIL |

## Decision rule (bench_protocol, mandatory)

All timings use `tests/bench_protocol.py::ab_median`: 10-trial paired
interleaved median, warmup 10, `torch.cuda._sleep` clock settle, A/A null arm
on the first shape for sigma_null. A variant counts as a win/loss only when
|Δ| > max(2*sigma_null, 1%). The old 3-trial min-of-3 protocol is retired.

## What this batch does NOT claim

- A2 does not apply to the update kernel (spill feeds counter logic, not
  float→half conversion).
- Single-seed timing numbers support no convergence claim.
- The fwd/dX spill removal may change occupancy; if occupancy drops, the
  gain may not materialize.

## Artefacts

`results/a2.<tag>.{json,log}` plus per-step logs on the VM, downloaded with
`colab download` before `colab stop`. rc: 0 = all gates pass, 1 = a gate
failed (valid result), 2 = commit-mismatch refusal (invalid; never commit
rc=2 artefacts).