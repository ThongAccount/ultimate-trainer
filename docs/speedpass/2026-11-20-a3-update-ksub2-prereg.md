# Pre-registration: A3 — update kSub=2 (2026-11-20)

Committed on `feat/a3-update-ksub2` **before any Colab allocation**.
Thresholds and decision rules below are binding; they are not changed after
measurement.

## Scope

One T4 session runs `scripts/colab_a3_validate.py`, which clones the branch at
`--expect-commit` and executes, in this fixed order:

0. `smoke` — `tests/probe_fwd_spill.py --probe fwdspill --skip-timing`
   (parity-only plumbing check; abort on failure)
1. `pytest` — correctness gate, `colab_pytest.TEST_MODULES`
2. `a3_parity` — bit-exact parity of the A3 variant vs production kernel
   (fc1, fc2, head; counters past threshold so flips actually occur)
3. `a3_ab` — same-VM whole-step A/B, 10-trial paired median vs measured
   custom baseline 2976.4 ms (`tests/perf_baseline_t4.json`)

No production file is modified outside the A3 change. Every probe compiles
its variant side-by-side with the production kernel in one process and
**must pass a bit-exact parity gate** before timing; a probe that fails
parity is a FAIL, not a result.

## Hypothesis

A3: update kSub=2 (not 4), unrolling only the X/dY load rollout, while
keeping 16-batch inner loop, i.e. 2 blocks/loop instead of 4. smem 8→12 KB,
regs +~6, sync same count.

## Pre-registered predictions

| Item | Prediction | Falsifier / close rule |
|------|-----------|----------------------|
| update A3 | −1..−3% fc1/fc2/head kernel time; step −0.5..−1% | win/loss judged per bench_protocol rule below |
| parity | bit-exact on all shapes (fc1, fc2, head) | FAIL if not bit-exact |
| pytest | 103 passed | any failure = FAIL |

## Decision rule (bench_protocol, mandatory)

All timings use `tests/bench_protocol.py::ab_median`: 10-trial paired
interleaved median, warmup 10, `torch.cuda._sleep` clock settle, A/A null arm
on the first shape for sigma_null. A variant counts as a win/loss only when
|Δ| > max(2*sigma_null, 1%). The old 3-trial min-of-3 protocol is retired.

## What this batch does NOT claim

- A3 is an update-only change; fwd and dX are untouched.
- Single-seed timing numbers support no convergence claim.
- The kSub=2 change may affect occupancy; if occupancy drops, the gain may
  not materialize.

## Artefacts

`results/a3.<tag>.{json,log}` plus per-step logs on the VM, downloaded with
`colab download` before `colab stop`. rc: 0 = all gates pass, 1 = a gate
failed (valid result), 2 = commit-mismatch refusal (invalid; never commit
rc=2 artefacts).