# Pre-registration: B2 — dX K32 + 2 outer loops (2026-11-20)

Committed on `feat/b2-dx-k32-2outer` **before any Colab allocation**.
Thresholds and decision rules below are binding; they are not changed after
measurement.

## Scope

One T4 session runs `scripts/colab_b2_validate.py`, which clones the branch at
`--expect-commit` and executes, in this fixed order:

0. `smoke` — `tests/probe_fwd_spill.py --probe fwdspill --skip-timing`
   (parity-only plumbing check; abort on failure)
1. `pytest` — correctness gate, `colab_pytest.TEST_MODULES`
2. `b2_parity` — bit-exact parity of the B2 variant vs production kernel
   (fc1, fc2, head; counters past threshold so flips actually occur)
3. `b2_ab` — same-VM whole-step A/B, 10-trial paired median vs measured
   custom baseline 2976.4 ms (`tests/perf_baseline_t4.json`)

No production file is modified outside the B2 change. Every probe compiles
its variant side-by-side with the production kernel in one process and
**must pass a bit-exact parity gate** before timing; a probe that fails
parity is a FAIL, not a result.

## Hypothesis

B2: dX K64 took dX to 3 blocks/SM (20 KB smem); keep K32 tiles but add a
second r0-block per warp (2 outer loops × K32), giving 4 MMAs/sync at 12 KB
+ improved dY reuse, without hitting the 20 KB smem cliff.

## Pre-registered predictions

| Item | Prediction | Falsifier / close rule |
|------|-----------|----------------------|
| dX B2 | −3..−8% head dX kernel time; step −1..−3% | win/loss judged per bench_protocol rule below |
| parity | bit-exact on all shapes (fc1, fc2, head) | FAIL if not bit-exact |
| pytest | 103 passed | any failure = FAIL |

## Decision rule (bench_protocol, mandatory)

All timings use `tests/bench_protocol.py::ab_median`: 10-trial paired
interleaved median, warmup 10, `torch.cuda._sleep` clock settle, A/A null arm
on the first shape for sigma_null. A variant counts as a win/loss only when
|Δ| > max(2*sigma_null, 1%). The old 3-trial min-of-3 protocol is retired.

## What this batch does NOT claim

- B2 is a dX-only change; fwd and update are untouched.
- Single-seed timing numbers support no convergence claim.
- The K32+2outer change may affect occupancy; if occupancy drops, the gain
  may not materialize.

## Artefacts

`results/b2.<tag>.{json,log}` plus per-step logs on the VM, downloaded with
`colab download` before `colab stop`. rc: 0 = all gates pass, 1 = a gate
failed (valid result), 2 = commit-mismatch refusal (invalid; never commit
rc=2 artefacts).