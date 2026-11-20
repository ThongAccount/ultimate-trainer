# Pre-registration: B4 — fwd fuse W decode into dY load (2026-11-20)

Committed on `feat/b4-fused-w-decode` **before any Colab allocation**.
Thresholds and decision rules below are binding; they are not changed after
measurement.

## Scope

One T4 session runs `scripts/colab_b4_validate.py`, which clones the branch at
`--expect-commit` and executes, in this fixed order:

0. `smoke` — `tests/probe_fwd_spill.py --probe fwdspill --skip-timing`
   (parity-only plumbing check; abort on failure)
1. `pytest` — correctness gate, `colab_pytest.TEST_MODULES`
2. `b4_parity` — bit-exact parity of the B4 variant vs production kernel
   (fc1, fc2, head; counters past threshold so flips actually occur)
3. `b4_ab` — same-VM whole-step A/B, 10-trial paired median vs measured
   custom baseline 2976.4 ms (`tests/perf_baseline_t4.json`)

No production file is modified outside the B4 change. Every probe compiles
its variant side-by-side with the production kernel in one process and
**must pass a bit-exact parity gate** before timing; a probe that fails
parity is a FAIL, not a result.

## Hypothesis

B4: Fuse W decode into the dY load (only decode the W you actually need).
Decode 16×16 pack in int2 registers then push a combined half2 into smem
vectorized. Remove `decode_ternary(int)→__int2half_rn(t)` scalar step.

## Pre-registered predictions

| Item | Prediction | Falsifier / close rule |
|------|-----------|----------------------|
| fwd B4 | −3..−6% fc1/fc2/head kernel time; step −1..−2% | win/loss judged per bench_protocol rule below |
| parity | bit-exact on all shapes (fc1, fc2, head) | FAIL if not bit-exact |
| pytest | 103 passed | any failure = FAIL |

## Decision rule (bench_protocol, mandatory)

All timings use `tests/bench_protocol.py::ab_median`: 10-trial paired
interleaved median, warmup 10, `torch.cuda._sleep` clock settle, A/A null arm
on the first shape for sigma_null. A variant counts as a win/loss only when
|Δ| > max(2*sigma_null, 1%). The old 3-trial min-of-3 protocol is retired.

## What this batch does NOT claim

- B4 is a fwd-only change; dX and update are untouched.
- Single-seed timing numbers support no convergence claim.
- The W decode fusion may affect occupancy; if occupancy drops, the gain
  may not materialize.

## Artefacts

`results/b4.<tag>.{json,log}` plus per-step logs on the VM, downloaded with
`colab download` before `colab stop`. rc: 0 = all gates pass, 1 = a gate
failed (valid result), 2 = commit-mismatch refusal (invalid; never commit
rc=2 artefacts).