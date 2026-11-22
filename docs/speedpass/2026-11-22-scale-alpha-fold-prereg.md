# Pre-registration: scale elimination via GEMM-alpha / decode fold (2026-11-22)

Committed on `feat/scale-alpha-fold` **before any Colab allocation**.
Thresholds and decision rules below are binding; they are not changed after
measurement.

## Scope

One T4 session runs `scripts/colab_scale_alpha_validate.py`, which clones the
branch at `--expect-commit` and executes, in this fixed order:

0. `smoke` — `tests/probe_fwd_spill.py --probe fwdspill --skip-timing`
   (parity-only plumbing check; abort on failure)
1. `pytest` — correctness gate, `colab_pytest.TEST_MODULES`
2. `scale_alpha_parity` — bit-exact parity of the scale-alpha variant vs
   production kernel (fc1, fc2, head; counters past threshold so flips
   actually occur)
3. `scale_alpha_ab` — same-VM whole-step A/B, 10-trial paired median vs
   measured shipped baseline 627.6 ms

No production file is modified outside the scale-alpha change. Every probe
compiles its variant side-by-side with the production kernel in one process
and **must pass a bit-exact parity gate** before timing; a probe that fails
parity is a FAIL, not a result.

## Hypothesis

Scale elimination via GEMM-alpha / decode fold: the 26 `* scale` passes cost
~43 ms/step (~7%). The naive W16 fold already failed (+361%), but folding the
constant 2⁻⁵ into the GEMM alpha or the decode epilogue is a *different
mechanism*, untested.

## Pre-registered predictions

| Item | Prediction | Falsifier / close rule |
|------|-----------|----------------------|
| scale-alpha fold | ~43 ms saved on shipped step | win/loss judged per bench_protocol rule below |
| parity | bit-exact on all shapes (fc1, fc2, head) | FAIL if not bit-exact |
| pytest | 103 passed | any failure = FAIL |
| subnormal fraction | fp16 subnormals < 1% of outputs | if > 1%, scale fold is not bit-exact |

## Decision rule (bench_protocol, mandatory)

All timings use `tests/bench_protocol.py::ab_median`: 10-trial paired
interleaved median, warmup 10, `torch.cuda._sleep` clock settle, A/A null arm
on the first shape for sigma_null. A variant counts as a win/loss only when
|Δ| > max(2*sigma_null, 1%). The old 3-trial min-of-3 protocol is retired.

## What this batch does NOT claim

- Scale-alpha fold is NOT the naive W16 fold (which failed +361%).
- Single-seed timing numbers support no convergence claim.
- The scale fold may affect fp16 subnormals; if subnormal fraction > 1%,
  the fold is not bit-exact.

## Artefacts

`results/scale_alpha.<tag>.{json,log}` plus per-step logs on the VM,
downloaded with `colab download` before `colab stop`. rc: 0 = all gates
pass, 1 = a gate failed (valid result), 2 = commit-mismatch refusal
(invalid; never commit rc=2 artefacts).