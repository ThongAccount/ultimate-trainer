# Pre-registration: C13 GO probe batch (2026-11-01)

Committed on `feat/c13-probe-batch` **before any Colab allocation**. Thresholds
and decision rules below are binding; they are not changed after measurement.

## Scope

One T4 session runs `scripts/colab_probe_batch.py`, which clones the branch at
`--expect-commit` and executes, in this fixed order:

0. `smoke` — `tests/probe_fwd_spill.py --probe fwdspill --skip-timing` (parity-only plumbing check; abort on failure)
1. `pytest` — correctness gate, `colab_pytest.TEST_MODULES` (11 modules)
2. `fwdspill` — `tests/probe_fwd_spill.py` (GO: fwd warp-private spill epilogue)
3. `fwdswizzle` — `tests/probe_fwd_swizzle.py` (GO: fwd n-outer rasterization swizzle)
4. `uppipeline` — `tests/probe_update_pipeline.py` (GO: update double-buffer)
5. `fwdwdedup` — `tests/probe_fwd_wdedup.py` (GO: fwd W-word register dedup)
6. `perfgate_check` — `colab_perfgate.py` vs committed `tests/perf_baseline_t4.json`
7. `perfgate_regen` — `colab_perfgate.py --update-baseline` (baseline regen under the current protocol)

No production file is modified. Every probe compiles its variant side-by-side
with the production kernel in one process and **must pass a bit-exact parity
gate** before timing; a probe that fails parity is a FAIL, not a result.
Production kernels load from the CI-prebuilt `kernels-latest` bundle
(`UT_PREBUILT=1` expected in provenance); only the small variant `.cu` files
JIT on the VM.

## Pre-registered predictions (from HANDOFF §26 GO queue + probe docstrings)

| Step | Prediction | Falsifier / close rule |
|---|---|---|
| fwdspill | smem 20480→8192 B, occ 3→5 CTAs/SM; head −3..−6%, e2e +3–5% if barrier-drain holds | win/loss judged per bench_protocol rule below |
| fwdswizzle | X DRAM 25.7 GB→~32 MB per head step; e2e ≤ +2.5%; ship head-only if fc regresses | same decision rule |
| uppipeline | overlap wins: fc1 −10..−20%; occupancy loss: +5..10% slower | **fc1 delta within ±2% or slower → pipelining class CLOSED permanently** |
| fwdwdedup | part of the fwd instruction bundle; predicted 1–3% e2e for the full bundle | same decision rule |
| perfgate_check | PASS expected (kernel code untouched since baseline) | WARN ≥ 2%, FAIL ≥ 5% regression vs committed baseline, rc=1 on FAIL |

## Decision rule (bench_protocol, mandatory C13 protocol)

All timings use `tests/bench_protocol.py::ab_median`: 10-trial paired
interleaved median, warmup 10, `torch.cuda._sleep` clock settle, A/A null arm
on the first shape for sigma_null. A variant counts as a win/loss only when
|Δ| > max(2*sigma_null, 1%). The old 3-trial min-of-3 protocol is retired.

## What this batch does NOT claim

- The fwd instruction bundle is only partially implemented (the W-word dedup
  variant ships as `tests/fwd_wdedup.cu`; branchless decode and half2 X/dY
  loads are not in this batch).
- A probe win does not by itself ship: wiring a variant into production
  dispatch is a separate change with its own branch, parity gate, and
  whole-step A/B.
- Single-seed timing numbers support no convergence claim (banked lesson:
  single-seed runs can't support claims).

## Artefacts

`results/probe_batch.<tag>.{json,log}` plus per-step
`results/probe_batch.<tag>.<step>.log` on the VM, downloaded with
`colab download` before `colab stop`. rc: 0 = all gates pass, 1 = a gate
failed (valid result), 2 = commit-mismatch refusal (invalid; never commit
rc=2 artefacts).
