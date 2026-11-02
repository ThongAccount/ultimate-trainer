# Pre-registration: dX-gate + batch-scale + shipped-path profile (2026-11-02)

Committed on `feat/dxgate-shipped-profile` **before any Colab allocation**.
Thresholds and decision rules below are binding.

Implements hyperopt-audit items 1 and 2 (`chore/hyperopt-audit` @ c117d9f) plus
the user's "increase multiplication batches" ask as a batch-scaling probe.

## Session contents (one T4, `scripts/colab_dxgate_batch.py`)

0. `smoke` — `tests/probe_dx_gate.py --parity-only` (dX 64 vs 32 bit-exact at
   M=16352/16384; abort on mismatch)
1. `dxgate` — `tests/probe_dx_gate.py` full:
   - **Q1 gate A/B**: dX 64×64 vs 32×32 at M ∈ {16352, 16384} × {fc1, fc2, head},
     bit-exact parity first, then `ab_median` 10-trial interleaved, null arm on
     fc1/M=16352. Decision bar |Δ| > max(2σ_null, 1%).
     **Wiring rule (pre-registered):** if dX-64 is faster beyond the bar at
     M=16352 on fc1 AND head, relax the `B % 64 == 0` gate in
     `backward_dx_tc` (custom_ops.py) to `% 16` on branch `feat/dx-gate-16`
     (pure-Python dispatch change; prebuilt bundle stays valid; the 64×64
     kernel has internal bounds guards — same argument as the update v2_64
     %16 gate). If not faster or mixed, gate stays; result documented.
   - **Q2 batch scale**: `solo_median` of fwd tc_64 / dX-64 / update v2_64 at
     M ∈ {16352, 16384, 32768} (M=32768 = B 64×512). Report ms per M; no
     threshold — this is a scaling curve, reported as-is. Memory guard: T4
     16 GB, shapes sized to fit (head W 12.9 MB int32, X/dY 67 MB fp16 at
     M=32768).
2. `shipped_profile` — `scripts/colab_shipped_profile.py` with
   PT_BACKEND=cublas PT_UPDATE_BACKEND=cublas, synthetic data, 300 steps,
   arms B=0 and B=5 (PT_FLIP_BUDGET), per-step wall ms, `empty_cache()` at
   step 150 of the B=5 arm, `torch.profiler` over steps 100–200 for kernel
   attribution. Answers: what drives the 510→575 ms within-run climb
   (B05prod300), and how much flip counting costs (B=0 vs B=5).

## Constraints

- Production code untouched in this batch (probe measures raw kernels
  directly; profiler is measurement-only). The only candidate change is the
  one-line gate relaxation, which lands on its OWN branch afterward, behind
  parity + whole-step A/B vs the measured 4810.507 ms baseline.
- Prebuilt kernels (`kernels-latest`) load for all steps: no kernel .cu is
  modified on this branch, so `UT_PREBUILT=1` is safe and expected.
- rc semantics: 0 = all steps ran (win/loss is a RESULT inside the logs),
  1 = gate failed, 2 = commit-mismatch (invalid; never committed).

## Artefacts

`results/dxgate_batch.<tag>.{json,log}` + per-step `.log` on the VM,
downloaded before `colab stop`.
