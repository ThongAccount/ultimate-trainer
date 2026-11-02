# Pre-registration: dX `%16` gate wiring validation (2026-11-02)

Branch `feat/dx-gate-16`, committed **before any Colab allocation**.

## Change

`kernels/packed_ternary/custom_ops.py` `backward_dx_tc`: the 64×64 dX gate
`B % 64 == 0` → `B % 16 == 0` (N % 16 and K % 64 unchanged). Pure-Python
dispatch change — no `.cu` touched, so the CI prebuilt bundle stays valid
(`UT_PREBUILT=1` expected). Motivation and measurement:
`docs/speedpass/2026-11-02-dxgate-shipped-results.md` (dX-64 bit-exact and
−40…−46% vs 32×32 at M=16352/16384 on fc1/fc2/head).

New regression test: `test_dx_64_kernel_used_at_real_step_M16352` asserts the
op takes the 64×64 kernel at M=16352 and matches the raw fn bit-exactly.

## Validation session (`scripts/colab_gate16_validate.py`, one T4)

1. pytest gate (`colab_pytest.TEST_MODULES`, 12 modules incl. the new test) —
   must be fully green.
2. `colab_perfgate.py` check vs the measured baseline
   (`tests/perf_baseline_t4.json` @ ee9cefe):
   - kernel rows at B=16384 must be **unchanged** within the
     max(2σ_null, 1%) bar (the %64 gate already passed there — this is the
     no-regression control);
   - `step_ms` (real step, M=16352) must improve beyond the bar vs
     4810.507 ms. Pre-registered expectation from the probe: **−11…−12%**
     (dX ≈ 1261 ms of the step at 41–46% saving ≈ 520–580 ms). Ship to
     master only if both hold; otherwise revert.

## Not changed

Thresholds, baseline file, kernel code, protocol. rc=2 artefacts never
committed.
