# Pre-registration: fwd-xpose wiring (2026-11-02)

Branch `feat/fwd-xpose`, committed **before any Colab allocation** and before
the per-branch CI bundle is published.

## Change

`kernels/packed_ternary/gemm_forward_tc.cu` — 3-line rasterization transpose:
kernel prologue reads n from `blockIdx.x`, b from `blockIdx.y`; launcher grid
becomes `(n_tiles, m_tiles)`. Only the CTA→tile map changes; inner math is
byte-identical, so parity vs master must be **bit-exact**
(`tests/probe_fwd_swizzle.py` measured both formulations bit-exact on all
shapes incl. edges).

Measured probe (ut-c13-probes3, `results/probe_batch.ee9cefe6.log`):
head fwd **−19.90%**, fc2 −2.26%, fc1 −1.20% (xpose formulation).

## CI bundle

`build-kernels.yml` dispatched on this branch with input
`bundle_tag=kernels-fwd-xpose` → release `kernels-fwd-xpose`. VM runs with
`UT_PREBUILT_TAG=kernels-fwd-xpose` so the branch kernel is measured, never
the stale master `.so`.

## Validation session (one T4)

1. pytest gate (`colab_pytest.TEST_MODULES`) — green required, incl.
   `test_gemm_forward.py` parity.
2. `colab_perfgate.py` check vs measured baseline (`tests/perf_baseline_t4.json`
   @ ee9cefe, step 4810.507 ms):
   - fwd kernel rows at B=16384: head expected **~−19.9%** (499.841 → ~400 ms),
     fc ~−1…2%, beyond the max(2σ_null, 1%) bar; dx/update rows must be
     unchanged within the bar (no-regression control);
   - `step_ms` expected ≈ **−108 ms (~−2.2%)** → ~4700 ms, beyond the bar.
   Ship to master only if all hold.

## Not changed

Thresholds, baseline, protocol, other kernels. rc=2 artefacts never committed.
