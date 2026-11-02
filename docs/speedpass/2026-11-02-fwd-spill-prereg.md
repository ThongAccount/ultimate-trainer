# Pre-registration: fwd-spill wiring (2026-11-02)

Branch `feat/fwd-spill`, committed **before any Colab allocation** and before
the per-branch CI bundle is published.

## Change

`kernels/packed_ternary/gemm_forward_tc.cu` — epilogue only: the cooperative
shared-memory→global store is replaced by a **warp-private spill** buffer
(`__shared__ float spill[kWarps][kWMMA_M*kWMMA_N]`, 4 × 4 KB). Each warp
stores its own WMMA fragment into its own slot
(`wmma::store_matrix_sync(&spill[warp_id][0], ...)`), then copies its own
16×16 tile to global with `__syncwarp()` between fragments instead of CTA
`__syncthreads()`. Ported from `tests/fwd_spill_warpprivate.cu` onto the
xposed kernel (post-`feat/fwd-xpose` rasterization).

No cross-warp data flow exists in either epilogue form — same float values go
to the same Y addresses — so parity vs master must be **bit-exact**
(`tests/probe_fwd_spill.py` measured both forms bit-exact on all shapes incl.
edges; probe run ut-c13-probes).

Measured probe (ut-c13-probes, `results/probe_batch.*.log` on
`feat/c13-probe-batch` @ 71f6a55): fwd **fc1 −14.36%, fc2 −13.97%, head
−7.00%** at production shapes.

## CI bundle

`build-kernels.yml` dispatched on this branch with input
`bundle_tag=kernels-fwd-spill` → release `kernels-fwd-spill`. VM runs with
`UT_PREBUILT_TAG=kernels-fwd-spill` so the branch kernel is measured, never
the stale master `.so`.

## Validation session (one T4)

1. pytest gate (`colab_pytest.TEST_MODULES`) — green required, incl.
   `test_gemm_forward.py` parity (bit-exact vs reference).
2. `colab_perfgate.py` check vs measured baseline (`tests/perf_baseline_t4.json`
   @ master, step 4810.507 ms). Note: this branch already contains gate16 +
   xpose, so fwd rows vs that baseline combine all three effects:
   - fwd fc1/fc2 expected ≈ **−15% combined** (xpose ~−1…2% + spill ~−14%),
     head ≈ **−26% combined** (xpose −19.9% + spill −7%), each beyond the
     max(2σ_null, 1%) bar;
   - dx/update rows must be unchanged within the bar (no-regression control);
   - `step_ms` expected ≈ **3410 ms** (post-xpose measured 3519.2, spill
     removes ~7–14% of the fwd share) — i.e. −29% vs the 4810.507 baseline.
   Ship to master only if all hold.

## Not changed

Thresholds, baseline, protocol, other kernels, the xpose rasterization.
rc=2 artefacts never committed.
