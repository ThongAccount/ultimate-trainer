# Pre-registration: dX rasterization transpose (2026-11-02)

Branch `feat/dx-xpose`, committed **before any Colab allocation** and before
the per-branch CI bundle is published. Next item in the 2026-11-01 audit's
ranked backlog ("apply the same rasterization to dX; head dY re-stream
~26 GB; expect fc-range gains, measure").

## Change

`kernels/packed_ternary/gemm_backward_dx_tc.cu` — 3-line rasterization
transpose, mirroring feat/fwd-xpose:
- kernel prologue reads k from `blockIdx.x`, b from `blockIdx.y`
  (`super_k0 = blockIdx.x * kSuperN`, `super_b0 = blockIdx.y * kSuperM`);
- launcher grid becomes `(k_tiles, b_tiles)`:
  `dim3 grid((in_features + kSuperN-1)/kSuperN, (batch_size + kSuperM-1)/kSuperM)`.
  gridDim.y = b_tiles = 256 at B=16352 (512 at B=32768) < 65535 cap — safe.

Mechanism: with the k-tile fastest, co-resident CTAs share one b-tile and
therefore the same dY[b0:b0+64, :] slice (6.4 MB at head shape) instead of
each re-streaming it — at head, dY is [16352, 50272] fp16 ≈ 1.64 GB and was
read once per k-tile ≈ 16× ≈ 26 GB of DRAM traffic.

Only the CTA→tile map changes. The N-reduction (`r0` loop) is intra-CTA and
sequential per CTA, there are no cross-CTA atomics, and the dX kernel already
has the warp-private spill epilogue — so parity vs master must be
**bit-exact** (same argument as fwd-xpose; pytest parity gate enforces it).

## CI bundle

`build-kernels.yml` dispatched on this branch with input
`bundle_tag=kernels-dx-xpose` → release `kernels-dx-xpose`. VM runs with
`UT_PREBUILT_TAG=kernels-dx-xpose` so the branch kernel is measured.

## Validation session (one T4)

`scripts/colab_gate16_validate.py --branch feat/dx-xpose --expect-commit <sha>`:

1. pytest gate — green required, incl. dX parity tests (bit-exact).
2. `colab_perfgate.py` check vs measured baseline (`tests/perf_baseline_t4.json`,
   step 4810.507 ms). The dx kernel rows at B=16384 **isolate the xpose
   effect**: gate16 only changes the M=16352 dispatch (%64 already passed at
   16384), and the fwd spill/xpose changes do not touch dX. So:
   - ship bar: head/dx AND ≥1 of fc1/dx, fc2/dx win beyond max(2σ_null, 1%)
     vs baseline dx rows (649.306 / 51.214 / 50.677 ms) — direction and size
     unknown, this session measures them;
   - no-regression control: fwd and update rows unchanged within the bar;
   - `step_ms` at M=16352 recorded; expected to drop beyond the bar if the
     dY re-stream was real (3395.638 ms on master is the reference).
   Ship to master only if the bar holds.

## Not changed

Thresholds, baseline, protocol, other kernels, the 32×32 dX kernel.
rc=2 artefacts never committed.
