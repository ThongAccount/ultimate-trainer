# Pre-registration: scale fold into decoded W16 (2026-11-02/03)

Branch `feat/scale-fold`, committed **before any Colab allocation**.
Researcher list A item #5. Measurement basis: `results/measure_p0.ab4dbeb0.shipped-profile.log`
— aten::mul ≈ 43 ms/step on the shipped arm (26 `* scale` passes).

## Change (cublas arm only, opt-in via PT_FOLD_SCALE)

- `kernels/packed_ternary/cublas_backend.py`: when `PT_FOLD_SCALE` is set,
  fwd and dX scale the decoded W16 by it after `decode_into`:
  `W16 = W16 * s`. s = K^-0.5 = 2^-5, a power of two → the fp16 scale is an
  exact exponent shift; fp32 GEMM products X·(sW) = s·(X·W) exactly, so the
  folded forward is **bit-exact vs the post-mul outside subnormals** (quantify
  in the parity probe below).
- `train_gigatoken.train_step_cudagraph`: skips its 13 `* scale` muls when
  `fold_scale()` is on; raises if PT_FOLD_SCALE ≠ K**-0.5 (mismatch guard).
- `kernels/packed_ternary/packed_linear.py` backward: the update's
  dW = dY^T @ X still needs the factor → applied to X (`X_upd = X * s`,
  smaller than dY); fused path skipped under fold (its dX uses unscaled W).
  dX itself is already correct via the scaled W16 in backward_dx.
- Default off: `PT_FOLD_SCALE` unset → byte-identical behavior to master.
  Graph path (train_step_graph_v2) untouched — do not set the env with it.

## Validation (one T4, cublas_cublas arm)

1. Parity probe: tiny W/X, fwd with and without fold (monkeypatched
   `_FOLD_SCALE`), assert max |Δ| == 0 outside subnormals and count
   subnormal diffs; plus a full-step parity at production shape: loss and
   flip counts identical to master arm for 50 steps (flip trajectory is the
   bit-stable gate).
2. Step A/B same VM: master (PT_FOLD_SCALE unset) vs branch
   (PT_FOLD_SCALE=0.03125), interleaved 60 steps after warmup, ab_median
   protocol, bar max(2σ_null, 1%). Expected: −25…−40 ms (~4–7%) from removing
   the fwd muls (13× activation traffic ≈ 2.4 GB) minus the X_upd muls.
3. pytest gate (colab_pytest.TEST_MODULES) green incl. new fold tests.
Ship to master only if 1+2+3 hold; otherwise record and revert.
