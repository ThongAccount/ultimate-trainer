# Pre-registration: head dW GEMM algo autotune (2026-11-03)

Branch `feat/dw-algo`, committed **before any Colab allocation** and before
the CI bundle. Researcher list A item #7 (audit §2: "head dW GEMM autotune —
80.7 ms dW vs ~51 ms fwd for equal FLOPs → up to ~50 ms").

## Change

`kernels/packed_ternary/gemm_update_cublas.cu`: `packed_ternary_dw_gemm_f32`
gains an `int algo` parameter; `algo < 0` keeps `CUBLAS_GEMM_DEFAULT`
(byte-identical behavior), `algo >= 0` pins an explicit `cublasGemmAlgo_t`.
`update_cublas.py` reads `PT_DW_ALGO` (default -1) and passes it through.
Compute type stays `CUBLAS_COMPUTE_32F`, fp16 in / fp32 out — unchanged.

## Validation (one T4)

`scripts/colab_dw_algo_sweep.py`:
1. pytest gate (colab_pytest.TEST_MODULES) green.
2. Sweep algos {-1, 0..23} on the head shape (B=16352, N=50272, K=1024) and
   an fc shape (B=16352, N=4096, K=1024): 10 interleaved trials each vs the
   default, ab_median-style; per-algo dW parity vs default: max |Δ|, fp32
   sign-disagreement rate (audit: summation order changes → sign-disagreement
   gate, NOT bit-exact).
3. Step A/B same VM, cublas_cublas arm, PT_DW_ALGO=best vs unset, 60 steps:
   ship bar = step win beyond max(2σ_null, 1%) AND sign-disagreement rate
   within the pre-registered bound (≤ 1e-4 of elements) AND flip trajectory
   matches master for 60 steps.
Ship to master only if all hold; the chosen algo becomes the new default.
