# Results: I3 — custom-arm glue attribution (2026-11-20) — CORRECTED

> **CORRECTION (2026-11-20): the "94% glue" conclusion below is REJECTED.**
> The profile measured one isolated layer and subtracted it from the
> full-model step — different workloads. The verified same-workload
> attribution (from `results/measure_p0.ab4dbeb0.json`) is: **custom kernels
> 3085.2 ms/step = 94.4% of the step; the custom arm is kernel-bound, not
> glue-bound.** See "Correction" at the bottom. The original text is kept
> unchanged for provenance; "Key finding" and "Recommendation" are superseded.

Branch `feat/i3-glue-attribution` @ e60b0d4, prereg
`2026-11-20-i3-glue-attribution-prereg.md`, session ut-i3.
Artefact: torch profiler output (inline below).

## Verdicts vs the pre-registered predictions

| Item | Prediction | Measured | Verdict |
|------|-----------|----------|---------|
| glue attribution | custom arm ~1.2 s of non-kernel time | kernels = 184.4 ms/step; step = 2976.4 ms → glue ≈ 2792 ms | CONFIRMED — glue dominates |
| pytest | 103 passed | 103 passed in 286.57s | PASS |

## Profiler output (10 steps, custom arm)

```
                             packed_ternary::forward_tc   930.325ms  50.46%  93.033ms/call
                         packed_ternary::update_tc_v2    481.744ms  26.13%  48.174ms/call
                       packed_ternary::backward_dx_tc     431.450ms  23.40%  43.145ms/call
```

Total kernel time for 1 fwd + 1 dX + 1 update: **184.4 ms/step**.
Full custom arm step: **2976.4 ms**.
Glue (Python dispatch, tensor movement, etc.): **~2792 ms/step** (~94%).

## Key finding (superseded — see Correction)

The custom arm is **glue-bound**, not kernel-bound. The kernels themselves
are only ~6% of the step time. The remaining ~94% is Python dispatch,
tensor movement, and other non-kernel overhead.

**Implication**: Further kernel tuning on the custom arm has limited
upside. The optimization opportunity is in reducing Python dispatch
overhead (e.g., CUDAGraph capture, batched dispatch, or switching to the
shipped cuBLAS path which eliminates the custom glue entirely).

## Recommendation (superseded — see Correction)

The custom arm's 4.7× slowdown vs the shipped cuBLAS path (627.6 ms) is
primarily due to Python dispatch overhead, not kernel inefficiency. The
shipped path (cublas_cublas) is the correct production choice. Further
custom-arm kernel work should be deprioritized in favor of the shipped path.

## Correction (2026-11-20) — custom arm is kernel-bound, not glue-bound

**What was wrong.** `scripts/colab_i3_profile.py` profiles **one isolated
layer** (B=16352, K=1024, N=4096; exactly 1 `forward_tc` + 1 `backward_dx_tc`
+ 1 `update_tc_v2` per step, 10 steps) but the doc subtracts that
single-layer kernel time (184.4 ms) from the **full-model** perfgate step
(2976.4 ms, all layers + embeddings + norms + logits + CE + optimizer).
Numerator and denominator are different workloads; the "~2792 ms glue (~94%)"
is mostly the other layers' kernel time, not host overhead. The profiler
table itself shows the tell: 93.033 ms/call over 10 steps = 1 call/step,
while the real model runs 13 packed-ternary layers per step.

**Verified attribution** — `results/measure_p0.ab4dbeb0.json`, child
`custom-glue`, 40 profiled steps, median step 3266.7 ms, op rows deduplicated
against their CUDA kernel rows (they carry identical `self_cuda_ms`):

| Bucket | ms/step | % of step |
|--------|--------:|----------:|
| custom kernels — update 1252.5 + dX 1111.6 + fwd 721.0 | **3085.2** | **94.4%** |
| aten device kernels (mul/add/softmax/layernorm/gelu/copies) | ≈167.2 | ≈5.1% |
| true host gap (step − all device time) | ≈14.4 | ≈0.4% |

Evidence (recomputed from the raw artefact):
- Launch counts 520 = 40 steps × 13 layers for each of update/dX/fwd;
  `aten::mul` 1040 = 40 × 26 scale passes.
- `device_kernel_sum_ms` = 262069.3 double-counts op + kernel rows:
  /40 = 6551.7 ms/step > the step itself (impossible). The earlier
  measure-p0 doc line "device kernel sum ≈2620 ms" is that sum **/100**;
  its "glue ≈640 ms" line is rejected by the same arithmetic (corrected in
  `2026-11-02-measure-p0-results.md` separately).
- `cudaDeviceSynchronize` self_cpu = 130.4 s ≈ 40 × 3266.7 ms = 130.7 s
  wall — the host is waiting on a saturated GPU, not dispatching.

**Revised implication.** The custom arm's cost *is* its three hand-written
kernels (update ≈38%, dX ≈34%, fwd ≈22% of the step). The shipped
cuBLAS arm stays production (~519–608 ms vs ~2976–3267 ms) because it
replaced exactly those kernels with cuBLAS at ~75% of T4 peak — that
whole-step A/B, not this doc's original "glue-bound" premise, is the sound
basis for directing remaining effort to the shipped arm. A CUDA
Graph/glue-elimination spike on the custom arm buys ≈0.4% — deprioritized.
If the custom arm must remain the bit-exact reference, kernel-work targets
are update and dX, not store-path micro-tweaks; A2/B2/B4/A3 stay closed on
their own measured step regressions. No new Colab profile needed — the
reconciliation is answered from the existing artefact.