# Results: P0 measurement bundle (2026-11-02/03, ut-measure-p0 @ ab4dbeb0 + f2c7428-r2)

Prereg `2026-11-02-measure-p0-prereg.md`; script `scripts/colab_measure_p0.py`.
Artefacts `results/measure_p0.ab4dbeb0.*`, `results/clocks.*.csv`.

## #3 B=0 vs B=5 same-VM A/B (shipped arm) — ANSWERED
budget-ab: B=0 median **593.98 ms**, B=5 median **595.01 ms** → flip counting
costs **1.03 ms (0.17%)**, 1259 flips over 70 steps. (Custom arm measured
~11 ms / 1.9% — the shipped arm's counter epilogue is much cheaper.)

## #1 shipped-arm profiler attribution
shipped-profile (cublas_cublas, 100 profiled steps): median **608.3 ms**,
loss 11.05→9.20. Per-step device time (self, profiler-inflated): aten::mm
≈264 ms (26 GEMMs/step — the cuBLAS fwd/dX/dW), the update op ≈143 ms, cutlass
epilogue ≈57 ms, **aten::mul ≈43 ms/step = the 26 `* scale` passes** — the
scale fold (#5) is worth ~40 ms on the shipped arm, bigger than the audit's
35–45 ms estimate's lower bound. Step checkpoints 574→656→585: no clean
monotonic climb on a fresh VM.

## #2 empty_cache falsifier (shipped arm)
shipped-climb (200 steps, no intervention): median_first50 **631.2** vs
median_last50 **608.2** — the step got FASTER late on this VM; the historical
510→575 climb does not reproduce. The cache child (empty_cache @150) failed
to allocate twice (Colab 503) and was re-run separately — see artefact
`measure_p0.f2c7428-r2.json`. Custom arm already ruled out fragmentation
(2026-11-02). Conclusion: climb is VM/session-state variance, not allocator
fragmentation.

## #4 clock/power confound
`results/clocks.shipped-profile.csv` (1235 samples): SM clock 300 idle →
1590 max, active bursts 780–1515 MHz; power median 28.8 W; temp 43–83 °C. No
thermal throttle (83 °C << T4 limit), no sustained clock droop within the
run → the climb is not clock-driven on this VM.

## #9 custom-arm glue attribution

**CORRECTION 2026-11-20 — the original #9 numbers below are superseded.**
Re-derived from the raw artefact `results/measure_p0.ab4dbeb0.json` (child
`custom-glue`) by Researcher in the Masters ultimate-trainer debate and
independently re-verified by Chief against the same JSON.

Two arithmetic faults in the original #9:

1. **Wrong divisor.** Every per-step figure used ÷100, but the profile
   covered **40 steps** — proven by the launch counts: update/dX/fwd counts
   are 520 = 40 steps × 13 layers, and `aten::mul` count 1040 = 40 × the
   26 scale ops per step. (The prereg §5 also says "40 profiled steps".)
2. **Double-counted device sum.** `device_kernel_sum_ms` (262069.3) sums
   **both** the op rows (`packed_ternary::update_tc_v2`, `aten::mul`, …)
   **and** their CUDA-kernel rows, which carry identical `self_cuda_ms`.
   ÷40 it gives 6551.7 ms/step — larger than the whole 3266.7 ms step,
   which is impossible. The doc's "device kernel sum ≈2620 ms/step" is
   literally that double-counted sum ÷100, and "glue ≈ 640 ms" is the step
   minus it, so the glue conclusion falls with the arithmetic.

Corrected attribution (CUDA-kernel rows only, ÷40, step 3266.7 ms):

| Bucket | ms/step | % of step |
|---|---:|---:|
| `packed_ternary_update_tc_v2_64_kernel` | 1252.5 | 38.3% |
| `packed_ternary_backward_dx_tc_64_kernel` | 1111.6 | 34.0% |
| `packed_ternary_forward_tc_64_kernel` | 721.0 | 22.1% |
| **Custom kernels, total** | **3085.2** | **94.4%** |
| aten device kernels (glue: mul/add/softmax/LN/gelu/copy/…) | 167.2 | 5.1% |
| True host gap (step − all device time) | ~14.4 | ~0.4% |

Cross-checks: custom + aten = 3252.3 ms/step ≈ the 3266.7 ms step; wall
40 × 3266.7 ms = 130.7 s ≈ `cudaDeviceSynchronize` self_cpu 130.4 s — the
host is waiting on a saturated GPU, not dispatching. `aten::mul` alone is
43.8 ms/step (1753.515 ÷ 40), not 17.5.

**Conclusion: the custom arm is kernel-bound (~94% custom kernels), not
glue-bound.** Glue elimination (CUDA Graph / batched dispatch) on this arm
addresses only the ~0.4% host gap plus part of the ~5% aten device time;
the large levers are the update (38%) and dX (34%) kernels. The audit's
1196 ms glue estimate and this section's "~640 ms" are both superseded, as
is the later "~94% glue" reading (2026-11-20 i3, single-layer kernels
subtracted from a full-model step — see the correction note on
`feat/i3-glue-attribution`). The shipped cuBLAS arm stays production
regardless (~519–608 ms vs 2976–3267 ms custom).

Original #9 (superseded, kept for provenance):

> custom-glue (custom arm, 40 profiled steps): median step **3266.7 ms**,
> device kernel sum ≈2620 ms/step → **glue ≈ 640 ms** (embed/norms/gelu/
> 26×scale/logits+CE/optimizer; aten::mul only ≈17.5 ms of it). Kernels:
> update 501, dX 444, fwd 288 ms/step. The audit's 1196 ms glue estimate was
> pre-gate16/xpose at 4810 ms; now ~640 ms at 3267 ms.

## #18 fc1 fwd conflict
fc1fwd child: first attempt failed on import (`tests` not a package — fixed,
load by path @ f2c7428); re-run artefact `measure_p0.f2c7428-r2.json`.
