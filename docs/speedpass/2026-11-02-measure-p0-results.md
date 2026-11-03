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
custom-glue (custom arm, 40 profiled steps): median step **3266.7 ms**,
device kernel sum ≈2620 ms/step → **glue ≈ 640 ms** (embed/norms/gelu/
26×scale/logits+CE/optimizer; aten::mul only ≈17.5 ms of it). Kernels:
update 501, dX 444, fwd 288 ms/step. The audit's 1196 ms glue estimate was
pre-gate16/xpose at 4810 ms; now ~640 ms at 3267 ms.

## #18 fc1 fwd conflict
fc1fwd child: first attempt failed on import (`tests` not a package — fixed,
load by path @ f2c7428); re-run artefact `measure_p0.f2c7428-r2.json`.
