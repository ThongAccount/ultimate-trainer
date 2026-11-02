# Pre-registration: P0 measurement bundle (2026-11-02)

Script `scripts/colab_measure_p0.py` on master, committed **before any Colab
allocation**. Measurement only — no kernel or dispatch code changes.
Researcher list A items #1–#4, #9, #18, one T4 session, six subprocess
children (backend read at import time, so each child gets its own env):

1. `shipped-profile` (cublas_cublas, 100 profiled steps after warmup):
   torch.profiler CPU+CUDA+memory — attribute the 510→575 ms within-run climb
   (`results/official_suite.B05prod300.log`) and the device-kernel-sum vs
   wall-step gap on the shipped arm. **Item #1.**
2. `shipped-climb` (cublas_cublas, 200 steps, no intervention) vs
   `shipped-climb-cache` (same + `torch.cuda.empty_cache()` at step 150):
   fragmentation falsifier on the SHIPPED arm (custom arm already ruled out,
   `2026-11-02-hyperopt-all3-results.md` item 2). **Item #2.** Prediction:
   if allocator fragmentation caused the climb, the cache arm drops to the
   step-1 rate after 150; otherwise climbs match.
3. `budget-ab` (cublas_cublas, same VM, models with FLIP_BUDGET 0 vs 5,
   interleaved per-step, 60 steps after warmup): price of flip counting on
   the shipped path. Bytes say ~1.6 ms; custom arm measured ~11 ms (1.9%).
   **Item #3.**
4. nvidia-smi clock/power/temp sampler at 500 ms in every child
   (`results/clocks.*.csv`): clock/power confound check. **Item #4.**
5. `custom-glue` (custom arm, 40 profiled steps): attribute the ~1.2 s
   non-kernel glue (embed/norm/gelu/26×scale/logits+CE/optimizer). **Item #9.**
6. `fc1fwd` (custom arm, ab_median A/A null + absolute median, B=16384,
   fc1 forward TC kernel): resolve the 4.15 vs 7.33 ms historical conflict
   under one protocol. **Item #18.**

Success = artefact `results/measure_p0.<tag>.json` with all six children's
RESULTS_JSON + provenance; rc=2 commit-mismatch artefacts never committed.
Interpretation thresholds unchanged (ab_median bar max(2σ_null, 1%)); this
session only measures — no ship decision rides on it except confirming or
killing the fragmentation hypothesis for #2.
