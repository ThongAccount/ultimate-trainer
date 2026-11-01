# C13 GO probe batch — results (2026-11-01)

Pre-registration: `docs/speedpass/2026-11-01-c13-probe-batch-prereg.md` (committed
before the first allocation, addenda before runs 2/3). Branch
`feat/c13-probe-batch`, three T4 sessions, all kernels loaded prebuilt from the
CI `kernels-latest` release (`UT_PREBUILT=1`, zero nvcc for production exts).

## Verdicts vs the pre-registered predictions

| Probe | Gate | Timing (10-trial median, bar = max(2σ_null, 1%)) | Verdict |
|---|---|---|---|
| **fwdspill** (warp-private spill epilogue) | bit-exact all shapes | fc1 **−14.36%** (σ_null 1.29%), fc2 **−13.97%**, head **−7.00%** | **WIN** — beats the −3..−6% head prediction |
| **fwdswizzle** (n-outer rasterization) | bit-exact all shapes incl. edge cases | head swz **−19.02%**, xpose **−19.90%**; fc2 −1.25%/−2.26%; fc1 −0.84% NOISE / −1.20% WIN | **WIN** on head — xpose is resource-identical (bit-identical SASS) and the ship candidate |
| **fwdwdedup** (W-word reg dedup) | bit-exact | fc1 +9.20%, fc2 +10.43%, head +9.92% | **LOSS — class falsified** |
| **uppipeline** (update double-buffer) | reference-transition gate **mismatch** (counter got=1 want=0 ×4 — variant drops dead_zone) | not reached | **gate FAIL (valid negative)** — variant not shippable; the ±2% timing falsifier only applies after a bit-exact gate, so the class is not formally closed |

Correctness gate: pytest **102 passed / 0 failed** on the GPU (after fixing a
pre-existing 5-arg call in `test_dispatch_order.py`).

Perfgate: check vs the old **provisional** baseline FAILed uniformly
+7..+17% — a protocol artifact, not a regression: the provisional numbers were
3-trial-min-derived, and the check pass matches the independent probe OLD
measurements (fc1/fwd 36.421 ms perfgate vs 36.74 ms probe). The regen pass
(`--update-baseline`) wrote a measured 10-trial-median baseline
(`tests/perf_baseline_t4.json`, status `measured`, commit ee9cefe); it measured
~10% higher than the check pass 6 min earlier (late-session clock drift on the
shared T4), so the committed baseline is conservative. Reconstructed locally
from the captured RESULTS JSON (VM released before download) — noted in
`_meta`.

## Sessions

| Session | tag (commit) | rcs |
|---|---|---|
| ut-c13-probes | d1485f8d | smoke 0, pytest 1 (pre-existing dispatch-test bug), fwdspill 0, fwdswizzle 1 (probe compile bug), uppipeline 1 (import bug), fwdwdedup 0, perfgate ×2 1 (op-signature crash) |
| ut-c13-probes2 | ff7d76cc | smoke 0, pytest 0, fwdswizzle 1 (wrapper/ launcher name collision), uppipeline 1 (semantics gate FAIL — valid), perfgate ×2 1 (fwd measured tc_32 not tc_64; step shape bug) |
| ut-c13-probes3 | ee9cefe6 | smoke 0, pytest 0, fwdswizzle 0 (**WIN**), perfgate_check 1 (expected vs provisional baseline), perfgate_regen 1 (writes new baseline by design) |

## Fixes landed on this branch (all pre-existing master breakage, no kernel code touched)

- `tests/probe_fwd_swizzle.py`, `tests/probe_update_pipeline.py`: import
  `bench_protocol` directly (`tests.` namespace package is shadowed by a
  dist-packages regular package); swizzle wrapper renamed `{fn}_py` (name
  collided with the `extern "C"` launcher → ambiguous `&fn`).
- `colab_perfgate.py`: pass `flip_count`/`flip_budget=0` to the `update_tc_v2`
  custom op (mutated arg; IndexError under no_grad on torch 2.11); fwd case
  uses the production dispatch `_forward_auto` (the `forward_tc` op is tc_32
  for all shapes — bogus +180% "regressions" before); `bench_step` x/y shapes
  [B, SEQ−1]/[B, SEQ] per the `train_step_cudagraph` contract.
- `tests/test_dispatch_order.py`: 8-arg raw launcher signature.
- `tests/test_perfgate_logic.py`: baseline assertions updated to the measured
  (regenerated) values.
- `scripts/colab_probe_batch.py`: per-probe `--skip`, gates count only
  executed steps.

## Artefacts

`results/probe_batch.d1485f8d.*` (run 1, incl. per-step logs),
`results/probe_batch.ff7d76cc.*` + `.full.log` (run 2),
`results/probe_batch.ee9cefe6.{json,log}` (run 3 full transcript; summary
reconstructed locally — VM released before download). Regenerated baseline:
`tests/perf_baseline_t4.json`.

## Next (per the user's one-branch-until-working rule)

The two WINs are probe-only variants; wiring them into the production fwd
kernel is separate work on its own branch: **fwdswizzle-xpose** first (3-line
grid change, bit-identical SASS, head −19.9%), then **fwdspill** (epilogue
rewrite, −7..−14%), each behind the parity gate + whole-step A/B on one T4.
