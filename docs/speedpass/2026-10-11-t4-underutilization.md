# T4 under-utilization campaign — 2026-10-11

Environment: Colab free-tier T4 (sm_75, 40 SMs, 65 TFLOPS FP16 peak,
~25-35 realistic at the 70W cap; sm_75 HAS LDGSTS/cp.async, no bf16,
64KB smem/SM). All numbers from `colab run --gpu T4` one-shot runners;
commit quoted per row; torch 2.11.0+cu130.

## Notes that contradict or update the brief

1. **`ncu` unavailable**: hosted VMs return ERR_NVGPUCTRPERM for hardware
   counters. Fallback per brief: CUDA events (median of >=20 after warmup 10),
   torch.profiler for kernel names/times, `nvcc -arch=sm_75 -Xptxas -v` for
   resources. Phase 3 acceptance therefore rests on measured >=1.15x speedups,
   never on reasoning.
2. **Brief 1.2 `_load_dx_tc` undefined — STALE**: it is defined
   (`pack_update.py:249`) and dX already had per-tile split pointers
   (`_dx_tc_fn` vs `_dx_tc_32_fn`). The real load-order defect was on the
   update side: the shared `_up_tc_v2_fn` was overwritten by the 32x32 build
   whenever `_load_tc_if_needed()` ran, so large-shape calls silently ran the
   small kernel depending on call order. Fixed + regression-tested; the dX
   claim is recorded as already-correct (no change needed).
3. **`unpack_tensor()` is a CPU Python loop** (triple-nested, checkpoint-only
   per its own docstring) — unusable for the cuBLAS arm (fc1 = 4.2M host
   iterations). Contradicts the brief's implicit assumption that a decode
   primitive exists. Wrote one: `kernels/packed_ternary/pack_decode.py`
   (GPU, one thread per element) — doubles as the Phase 2 decode-once primitive.
4. **`colab run` takes a script path only** — `colab run ... bash -c '...'`
   fails with "Script not found: bash". All setup lives inside the Python
   entry (`scripts/colab_bench.py`), not a shell wrapper.
5. **Fresh clones land on `master`, not the work branch.** Pin the branch and
   reset before measuring, or you silently benchmark a stale commit (this bit
   once: a run measured commit 28d2bab while the work branch was elsewhere).
6. **`torch.cuda._sleep` for the clock-settle barrier is fine on sm_75** but
   does NOT pin clocks; T4 power state still wobbles between trials. The
   A/A null arm (bench_protocol) is the empirical answer, not the sleep.

## Phase 0 — baseline roofline (B=16384 per production shape)

(pending — table populated below when the T4 run lands)

## Phase 1 — plumbing fixes

| Step | Change | Commit | Status |
|---|---|---|---|
| 1.1 | current CUDA stream at all 18 launch sites (was `nullptr`) | `d31b4c7` | compile-verified; capture test written |
| 1.2 | explicit load-order-independent update selector + per-tile pointers + %16 gates + PT_VERBOSE_DISPATCH | `44aeba5` | regression tests written; T4-pending |
| 1.3 | existing 5 test modules on T4 | — | pending same session |

## Phase 2 / 3 — pending Phase 0 numbers.
