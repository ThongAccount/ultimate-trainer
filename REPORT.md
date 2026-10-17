# REPORT — T4 under-utilization work (ThongAccount/ultimate-trainer)

Environment: Colab free-tier T4 (sm_75, 40 SMs, 65 TFLOPS FP16 peak,
~25-35 realistic at 70W cap, no cp.async concerns — sm_75 HAS LDGSTS,
no bf16, 64KB smem/SM). All numbers from `colab run --gpu T4` one-shot
runs; commits quoted per row; torch 2.11.0+cu130.

## Notes that contradict or update the brief

1. **`ncu` unavailable**: hosted VMs return ERR_NVGPUCTRPERM for hardware
   counters. Fallback per brief: CUDA events (median of >=20 after warmup 10),
   torch.profiler for kernel names/times, `nvcc -arch=sm_75 -Xptxas -v` for
   resources. Phase 3 acceptance therefore rests on measured >=1.15x speedups,
   never on reasoning.
2. **Brief 1.2 `_load_dx_tc` undefined — STALE**: it is defined
   (pack_update.py:249) and dX already had per-tile split pointers
   (`_dx_tc_fn` vs `_dx_tc_32_fn`). The real load-order defect was on the
   update side (shared `_up_tc_v2_fn` overwritten with the 32x32 build).
   Fixed + regression-tested; the dX claim is recorded as already-correct.
3. **`unpack_tensor()` is a CPU Python loop** (triple-nested, checkpoint-only)
   — unusable for the cuBLAS arm. Contradicts the brief's implicit assumption
   that a decode primitive exists. Wrote one: `kernels/packed_ternary/pack_decode.py`
   (GPU, one thread per element) — also the Phase 2 decode-once primitive.
4. **`colab run` takes a script path only** — `colab run ... bash -c '...'`
   fails with "Script not found: bash". All setup is inside the Python entry
   (`scripts/colab_bench.py`), not a shell wrapper.
5. **Fresh clones land on `master`, not the work branch.** Pin the branch +
   reset before measuring, or you silently benchmark a stale commit (this bit
   once: measured commit 28d2bab while HEAD was elsewhere).

## Phase 0 — baseline roofline (B=16384 per production shape)

(pending T4 run — table populated below when results land)

## Phase 1 — plumbing fixes

(pending)

## Phase 2 — decode-once + cuBLAS backend

(pending)

## Phase 3 — custom kernel work

(pending)
