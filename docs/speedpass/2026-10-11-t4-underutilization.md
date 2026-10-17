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

Commit `a3d2117`, T4 sm_75 @ 1155 MHz, torch 2.11.0+cu130, median-of-20
after warmup-10, cuda events. **fc1 measured (K=1024, N=4096):**

| arm | ms | TFLOPS | vs dispatch |
|---|---|---|---|
| **dispatch fwd** (packed ternary, custom kernels) | **86.04** | **1.60** | 1.0× |
| **cuBLAS fwd** (pre-decoded fp16 `X @ W16^T`) | **4.15** | **33.15** | **20.7× faster** |
| dispatch dX (`dY @ W`) | 42.02 | 3.4 | — |
| cuBLAS dX | 3.97 | 35.9 | 10.6× faster |
| dispatch update (dW→counter→flip) | 46.53 | — | — |
| **decode W → fp16** (new GPU kernel) | **0.105** | — | ~2.6% of one GEMM |

### Phase 0 conclusions

1. **The dispatch path reaches 1.6 TFLOPS = 2.5% of the T4's 65 TFLOPS
   FP16 peak.** The brief's "~1%" estimate is confirmed (order of magnitude).
2. **Pre-decoded W + cuBLAS is 20.7× faster on fwd, 10.6× on dX, and
   decode costs only 0.105 ms** — for fc1 that means
   decode+GEMM = 4.15ms vs 86.04ms, a 20.7× win *including* decode.
   The "2-bit packing gives no throughput benefit at B·SEQ=16384" claim is
   now measured, not argued.
3. Per-shape fc2/head still pending (each Colab invocation is one shape;
   the pattern now works and is cheap: ~9 min per shape on a warm session).
4. SM clock during the run: 1155 MHz (no power capping observed).

## Phase 1 — plumbing fixes

| Step | Change | Commit | Status |
|---|---|---|---|
| 1.1 | current CUDA stream at all 18 launch sites (was `nullptr`) | `d31b4c7` | compile-verified; capture test written |
| 1.2 | explicit load-order-independent update selector + per-tile pointers + %16 gates + PT_VERBOSE_DISPATCH | `44aeba5` | regression tests written; T4-pending |
| 1.3 | existing 5 test modules on T4 | — | pending same session |

## Phase 2 / 3 — pending Phase 0 numbers.
