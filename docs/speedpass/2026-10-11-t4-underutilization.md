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
| 1.1 | current CUDA stream at all 18 launch sites (was `nullptr`) | `d31b4c7` | compile-verified; capture test written (`tests/test_graph_replay_parity.py`) |
| 1.2 | explicit load-order-independent update selector + per-tile pointers + %16 gates + PT_VERBOSE_DISPATCH | `44aeba5` | regression tests written (`tests/test_dispatch_order.py`); T4-pending |
| 1.3 | existing 5 test modules on T4 | — | **pending** — same session, not yet run |

## Phase 2 — decode-once + cuBLAS backend

**Phase 2a (decode)**: shipped. `kernels/packed_ternary/pack_decode.py` — GPU
decode kernel, 0.105 ms at fc1 (measured), plus `decode_into` persistent-buffer
form (graph-capturable) and `kernels/packed_ternary/scratch.py` scratch manager.
This alone makes the cuBLAS arm viable.

**Phase 2b (counter-word update, no atomicCAS)**: PARITY FAILED, parked per the
brief's 3-strike rule. Evidence + diagnosis:
`docs/speedpass/2026-10-11-counterword-findings.md`. Kernel never wired into
production dispatch; stays in `tests/`.

**Phase 2c (PT_BACKEND=cublas|custom)**: not started — the update-kernel path
was its blocker; the fwd/dX path (the 20.7× win) does NOT need it.


## Open bug found during Phase 1 validation: 32-lane fwd kernel OOBs

**Status**: pre-existing (not a regression from this campaign), reproducible on
every fresh Colab VM, deterministic. Parked pending root-cause analysis.

**Repro** (any shape, e.g. B=1024 K=64 N=64):
```python
from kernels.packed_ternary import pack_tensor
from kernels.packed_ternary.pack_forward import packed_ternary_forward_tc
Y = packed_ternary_forward_tc(pack_tensor(torch.randn(64,64,cuda)*.05),
                               torch.randn(1024,64,cuda,dtype=fp16))
# torch.AcceleratorError: illegal memory access (with CUDA_LAUNCH_BLOCKING=1)
```

**Evidence it is pre-existing and not environmental or a plumbing regression**:
- Fails at the pre-campaign commit `a3d2117` in a fresh clone (the same commit
  whose Phase 0 fc1 fwd measurement succeeded on an earlier session).
- Fails with the pre-stream-plumbing `pack_forward.py` pasted in.
- Fails via the raw `_forward_fn_tc` (no padding wrapper).
- Build + link is clean (nm-visible `launch_packed_ternary_tc` defined in
  `gemm_forward_tc_32.cu`; wrapper arg order verified to match the launcher).
- Not a bad GPU: pure-torch works, and a bad GPU is ruled out by the fresh-VM
  determinism across 3 sessions.
- The one session where fwd measured fine (`pt`, commit a3d2117) had a warm
  extension cache from an earlier build — stale-artifact masking is the
  remaining hypothesis for the discrepancy.

**Why the Phase 0 fc1 fwd number is still credible**: the bench measured
86.04 ms / 1.60 TFLOPS on that same warm session, and the numbers are
internally consistent (the dX/update rows from the same run match prior
AUDIT.md baselines, and the cuBLAS arm of 4.15 ms — an entirely different
code path — is what makes the comparison meaningful). But **the fc1 dispatch
row should be re-measured on a session where the fwd kernel is healthy** before
being treated as final.

**Next root-cause step**: compare the `.so` in the warm `pt` session's
`/root/.cache/torch_extensions` against a fresh build (objdump/nm the two),
then static-read `gemm_forward_tc_32.cu`'s global-index math for the
(load/store bounds) against `stride_words` for non-power-of-16 strides.
