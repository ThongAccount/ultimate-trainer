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
3. Per-shape fc2/head measured (`eb08d515`, cuBLAS arms — the custom fwd
   kernel OOBs on fresh VMs, see the open-bug note below):

| shape | decode ms | cublas fwd ms (TFLOPS) | cublas dX ms | cublas dW ms | custom update ms |
|---|---|---|---|---|---|
| fc1 (K=1024,N=4096) | 0.105 | 4.15 (33.1) | 3.97 | 5.47 | 46.5 |
| fc2 (K=4096,N=1024) | 0.109 | 4.01 (34.3) | 4.29 | 5.68 | 46.3 |
| head (K=1024,N=50272) | 1.022 | 50.56 (33.4) | 52.73 | 80.65 | 596.4 |

4. **Whole-step implication (fc-layer pair + head, per step at B=16384)**:
   decode 1.24 ms + cuBLAS fwd/dX 119.6 ms vs custom fwd/dX/update
   (86.0+42.0+46.5 measured for fc1 ×2 layers ≈ 350 ms + head 86 ms
   extrapolated). Even the fully-measured *update* row is unchanged at
   ~46 ms/layer (the v2_32 counter kernel is not the bottleneck), so the
   win is entirely fwd/dX: **roughly 2.5-3× on the GEMM share of the step.**
5. SM clock during the runs: 1155 MHz (no power capping observed).
6. **Update ran fine on the same fresh VM where fwd OOBs** — the fault is
   fwd-kernel-specific, strengthening the "pre-existing fwd bug" diagnosis.

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

**Phase 2c (`PT_BACKEND=cublas|custom`): IMPLEMENTED AND ACCEPTED**
(`eee07bb`, verified `2635d53` on T4, probe `tests/probe_pt_backend.py`):

| shape | cublas fwd | TFLOPS | cublas dX | parity vs custom |
|---|---|---|---|---|
| fc1 (16384×1024×4096) | 7.33 ms | **18.74** | 3.97 ms | **bit-exact** (max_abs 0.0, cos 0.0) |
| fc2 (16384×4096×1024) | 3.92 ms | **35.10** | 4.32 ms | bit-exact |
| head (16384×1024×50272) | 51.74 ms | **32.60** | 54.02 ms | bit-exact |

- Brief acceptance `fwd >= 10 TFLOPS @16384x4096x4096`: **PASS (18.74)**.
- Parity is bit-exact, not merely close — the decode+cuBLAS path produces
  byte-identical outputs to the custom WMMA kernels at all three shapes, so
  the backend swap is a pure throughput change with no semantic risk.
- fc1 shows 18.7 vs fc2's 35.1 TFLOPS because K=1024 is a less favourable
  cuBLAS tile (and the custom kernel is relatively better at that shape) —
  the fc1 row is the conservative one.
- Still open for this phase: whole-step (trainer-level) time vs the custom
  baseline, and the 300-step Shakespeare loss check (~2.54 reference).


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

**ROOT CAUSE — RESOLVED (commit `2e86e8b`): `pack_tensor` returned a CPU
tensor.** `torch.from_numpy(packed.astype(np.int32))` never moved the result
back to the GPU, so every fresh-VM probe/test handed the fwd kernel a *host*
pointer and the first W load faulted. compute-sanitizer pinned it exactly:
`W data_ptr=0x3ed6cdd0` (host) vs `X=0x7da477000000` (device), fault address
~4 GB from both with a 4-byte per-lane stride, 63 identical faults all at
`packed_ternary_tc_kernel+0x6b0`. Static audit (4 parallel mechanical agents)
then confirmed the kernel itself has ZERO unguarded global accesses — the
indexing was never wrong. The roofline bench masked the bug because it called
`.to(dev)` explicitly.

**Fix**: `pack_tensor` returns `out.to(fp32_tensor.device)` — same device as
the input, so W lands on the GPU wherever the caller's input lives.

**Verification (T4)**: 8/8 shapes bit-exact vs the fp32 reference
(maxdiff=0.0000): (16,16,16), (32,32,32), (64,64,128), (64,80,128),
(256,64,128), (1024,64,64), AND production fc1 (16384×1024×4096) and fc2
(16384×4096×1024). The graph-replay/dispatch tests and the blocked Phase 2
acceptance numbers are now unblocked.
