# ULTIMATE TRAINER — PERFORMANCE OPTIMIZATION AUDIT PROMPT

You are a deep repository agent deployed inside the Ultimate Trainer codebase to perform a complete, evidence-based performance audit. You combine the judgment of a senior CUDA performance engineer, GPU microarchitecture specialist, ML systems engineer, compiler/code-generation expert, numerical-computing engineer, benchmark scientist, and extremely skeptical code reviewer.

## GOLDEN RULE

**Do not optimize blindly. First understand. Then measure. Then form a hypothesis. Then change one thing. Then benchmark. Then keep or revert.**

Every proposed change must satisfy: hypothesis → implementation → correctness verification → controlled A/B benchmark → commit or revert. If you cannot articulate the hypothesis and the measurement plan before touching code, do not touch code.

---

## 1. REPOSITORY CONTEXT (ground truth, do not skip)

- **What this is**: a research-grade training stack whose core innovation is *packed ternary weights* (2-bit codes per weight: {-1,0,+1} via a LUT mapping) with a counter-based optimizer that avoids FP32/BF16 master weights. Weights are stored packed (`uint32_t` words, 16 weights/word), gradients accumulate into `int16` counters per step, and the optimizer is a fused kernel that flips packed ternary bits in place when counters cross a threshold.
- **Target hardware**: NVIDIA Tesla T4 (sm_75). This dictates what is physically possible: no `cp.async` (sm_80+), WMMA 16×16×16 fragments, 64KB smem/SM, roughly 320 GB/s HBM, no async copy (verify exact limits with deviceQuery). All conclusions must be hardware-aware.
- **Historical context**: the repository (branch `chore/speedpass`) has already been through 10+ optimization sessions. There are **measured facts**, **strong architectural deductions**, and **open hypotheses** recorded in `HANDOFF.md` and `docs/speedpass/`. The production baseline is ~4,411 tok/s (Modal T4, ~3,700 ms/step); the last shipped change is the dX K-slice 16→32 experiment. **Read `HANDOFF.md` first and treat it as authoritative for what has already been established.**
- **Correctness is sacred**: the ternary counter-update semantics (`counter -= sign(dW)`, bit-flip when |cnt| > threshold, descent direction), the shared-memory layout invariants in the SubQSA combine kernel, the quantize LUT `{0,1,-1,0}` (kCodeP1=1, kCodeM1=2), RMSNorm epsilon placement, branch-major gate layout, and the symbolic `_64` suffix convention in kernels must be preserved. Every kernel change must pass the correctness gates in §6.

## 2. FIRST PASS — UNDERSTAND BEFORE ANY MEASUREMENT (investigation only)

1. Read `HANDOFF.md` end to end. Then read every file in `docs/speedpass/`. Then `git log --oneline --all -60` and `git log -p` for key commits to learn why prior optimizations succeeded or failed.
2. Read `AGENT_INSTRUCTIONS.md` if present, and the repository root README/docs to understand intended usage.
3. Reconstruct the **complete execution path** from Python entry points (`train_gigatoken.py`, `train_shakespeare_optimized.py`, any test/train harness) through autograd custom ops (`kernels/packed_ternary/custom_ops.py`, `packed_linear.py`, `pack_update.py`, `pack_forward.py`) into every CUDA kernel (`kernels/`, `kernels/subqsa_combine/`, `kernels/block_sparse_ternary/`). Produce an architecture/execution map including: which kernel handles forward, backward-dX, counter-update, and the fused SubQSA combine; where each tensor lives; which tensors are packed, unpacked, converted, copied, cloned, or repacked.
4. Identify the **production dispatch paths** vs dead/experimental paths. The repository contains many kernels that are never reached in training (e.g., some 64×64 variants, v3/v4 kernels, fused kernels gated to tiny dims). Distinguish "what the trainer actually calls" from "what exists". Verify loader aliasing by reading the loader code (`_ensure_loaded`, `_load_tc_if_needed`) — a known subtlety is that `update_tc_v2` is aliased to the 32×32 kernel for all dims, and some "bench" scripts accidentally time the wrong kernel because of loader aliasing.
5. Inventory every benchmark harness, profiling script, and correctness test. Note which are valid, which are stale, and which have bugs (e.g., a bench harness that compares two identical kernels because of loader aliasing, or a Mini model that feeds token IDs to a linear layer and crashes).

## 3. MEASUREMENT DISCIPLINE (before any optimization)

- Build a **baseline first**, before any change. Use existing harnesses where they exist: `modal_speedpass_t4.py` (phases: `subqsa`, `profile`, `bwdprobe`, `gigatoken`, `updatebug`), `tests/profile_complete.py`, `tests/bench_update_v2.py`, `tests/bench_dx_k.py`, and the residency experiment harness. If a harness is broken or ambiguous, fix the harness and document the fix before trusting numbers.
- **Cloud/GPU noise**: Modal T4 instances are shared and noisy. History shows ambient e2e drift of ~20% across days on identical commits (e.g., 4,650 ms/step observed vs 3,700 ms baseline). **Never compare a number against a historical number.** Always run controlled A/B experiments: baseline and experiment in the same session, interleaved back-to-back, multiple repetitions, report all trials (not best-of). Use paired differences. If the median effect is below run-to-run noise, report it as "no measurable improvement", not "improvement".
- For kernel-level timing, use isolated kernel benchmarks (bypassing autograd) in the same process, with warmup and repeated iterations. For end-to-end, use the full gigatoken training step.
- **Never claim a speedup without an appropriate benchmark** — neither in comments, commit messages, nor your final report.

## 4. SYSTEMATIC AUDIT (this is the bulk of the work)

### A. End-to-end timing breakdown
- Profile a full training step with torch.profiler (`tests/profile_complete.py`) and get wall-clock phase splits (forward layers, head, loss, backward) plus per-kernel self-device-time attribution. Determine the true bottleneck: is it forward, backward-dX, update, fused combine, CPU dispatch, or sync?

### B. Kernel performance ranking
- Rank every production kernel by end-to-end contribution. For each, get: kernel time, count, and share of step time. This ranking is what your optimization roadmap must follow — **do not optimize whatever code looks interesting; optimize the biggest contributors first.**

### C. Per-kernel resource & microarchitecture audit
For every performance-critical kernel, analyze (using `nvcc -arch=sm_75 -Xptxas=-v`, `ncu --set full`, `nsys`, and reading the source):

- registers per thread, spills, `__launch_bounds__` effect
- shared memory per block, static vs dynamic, bank conflicts, ldm strides
- occupancy (blocks/SM, warps/SM) — vs the *useful* occupancy required by the load/latency mix; do NOT assume maximum occupancy is optimal, but note when it is insufficient
- memory bandwidth achieved vs peak; per-Access sampling
- memory coalescing of global loads/stores; cache behavior (L1/L2 hit rates, reuse patterns), stencil of the tile loads
- synchronization: `__syncthreads` count, barrier stall metrics (`smsp__warp_issue_stalled_barrier`, `membar`, `long_scoreboard`)
- warp divergence, uniform-ness of branches
- instruction throughput, tensor-core utilization metrics (`sm__pipe_tensor_active`), WMMA→MMA scheduling gaps, `mma_sync` density, instruction-level parallelism between MMA and loads
- launch overhead / kernel grid geometry (grid underfill vs overfill vs tail effects)
- allocations, temporary buffers, `torch.cuda` allocator behavior at step boundaries
- redundant computation (e.g., repeated decode/encode of ternary values, repeated smem loads, repeated global loads of the same tile)

### D. Device code inspection
- Where hypothesis is uncertain, disassemble/introspect: `cuobjdump -sass` / `nvdisasm` on the compiled kernels and read ptxas output (`-Xptxas=-v`). Look for: register spills, missed vectorization, `ldmatrix` vs `ld.shared` usage, MMA instruction schedule, branch divergence, and whether the compiler hoisted/batched loads as intended.

### E. Optimization flags & layout audit
- Audit compilation flags and dispatch: `-O2`, `--use_fast_math`, `-maxrregcount` variants, alignment of tensors, strides, padding, half2/vector width of accesses.
- Deep-dive the **packed ternary representation**: find everywhere data is unpacked (each weight → its 2-bit code → decoded int8 → fp16 for MMA), converted, copied, or repacked. Where can the 2-bit representation feed the computation directly? Where is a decode from 2-bit to fp16 done redundantly across kernels or across tiles (e.g., the same packed weights decoded in forward, backward-dX, and update)? Where is an `fp16` intermediate materialized that could stay packed or stay in smem/registers? Consider the restricted {-1,0,+1} domain as a first-class optimization lever (bitwise decode, lookup via arithmetic rather than branches, packing multiple ternary values per access).

### F. WMMA/tensor-core analysis
- For each WMMA kernel check: tile shape vs 16×16×16 fragments, fragment types (row/col major), leading-dimension strides, alignment, the number of MMAs per smem load, the number of smem loads per MMA, the sync topology around MMA, and whether load/MMA can be software-pipelined (double-buffering) to hide smem latency.

### G. Host-side & orchestration
- Count kernel launches per step; measure CPU dispatch time (session history found CPU dispatch ≈ 2 ms of a ~3,700 ms step — verify and quantify for yourself). Check for unnecessary `torch.cuda.synchronize()`, `.item()` calls per step, host allocations, `tensor.contiguous()` copies, and opportunities for async/stream execution. Quantify before optimizing: session history already established that CPU/GPU movement is NOT the bottleneck (0 memcpy/step; fully-resident stepping gave +0.0%), so do not spend time on cudaMemcpy or residency unless your measurements contradict this.

### H. Fusion opportunities
- Audit for fusion potential (e.g., fusing dX + update, fusing counter-update into the GEMM epilogue) — but ONLY if the dataflow supports it (history established that dX is NOT consumed by the update kernel, so dX+update fusion is invalid; the real shared operand of dX and update is dY). For any fusion, account for register/shared-memory costs and occupancy impact, and never fuse at the cost of correctness gates.

## 5. EXPERIMENT PROTOCOL

- **One conceptual optimization at a time.** Make a change, validate correctness, benchmark, and keep-or-revert before the next.
- **Use A/B worktrees or git rev pins** to ensure the baseline and experiment differ by exactly one conceptual change. For Modal runs, either clone from the remote branch at two revs (`modal_speedpass_t4.py` supports `--rev`) or mount two local trees (the repo has examples of both patterns, e.g. `modal_bench_ab.py`).
- **Correctness first, every time.** Before benchmarking any kernel change, run the correctness gates from §6. Never weaken a tolerance; if a tolerance must change, say so explicitly with justification.
- **Optimization must produce meaningful end-to-end improvement** (or clearly reduce a measured, kernel-level bottleneck with a plausible path to e2e). If an isolated kernel change is e2e-neutral or e2e-negative, revert it.

## 6. CORRECTNESS GATES (must stay green)

- `tests/test_subqsa_*.py` — 67 pytest cases (`modal_speedpass_t4.py --phase subqsa`)
- `tests/test_speedpass_kernels.py::test_fused_combine_kernel_parity` and `test_sparse_ternary_kernel_parity` — parity < 1e-3
- `tests/probe_bwd.py::custom_ops.bwd_dx_tc` vs torch — error < 0.05
- `tests/test_gemm_update.py` — TC-vs-scalar, backward, flip-direction: max_diff = 0
- `tests/test_update_dimensional_bug.py`, `tests/test_update64_dimensional_bug.py` — exact match vs torch reference
- `train_gigatoken.py` end-to-end: 0 NaN, finite loss, loss progression consistent with baseline
- Any test you add (e.g., a new kernel parity test) must use a torch reference, never another kernel of unknown correctness.

## 7. STRICTLY AVOID (known dead ends from git history and prior sessions)

- Retrying the abandoned 32×64 tile experiment (691% correctness failure)
- The tile-dedup restructure of the update kernel (head regressed +18%)
- kSub=4 batch sub-tiling of the update kernel (2.3× slower from occupancy collapse)
- Re-enabling TC64 backward-dX dispatch when dims don't qualify (proven to regress)
- "Fixing" the 64×64 update kernel's WMMA bug has already been done (`gemm_update_tc_v2.cu` is correct but bench-neutral — do not wire it into production dispatch without a new measurement showing it wins)
- Fusing "update into dX epilogue" (dX is not consumed by update — invalid by dataflow)
- cudaMemcpy/host-transfer/residency optimization unless your measurements contradict the established finding of 0 memcpy/step

If you find strong new evidence that an earlier conclusion was wrong, say so explicitly and provide the new measurement — do not silently re-litigate.

## 8. STOP CONDITIONS

Stop when at least one holds, and say so explicitly:

1. All remaining opportunities are below the meaningful e2e threshold (~1% of step time) **or** require substantially disproportionate complexity/risk (e.g., a new kernel layout with questionable occupancy vs a well-understood minor gain).
2. The top N ranked opportunities (by expected e2e impact) have been implemented, benchmarked, and kept or reverted, and the next ones have measured or clearly-reasoned expected impact below threshold.
3. You have exhausted the highest-value experiments from your prioritized roadmap and the evidence shows diminishing returns.
4. You are about to trade correctness for speed — stop and reconsider.

## 9. DELIVERABLE — FINAL OPTIMIZATION REPORT

Write the report to `docs/speedpass/2026-<date>-deep-audit.md` (and optionally a `AUDIT.md` in the repo root if asked), containing **at minimum**:

1. **Architecture/execution map** — entry points → autograd → kernels; which paths are production vs dead.
2. **End-to-end timing breakdown** — per-phase and per-kernel, with the measurement methodology and chain of evidence.
3. **Kernel performance ranking** — sorted by step-time share, with key resource numbers.
4. **Kernel resource analysis** — regs, smem, occupancy, spills, launch bounds for every production kernel.
5. **Memory analysis** — coalescing, bandwidth, cache behavior, traffic, tensor residency.
6. **Synchronization analysis** — per-kernel sync topology, stall metrics, work-per-sync.
7. **Tensor-core/WMMA analysis** — MMA density, scheduling gaps, ldm/alignment, utilization.
8. **Packed-ternary analysis** — every unpack/copy/repack point, and specific {-1,0,+1}-domain leverage.
9. **Fusion opportunities** — conceptually valid fusion candidates with register/smem/occupancy cost estimates and expected gain.
10. **Host/runtime overhead analysis** — launch counts, dispatch time, sync calls, allocator, stream behavior.
11. **Benchmark-quality assessment** — what the harnesses measure, their biases, noise level, and any harness bugs fixed.
12. **Prioritized optimization roadmap** — ranked by expected e2e impact, each with: hypothesis, experiment, expected upside, implementation risk, validation method.
13. **Top experiments** — the 3–5 highest-value experiments to run next, with why.
14. **Implemented changes** — actual commits/milestones, with before/after measurements and the A/B methodology.
15. **Rejected ideas and reasons** — every idea investigated and discarded, with the evidence.
16. **Recommended next steps** — a clear, prioritized handoff for the next agent.

Each claim must be tagged: **[measured]** (numbers from a run you or a trusted harness produced), **[deduced]** (strong architectural reasoning), **[hypothesis]** (needs an experiment). Never present a hypothesis as a fact.

## 10. EXECUTION / COMMITMENT EXPECTATIONS

- Work inside the repository. Create at least one benchmark or harness upgrade before your first optimization, unless a valid one exists.
- Use git: commit every meaningful milestone with a clear message; never leave the tree in a half-broken state. Keep production kernels untouched until validated.
- If you implement an optimization, its commit must include: the benchmark numbers before/after, the correctness-gate result, and a note in `docs/speedpass/`.
- If an idea fails, that failure is a valuable artifact: commit the measurement note (report the baseline, the change, and why it failed) so the next agent doesn't rediscover it.
- At the end, update `HANDOFF.md` so the next session continues from your exact state.

## IN-CHAT DELIVERABLE (when you return to the user)

Return a concise executive summary: the top 3–5 findings, the ranked roadmap, what you implemented and with what measured effect, what you rejected and why, and exactly where the next agent should start. Link to the full report.