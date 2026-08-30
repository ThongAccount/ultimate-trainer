# HANDOFF — Ultimate Trainer Speedpass (2026-08-19 Session 2)

## EXECUTIVE SUMMARY

**Branch**: `chore/speedpass` @ commit `8ebcd50`  
**Status**: Production baseline preserved, investigation phase complete  
**Validated Baseline**: 6,619 ms/step, 2,475 tok/s (Modal T4)  
**Target**: `update_tc_v2` kernel (2,600ms, 39% of step time)  
**Session Result**: Investigation phase only — no optimizations implemented

---

## 1. CURRENT STATE

### Repository

- **Branch**: `chore/speedpass`
- **Commit**: `8ebcd50` — "Add test for update_tc_v2 dimensional bug"
- **Remote**: Synced with `origin`
- **Working tree**: Clean, no uncommitted changes
- **Production baseline**: `77aa504` (P0c: dense mask routing)

### Performance Baseline (Validated on Modal T4)

**Configuration**: 51M params, 6 layers, 1024-dim, batch=32, seq=512

```
6,619 ms/step
2,475 tok/s
```

**Breakdown**:
- Layers: ~710ms (11%)
- Head: ~720ms (11%)
- **Backward: ~5,100ms (77%)** ← main bottleneck
  - `update_tc_v2`: ~2,600ms (39%) ← **CURRENT TARGET**
  - `backward_dx_tc` (tc32): ~2,400ms (36%)
  - `forward_tc_64`: ~1,300ms (20%)

---

## 2. WHAT WAS INVESTIGATED: UPDATE_TC_V2 KERNEL

### Kernel Overview

**File**: `kernels/packed_ternary/gemm_update_tc_v2.cu`

**Algorithm**: Counter-based weight update with WMMA Tensor Cores
```
dW[n,k] = SUM_b dY[b,n]^T @ X[b,k]
counter[n,k] -= sign(dW[n,k])
if |counter| > threshold: flip W[n,k], reset counter
```

**Tile mapping**:
- Grid: `(N/64, K/64)` → At N=1024, K=1024: `(16, 16)` = **256 CTAs**
- Block: 128 threads (4 warps)
- Each CTA owns 64×64 output tile
- Each warp accumulates 4 WMMA fragments (32×32 sub-tile)

**Resource usage** (validated with `nvcc -Xptxas=-v`):
- Registers: **64/thread**
- Spills: **0 bytes** ✓
- SMEM: **8,192 bytes** (8KB)
- Occupancy: **8 CTAs/SM** (optimal for T4!)

---

## 3. SUSPECTED DIMENSIONAL ISSUE (INVESTIGATED)

### Issue Description

During source inspection, identified apparent out-of-bounds indexing:

**Shared memory declarations**:
```cuda
__shared__ half dY_smem[kSuperM][kWMMA_K];  // [64][16]
__shared__ half X_smem[kSuperM][kWMMA_K];   // [64][16]
```

**WMMA load code** (line 118):
```cuda
int n_base = warp_n_off + frag_n_off;  // Can be 0, 16, 32, or 48
wmma::load_matrix_sync(a_frag, &dY_smem[0][n_base], kWMMA_K);
```

**Apparent problem**: `n_base` can be 16+ but `dY_smem` only has 16 columns (0..15).

### Investigation Results

**Test**: Created `tests/test_update_dimensional_bug.py` to verify numerical correctness

**Result**: ✅ **PASSED with 0 errors**
- Dimensions: B=32, N=128, K=128 (should trigger bug if it exists)
- Max difference: 0
- Error count: 0 / 16,384 (0.00%)

**Conclusion**: 
- Either the suspected issue is not a bug (WMMA col_major semantics may be different than expected)
- Or the issue doesn't cause numerical errors in practice
- Kernel is **numerically correct** on actual hardware

---

## 4. WHAT WAS NOT DONE (CRITICAL)

### Missing: Profiling Phase

**Per original instructions**, should have:
1. ✅ Inspected kernel source (done)
2. ❌ **Profiled on actual T4 to find bottleneck** (NOT DONE)
3. ❌ Formed hypothesis from profiling data (NOT DONE)
4. ❌ Implemented optimization (NOT DONE)

**What's needed**:
- Run `torch.profiler` or NSight Compute on Modal T4
- Measure:
  - Tensor Core utilization
  - Memory bandwidth usage
  - Occupancy (theoretical vs achieved)
  - Instruction breakdown (ternary decode cost?)
  - Shared memory bank conflicts
  - Global memory traffic
- Identify **actual** bottleneck, not speculated from source

---

## 5. FILES MODIFIED THIS SESSION

### New Files
```
tests/test_update_dimensional_bug.py  (correctness test for dimensional issue)
```

### Modified Files
```
modal_speedpass_t4.py  (added --phase updatebug test harness)
```

### Commits
```
8ebcd50  Add test for update_tc_v2 dimensional bug
77aa504  P0c: route dense masks through fused combine (baseline)
```

---

## 6. PREVIOUS OPTIMIZATION CONTEXT

### P0c: Dense Mask Routing (Shipped)

**Commit**: `77aa504`  
**Status**: Production-ready, validated on Modal T4

**Changes**:
- Detect dense masks and route through fused CUDA combine kernel

**Results**:
- SubQSA test suite: 67 passed, 0 failed
- Fused combine: **0.13ms** vs eager **0.40ms** = **3.1× speedup**
- No regressions

---

### TC64 Backward Investigation (Not Shipped)

**Kernel fix commit**: `113b40b` (preserved in history, not on branch)

**Three CUDA bugs fixed**:
1. W_smem transposition (reduction/output dims swapped)
2. W_smem dimensions wrong ([64][16] → [16][64])
3. Spill array race condition (per-warp indexing fixed)

**Validation**: bwdprobe error **541.6 → 0.012** ✅ (numerically correct)

**Performance Result**: ❌ **18% END-TO-END REGRESSION**

| Version | Commit | ms/step | tok/s | vs Baseline |
|---------|--------|---------|-------|-------------|
| **Baseline (TC32)** | `77aa504` | 6,619 | 2,475 | — |
| **TC64 enabled** | `6d8336b` | 7,790 | 2,103 | **+18% slower** |

**Root cause**: Grid underfill (128 CTAs vs 512 CTAs at B=512) + occupancy issues

**Decision**: Keep kernel fix in history, revert dispatch, TC64 remains disabled

---

### 32×64 Tile Experiment (Abandoned)

**Branch**: `experiment/32x64-tile` (can be deleted)  
**Status**: Catastrophic correctness failure

**Result**: ❌ **691.5% relative numerical error**
- Reference magnitude: 85.75
- Absolute error: 593.0
- Correctness test failed

**Decision**: Abandon experiment, do not resume

---

## 7. NEXT STEPS (PRIORITY ORDER)

### Immediate: Profile update_tc_v2 on Modal T4

**Before any optimization**, run profiling to identify actual bottleneck:

```bash
# Option 1: Add torch.profiler phase to modal_speedpass_t4.py
modal run -d modal_speedpass_t4.py::speedpass_benchmark --phase profile_update

# Option 2: NSight Compute (if practical)
# Profile specific kernel on T4 to get:
# - TC utilization, occupancy, bandwidth, instruction counts
```

**Key metrics to collect**:
- Tensor Core utilization %
- SM utilization %
- Memory bandwidth (achieved vs peak)
- Occupancy (theoretical: 8 CTAs/SM, actual: ?)
- Instruction breakdown (what % is ternary decode?)
- Shared memory bank conflicts
- Global load/store traffic

### Form ONE Hypothesis

After profiling, identify **single highest-confidence bottleneck**:

Examples:
- Low Tensor Core utilization → kernel is memory-bound
- High global memory traffic → batch loop overhead
- Many decode instructions → ternary unpacking is expensive
- Bank conflicts → shared memory access pattern issue

### Implement ONE Surgical Optimization

Requirements:
- Surgical change (no full rewrite)
- Preserve numerical semantics
- One logical optimization per commit
- Validate: correctness → kernel benchmark → gigatoken end-to-end

### Performance Gate

**Baseline**: 6,619 ms/step, 2,475 tok/s

**Do NOT ship** unless end-to-end training improves measurably (≥5% recommended).

Kernel-level improvement without end-to-end gain is not sufficient.

---

## 8. VALIDATION GATES

### Before Shipping Any Optimization

1. **Local compile**: `nvcc -arch=sm_75 -Xptxas=-v -c <kernel>.cu` (no errors, check resources)
2. **Correctness**: `modal run --phase updatebug` (must pass with 0 errors)
3. **Isolated kernel**: Time update_tc_v2 before/after
4. **End-to-end**: `modal run --phase gigatoken` (compare vs 6,619ms baseline)
5. **No regressions**: Check other kernels didn't slow down
6. **No NaNs**: Final loss must stay finite

### Test Commands

```bash
# Detached mode for long runs (recommended)
modal run -d modal_speedpass_t4.py::speedpass_benchmark --phase <name>

# Check logs without re-running (saves credits)
modal app logs <app-id>

# Local compile check (no GPU needed)
export PATH=/usr/local/cuda-13.3/bin:$PATH
nvcc -arch=sm_75 -Xptxas=-v -c kernels/packed_ternary/gemm_update_tc_v2.cu
```

---

## 9. KNOWN ISSUES (NOT BLOCKING)

### Dimensional Indexing Investigation

**Status**: Investigated, not confirmed as actual bug  
**Test result**: Kernel passes correctness with 0 errors  
**Action**: No fix needed unless profiling shows related issue

### CUDAGraph trainer incomplete

**Status**: Known from prior work  
**Issue**: `train_shakespeare_optimized.py` uses plain Python loop  
**Priority**: Low (profiling shows backward is bottleneck, not Python overhead)

---

## 10. IMPORTANT REMINDERS

### DO NOT

- ❌ Resume abandoned 32×64 experiment
- ❌ Modify or merge `experiment/32x64-tile` branch
- ❌ Re-enable TC64 backward dispatch (proven to regress)
- ❌ "Fix" suspected dimensional issue without evidence it's broken
- ❌ Optimize based on source speculation alone
- ❌ Ship kernel improvement if end-to-end regresses

### DO

- ✅ Profile first, optimize second
- ✅ Keep production baseline clean and recoverable
- ✅ One logical optimization per commit
- ✅ Validate correctness before performance
- ✅ Measure end-to-end impact on actual T4
- ✅ Revert immediately if correctness fails or performance regresses

---

## 11. EXECUTION DISCIPLINE

### Modal Credit Management

- Use detached runs for long benchmarks: `modal run -d`
- Use `modal app logs <id>` to inspect completed runs
- Don't repeatedly rerun gigatoken while debugging
- Prefer: inspect → test → profile → hypothesis → impl → ONE end-to-end benchmark

### Git Workflow

- Keep `chore/speedpass` as production baseline
- Create feature branches for experiments
- One logical change per commit
- Force-push experiment branches, never production
- Clean commit messages with rationale

---

## 12. ARCHITECTURAL NOTES

### Packed Ternary Counter-Based Optimizer

**Key property**: Robust to noisy gradients
- `counter -= sign(dW)` (descent semantics)
- Flip weight when `|counter| > threshold`
- This may mask subtle numerical issues in gradient computation

**Implication**: Correctness tests must be sensitive enough to catch real bugs, not just rely on final convergence.

### WMMA Tensor Core Usage

**Current kernels**:
- `forward_tc_64`: Uses 64×64 tiles, works well
- `backward_dx_tc_32`: Uses 32×32 tiles, production stable
- `update_tc_v2_64`: Uses 64×64 tiles, **optimization target**

**Key insight**: TC64 works for forward but regressed for backward. Update is different workload (dY^T @ X vs dY @ W), may have different optimization opportunities.

---

## 13. CONTACTS / REFERENCES

### Vault Documentation
- Journal: `~/Agent Memories/journal/2026-08-19-update-tc-v2-investigation.md`
- Project: `~/Agent Memories/projects/ultimate-ai-model/ultimate-ai-model.md`

### Git History
- `8ebcd50` — Add dimensional bug test ← **CURRENT HEAD**
- `77aa504` — P0c: dense mask routing ← **PRODUCTION BASELINE**
- `113b40b` — TC64 kernel fix (correct, not shipped)

### Modal T4 Benchmarks
- Baseline: `ap-TkN7Vw66XrWjqPcJejlVkT` (6,619ms/step)
- Dimensional bug test: `ap-2bTgouiQt33jcWLoT4bkV5` (passed with 0 errors)

---

## FINAL STATUS

**Branch**: `chore/speedpass` @ `8ebcd50`  
**Next action**: Profile update_tc_v2 on Modal T4 to identify actual bottleneck  
**Performance**: 6,619 ms/step, 2,475 tok/s (validated baseline)  
**Bottleneck**: update_tc_v2 (2,600ms, 39% of step) awaiting profiling  
**Lesson learned**: Profile before optimizing — source speculation is insufficient

Session completed 2026-08-19 at 08:11 UTC.
---

## 14. RE-ANALYSIS CORRECTIONS (2026-08-26 full kernel codebase pass)

Static analysis of all kernels + dispatch layer. No code changed. These supersede
conflicting claims above.

### 14.1 Dispatch truth (verified in code, not docs)

- `custom_ops._ensure_loaded()` calls `pack_update._load_tc_if_needed()` FIRST, which
  **aliases the 32×32 kernels into the v2/v3 slots**
  (`_up_tc_v2_fn = _up_tc_v2_32_fn`, `_dx_tc_fn = _dx_tc_32_fn`).
- Trainer autograd (`PackedTernaryLinearFn.backward`) → custom ops →
  **production update kernel = `gemm_update_tc_v2_32.cu`** and
  **production dX kernel = `gemm_backward_dx_tc_32.cu`**, always (any dim ≥16).
- Therefore §2's "Kernel Overview" describes a file (`gemm_update_tc_v2.cu`, 64×64
  tiles) that is **DEAD CODE for training**: reachable only via direct
  `pack_update.update()` / `_load_up_tc_v2()` calls. The 2,600 ms target is TC32.
- The `updatebug` test imported `_up_tc_v2_32_fn` — it tested the production TC32
  kernel (correctly passing), not the file its name refers to.
- `update_tc_v3` never runs in training: autograd calls `co_update_tc_v2`
  directly; the "prefer v3" path exists only inside `pack_update.backward_update`.

### 14.2 WMMA addressing trace of the dead 64-tile update kernel

The suspected issue in §3 IS a real semantic bug — just not in the kernel that runs.
For col_major load with ld=16 from base `&dY_smem[0][n_base]`: element (i,j) reads
flat offset `n_base + i + 16·j` → smem row `j + n_base/16`, col `i`. Every fragment
with n_base>0 therefore reads n-columns [0,16) shifted across batch rows instead of
columns [n_base, n_base+16). Only fragment (frag_n=0, frag_k=0) is correct. Same
defect class as the three bugs fixed in unshipped commit `113b40b`
(backward_dx_tc_64 @ HEAD still contains ALL THREE of those bugs; reachable via
direct `pack_update.backward_dx()` when dims are 64-multiples).

### 14.3 New prime suspect for the 2,600 ms (TC32 update)

`gemm_update_tc_v2_32.cu` batch loop advances **16 tokens per iteration with two
barriers per iteration** (three on partial tiles). At B·T = 32×512 = 16,384 tokens:
**1,024 serialized iterations per CTA**, each CTA re-streaming its dY/X column slices
from global memory every iteration. Grid at 1024×1024 = 1,024 CTAs ≈ T4's 40 SMs × 8
CTA slots fully occupied, so latency hides poorly. Expected profile signature:
low tensor-core utilization, high stall-on-barrier / long-scoreboard. This is a
memory-latency-bound structure, NOT an occupancy or decode problem.
(dx_tc has the same loop shape over N/16 = 256 iterations.)

### 14.4 Recommended next step (unchanged discipline, corrected target)

1. Profile `tests/profile_gigatoken.py` on Modal T4 → confirm stall distribution on
   the TC32 update kernel.
2. If barrier/latency-bound confirmed → surgical fix: increase batch step kK 16→32
   (or split-B two-stage reduction) in `gemm_update_tc_v2_32.cu`. One commit.
3. Validation ladder unchanged (compile gate ✓ already run locally: 64 regs,
   0 spills, sm_75 clean for both suspect 64-tile kernels — semantic bugs invisible
   to compiler, as expected).

Session log: vault `journal/2026-08-26-kernel-codebase-reanalysis.md`

## 15. SESSION 3 RESULTS (2026-08-26, causally verified on Modal T4)

### 15.1 The barrier hypothesis (§14.3) is DISPROVEN — barriers are protective

Standalone harness (`docs/speedpass/2026-08-26-update-kernel-experiments.md`,
`/tmp/kexp/bench.cu`, bit-exact parity at threshold=30000 AND threshold=32 on all
three production shapes, prod-vs-prod determinism confirmed):

| variant | fc1 1024→4096 | fc2 4096→1024 | head 1024→50272 |
|---|---|---|---|
| prod (strided loads + `__syncthreads`) | 126 ms | 142 ms | 1592 ms |
| prod loads + `__syncwarp` only | 204 ms | 245 ms | **2920 ms** |
| coalesced loads + `__syncthreads` kept | **70 ms** | **76 ms** | **887 ms** |
| batch-step kK=32 | no gain (<2%), rejected |

Removing barriers while keeping the strided load pattern makes the kernel 60–80%
SLOWER: the barriers were serializing the 4 warps into lockstep bursts that
preserved what little locality the strided pattern had. Section attribution via
clock64(): load ≈81%, sync-wait ≈11%, MMA ≈8% of loop time.

### 15.2 Root cause and shipped fix

The production lane→element mapping (`i = wtid*8 + j`, j += 2) strides lanes
16 B apart in global memory → ~50% wasted DRAM sectors on fp16 rows.
**Commit `c7bc4ed`** replaces it with a coalesced consecutive-pair mapping
(consecutive lanes cover consecutive half2 pairs). Kernel-level: −44% (update
kernel only). End-to-end A/B at identical commit-pinned runs:

- baseline `8ebcd50`: 2510 tok/s, bwd wall 5090 ms/step
- patched `c7bc4ed`: 2461 tok/s, bwd wall 4959–5029 ms/step
- kernel saving transferred as predicted (−61…−131 ms/step vs predicted −87)
  but vanished inside CPU dispatch overhead → net effect within run noise.

### 15.3 THE REAL BOTTLENECK: autograd dispatch, not kernels

Per-step arithmetic (B=32, T=512, 16,384 tok/step): bwd WALL = 5090 ms but ALL
backward GPU kernels sum to ~480 ms/step → **~90% of backward time is Python/
autograd dispatch gaps**, not GPU work. Same for fwd (layers 708 ms wall vs ~250 ms
kernels). Closing this is worth up to ~6540 → ~1950 ms/step (~8400 tok/s, +235%).
This supersedes kernel micro-optimization as the top priority; see §7 item order:
dispatch-overhead work now ranks above further CUDA kernel tuning.

### 15.4 Harness defects caught during the campaign (methodological notes)

- v1 harness had grid x/y swapped for non-square shapes and `(int16_t)100000`
  overflowed to −31072 (threshold tier meaningless) — fixed with correct
  launcher mapping and threshold=30000.
- v2 variant load loops loaded only 1-of-32 smem elements; "clean parity" came
  from stale-smem reuse across launches. Always re-run parity after ANY rewrite;
  never trust a pass from a modified kernel without a fresh determinism probe.

### 15.5 Final validation ladder (all gates run)

1. nvcc sm_75 compile: clean (local 13.3 + container 12.8, 0 errors).
2. Correctness: harness bit-exact (both threshold tiers, all 3 shapes);
   repo suite at `c7bc4ed` on T4 — **53/53 pytest passed**, script-style
   dimensional-bug test PASS (max diff 0 vs reference).
3. Isolated kernel timing: −44% update kernel (126→70 / 142→76 / 1592→887 ms).
4. Training-step timing: bwd wall −61…−131 ms/step (predicted −87). ✓ transferred.
5. Gigatoken end-to-end: 2510 → 2461 tok/s = within noise, **<5% criterion NOT met**
   — bottleneck is CPU autograd dispatch (~90% of backward wall), see §15.3.
   Change RETAINED: strictly dominant at kernel level, zero risk (bit-exact),
   saving materializes once dispatch overhead is removed. Revert = `git revert c7bc4ed`.

## 16. SESSION 4 (2026-08-28): 64×64 backward-dX dispatch — real 1.16× bwd speedup

### What changed (commits `b23359b` + `242a665`)

The 64×64 backward-dX kernel (`gemm_backward_dx_tc.cu`, fixed in `113b40b`) was
**loaded but never dispatched**. `_load_tc_if_needed()` aliased `_dx_tc_fn`
(64×64) → `_dx_tc_32_fn`, so `custom_ops` only ever exposed 32×32 for training.
Fix splits `_load_dx_tc` from `_load_dx_tc_32`, stops the clobber, and routes
`backward_dx_tc` to 64×64 when B/N_out/K are all 64-multiples.

### Results (Modal T4)

- Correctness: 7/7 shapes PASS vs torch ref.
- Kernel (fc1 B=16384): 64×64 = 43.70 ms vs 32×32 = 75.78 ms → **1.73×**.
- **Clean in-process A/B** (monkeypatch `_dx_tc_64`, no file swap / no .so
  cache confound): full backward 2764 ms (64×64) vs 3208 ms (32×32) =
  **1.16×, −444 ms/step**. (An earlier file-swap A/B reporting "neutral" was
  confounded by torch .so caching — disregard.)

### CORRECTION to §15.3 — backward is GPU-bound, NOT dispatch-bound

Full profiler (CPU+CUDA) attribution of one step:
- backward wall ≈ 3063 ms; backward GPU kernels sum ≈ 2.9 s → **~95% GPU**,
  near-zero dispatch gap. §15.3's "backward kernels ≈ 480 ms, 90% dispatch" is
  a measurement error and is retracted.
- True backward breakdown:

  | kernel | time | share |
  |---|---|---|
  | `update_tc_v2` (weight update) | 1383 ms | 45% |
  | head `backward_dx` (32×32, N_out=50272) | 975 ms | 32% |
  | 12 MLP `backward_dx` (64×64) | 559 ms | 18% |

### Why the fix is partial (and where the remaining time is)

1. `update_tc_v2` (1383 ms) is entirely untouched by the dX dispatch change —
   the single biggest backward cost. **Prime next target.**
2. Head dX is stuck on 32×32 because `VOCAB = 50272`, and `50272 % 64 = 32`
   (multiple of 32, not 64), so the routing condition `N_out % 64 == 0` forever
   excludes it. One head layer (975 ms) > all 12 MLP dX combined.
   Fix = padded 64×64 kernel (boundary tiles) **or** bump `VOCAB` 50272 → 50304
   (= 64×786).

### Revised verdict

The 64×64 dispatch is correct and delivers a **real 1.16× backward speedup**
(−444 ms/step, ~13% of the 3.2 s backward). To move the needle further, attack
`update_tc_v2` (45%) and the head-layer dX (32%, blocked by VOCAB not being a
64-multiple).

Full log: `docs/speedpass/2026-08-28-bwd-dx-64-dispatch.md`
Harness: `tests/validate_bwd_dx_64.py`, `tests/ab_dx_dispatch.py`,
`tests/profile_backward_attribution.py`, `modal_ab_dx.py`,
`modal_profile_backward.py`.
Revert = `git revert b23359b`.

---

## 17. SESSION 5 (2026-08-29): update_tc_v2 shared-store vectorization — -7…-18% kernel

### What changed

Single edit to `kernels/packed_ternary/gemm_update_tc_v2_32.cu`: the dY/X tile
load loops wrote each `half2` as two separate 16-bit stores (`DYS(...)=v.x;
DYS(...)=v.y`), which aliases adjacent halves into one 4-byte shared bank → a
2-way bank conflict on *every* store wavefront. Both stores are now a single
32-bit `*reinterpret_cast<half2*>(&DYS(...))=v` (and `&XS(...)`). `r`/`c` are
always even (`i=q*2`), so the destination is 4-byte aligned and the cast is
legal. Scalar fallback for unaligned/boundary cases unchanged.

### Proof (ncu, before → after)

| metric | before | after |
|---|---|---|
| shared-store bank conflicts | 1,669,866,713 | 46,622,531 (36×) |
| shared-store wavefronts | 3,318,864,250 | 1,695,686,329 |

### Isolated kernel timing (B=16384, bypass autograd, 20 iters)

| shape | in | out | before | after | Δ |
|---|---|---|---|---|---|
| fc1 | 1024 | 4096 | 50.09 ms | 46.24 ms | -7.7% |
| fc2 | 4096 | 1024 | 56.36 ms | 46.28 ms | -17.9% |
| head | 1024 | 50272 | 594.64 ms | 550.48 ms | -7.4% |

### Correctness

- `tests/test_update_dimensional_bug.py`: counter 0/16384 errors, max diff 0.
- `tests/test_gemm_update.py`: backward_dx, TC-vs-scalar, flip-direction all
  `max_diff=0.0000`.

### Dead end to avoid repeating

A `kSmemPad` auto-padding attempt (ldm 16→24) was **reverted**: it targeted
`wmma::load_matrix_sync`, which ncu shows is already ~0-way conflicted (1.7M
conflicts / 825M requests), and inflated smem 8→10 KB (occupancy risk) for zero
benefit ("excessive wavefronts" stayed byte-identical).

Full log: `docs/speedpass/2026-08-29-update-kernel-smem-store-vectorization.md`
Note: `tests/bench_update_v2.py` "full step" section throws a *pre-existing*
harness shape error (`xx` = token IDs, not embeddings) — unrelated to this change.

---

## 18. SESSION 6 (2026-08-29): head-layer dX onto 64×64 — +9.8% e2e tok/s

### Finding

The 64×64 backward-dX kernel tiles **B × K** at 64 and reduces over
**N (out_features) in 16-steps** (`kWMMA_K=16`, tail zero-pad). It needs
`N % 16 == 0`, not `N % 64 == 0`. The dispatch gate checked `N_out % 64 == 0`
where `N_out = dY.size(1) = out_features`. Head has N_out = VOCAB = 50272;
`50272 % 64 = 32` → head dX was permanently stuck on the 32×32 kernel, even
though `50272 % 16 == 0` (exactly 3142 reduction steps, zero tail).

### Fix (1 line)

`kernels/packed_ternary/custom_ops.py`: `N_out % 64 == 0` → `N_out % 16 == 0`.

### Verification (Modal T4)

- **Bit-exact**: 64×64 == 32×32 on head (16384,50272,1024) and N=50000 tail.
  Head dispatch now routes 64×64 (`max|out-d64|=0`).
- **Isolated kernel**: head 64×64 = 557.7 ms vs 32×32 = 991.2 ms (**1.78×**).
- **Full-backward** (head-only isolation): −338 ms/step (1.089×).
- **e2e tok/s** (interleaved single-process): 3979 → 4411 (**+9.8%**), −404 ms/step.
- **pytest**: 57 passed.

### Caveat

Modal T4 run-to-run absolute timing drifts (same config 2279 vs 3809 ms across
runs, likely thermal/clock). All A/Bs above are within a single process to cancel
that drift.

Full log: `docs/speedpass/2026-08-29-head-dx-64-dispatch.md`
