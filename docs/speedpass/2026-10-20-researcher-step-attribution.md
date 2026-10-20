# Step attribution inside the cublas backend + ranked heavy-optimization targets — 2026-10-20

Author: Researcher (analysis only — no GPU sessions, no code changes).
Inputs: HANDOFF §26/§28/§29, AUDIT.md, docs/speedpass/2026-10-11-t4-underutilization.md,
docs/speedpass/2026-10-11-counterword-findings.md, and source reads verified this
session (cublas_backend.py, pack_update.py call sites, tests/bench_roofline.py,
tests/bench_update_v2.py, gpu_util.py, train_gigatoken.py, train_step_graph.py).
Skills applied: mle-workflow, agency-performance-benchmarker,
agency-minimal-change-engineer, agency-reality-checker, unlazy (note: the skill
MLForge named as `agency-unlazy` does not exist; the catalog entry is `unlazy`),
agency-codebase-archaeologist.

**Scope guardrail**: perf only. The loss track is separate and still blocked on
real-text data (deterministic step-99 NaN on synthetic tokens) — nothing in this
document touches model quality.

Evidence grades: **[measured]** = same-VM or repo measurement cited;
**[derived]** = arithmetic on measured numbers; **[inference]** = interpretation,
stated as such; **[unverified]** = could not establish from sources.

## 1. Attribution — where the 2,942.8 ms cublas step goes

Per-shape measured rows × per-step counts (13 linear layers: 6×fc1, 6×fc2, 1×head;
B_eff = 16384).

| Bucket | Per-shape ms (fc1 / fc2 / head) | Step ms | % of 2942.8 | Source / grade |
|---|---|---|---|---|
| **Update v2_32 (custom; cublas backend never routes it)** | 46.5 / 46.3 / 596.4 | **1,153.2** | **39.1%** | Phase 0 roofline via `pack_update.update()` directly (tests/bench_roofline.py:82,121), T4 @1155 MHz, cuda-event median-of-20 [measured] |
| cublas fwd | 7.33 / 3.92 / 51.74 | 119.2 | 4.0% | probe 2635d53, same VM [measured] |
| cublas dX | 3.97 / 4.32 / 54.02 | 103.8 | 3.5% | probe 2635d53 [measured] |
| decode (2-bit→fp16) | 0.105 / 0.109 / 1.022, **×2** | 4.6 | 0.2% | Phase 0; cublas_backend.py `forward()` AND `backward_dx()` each call `decode_into` [measured+derived] |
| SubQSA/attention | — | ~45 | 1.5% | AUDIT §2.1, custom era (Modal) [measured, older] |
| LayerNorm+embed | — | ~38 | 1.3% | AUDIT §2.1 [measured, older] |
| GELU | — | ~26 | 0.9% | AUDIT §2.1 [measured, older] |
| CE loss (50272 vocab) | — | ~24 | 0.8% | AUDIT §2.1 [measured, older] |
| **Identified subtotal** | | **~1,513.8** | **51.4%** | [derived] |
| **UNATTRIBUTED** | | **~1,429** | **48.6%** | no measurement exists inside the cublas step [derived] |

Provenance checks run this session (archaeologist pass):
- The 46.5/46.3/596.4 ms update rows come from `pack_update.update()` called
  **directly** — they never went through the custom-op wrapper that `91ec193`
  fixed, so the numbers stand. The suspect harness MLForge flagged,
  `tests/bench_update_v2.py`, imports `custom_ops` and goes through that wrapper,
  so it sat in the broken window; its Part 2 additionally has the AUDIT-noted
  embedding bug. Any update number whose entry point cannot be named is
  [unverified].
- Decode runs 2× per layer per step by construction (fwd and dX both decode).
- The glue rows (~133 ms) are custom-era Modal numbers; the A/B ran power-capped
  at 57–69 W of 70 W with SM clocks 720–1110 MHz (Phase 0 ran uncapped at
  1155 MHz). Small rows barely move with clock state; GEMM rows scale ~linearly,
  so the update share inside the cublas step is a floor [inference].

Roofline cross-check [derived]: GEMM work per step = 10.0 TFLOP (fwd 3.34 + dX
3.34 + dW 3.34). At the measured cuBLAS class (~33 TFLOPS) the floor is ~300 ms;
measured GEMM time is 1,376 ms (223 fwd/dX + 1,153 update). So ~1.0 s of GEMM
over-floor sits in the update kernel, which runs at ~2.9 TFLOPS = 4.4% of the
T4's 65 TFLOPS peak, 1.8% of the DRAM roofline, and 100% occupancy — i.e.
barrier-latency-bound (AUDIT §6.2: 1,024 reduction iterations × 2 barriers, one
16×16×16 MMA per warp per sync window), not bandwidth- or occupancy-bound.

## 2. Ranked targets

### #1 — Re-host the update's dW reduction on cuBLAS; keep v2_32 counter/flip semantics as a thin epilogue

**(a) Measured current cost + source**: update = 1,153.2 ms = 39.1% of the step
(rows above); the same shapes on the same VM via cuBLAS dW = 5.47 / 5.68 / 80.65
ms = 147.6 ms total [measured] — 7.4–8.5× faster per shape. The existing kernel's
tuning space is CLOSED (§26: counter-word variant parked under the 3-strike rule;
compiler-flag space byte-identical SASS; tile/occupancy/counter re-proposals
closed), so this is a re-hosting, not a tune.

**(b) Expected win + basis**: update 1,153 → ~150–165 ms (measured cublas dW rows
+ an epilogue moving ~115 MB ≈ 0.4 ms) → step ~1.95–1.97 s, **−33%**, i.e.
3.37× vs the 6,619 ms baseline [derived from measured components]. Every
component of the replacement has a same-VM measurement already; nothing here is
modeled.

**(c) Verification protocol** (gates, runnable, in order — unlazy shape):
1. Correctness/sign gate FIRST: v2_32 vs the Python reference on the transition
   matrix (tests/test_update_semantics.py pattern) + an explicit
   sign-disagreement count on the 13 prod shapes. Falsifier/stop: any
   transition-matrix failure → abandon the class, do not iterate on the kernel
   (the 3-strike rule's logic applied up front).
2. Isolated kernel A/B: current v2_32 vs cublasGemmEx(fp32-out)+epilogue per
   shape (fc1/fc2/head) under tests/bench_protocol.py — 10-trial paired
   interleaved median, warmup 10, `torch.cuda._sleep` settle, A/A null arm,
   decision rule |Δ| > max(2σ_null, 1%).
3. Whole-step A/B via scripts/colab_step_ab.py extended to route the update
   (NSTEPS=300): median step ms + tok/s + loss trajectory; ship bar ≥5% e2e.
4. Before long runs: the already-queued flip-rate trajectory test (convergence
   has no damping — limit-cycle risk is a separate known gap).

**Minimal-diff scope** (per minimal-change-engineer): touch only the update
dispatch point in `packed_linear.py:163` (route to the new path), one new GEMM
shim (cublasGemmEx fp32-in/fp32-out — torch.matmul on fp16 returns fp16, which
rounds dW and can flip near-zero signs), and one new thin epilogue kernel
(sign → counter → in-place word flip, one thread per W word, no atomics). Do NOT
touch fwd/dX routing, pack_linear's autograd hook, the counter representation,
or any custom WMMA kernel. Counter/flip semantics stay byte-identical; only the
dW computation moves.

**Risk**: sign parity. Different summation order vs WMMA fp32 fragments →
elements near zero can disagree between paths. The counter-word failure was a
sign-parity-class failure whose documented diagnosis (accumulator/fragment
wiring in a copied WMMA loop — [unverified] diagnosis) is deleted by
construction when the front-end is cuBLAS. The gate above quantifies it before
any perf claim. **Do not resurrect the counter-word variant itself** (3-strike,
parked). **Coupling to #2**: if the per-step climb is allocator-side, its
~590 ms/plateau-step cost survives this kernel fix and becomes the
next-largest item — so #2's session runs before #1's implementation numbers are
banked.

### #2 — Attribute the unattributed ~1,429 ms AND the per-step climb — one profiling session, two questions (priority-coupled to #1)

**(a) Question 1 — the unattributed gap.** Identified kernels sum to ~1.51 s vs
the measured 2.943 s step [derived]; AUDIT §2.1's full-step table is custom-era
synthesis on a different VM/clock state [measured, older].

**(a) Question 2 — the per-step climb** [measured series from the A/B run log]:

| backend | step 1 | step 49 | step 99 | step 299 | climb | shape |
|---|---|---|---|---|---|---|
| custom | 4049.8 | 5321.1 | 5473.2 | 5529.0 | +1479.2 ms (+36.5%) | still rising |
| cublas | 2349.0 | 2894.0 | 2966.6 | 2941.9 | +592.9 ms (+25%) | then plateau |

At plateau the climb costs 20.1% of the cublas step and 26.8% of the custom step
[derived] — and both shipped medians (2942.8 / 5443.6) sit AT the plateau, i.e.
the accepted numbers are climbed-state numbers; the step-1 cublas rate would be
~6,950 tok/s if the early rate persisted [derived]. Mechanism read [inference]:
the trend is shared by both backends but shaped differently — a backend-specific
kernel cost would not do that; the leading candidates are (i) allocator/pool
growth or fragmentation from the 10.6 GiB working set, (ii) Python-side ref/GC
accumulation, (iii) power/clock degradation under sustained 70 W load (clocks
already measured wobbling 720–1110 MHz). The climb continues past the step-99
NaN in both arms, which weakens data-dependent kernel-cost explanations (e.g.
flip-rate growth) [inference].

**(b)** Unknown until measured; gap ceiling 1.43 s, climb ceiling ~590 ms/step
(cublas). It gates whether #1 and #3 are the whole story, whether the per-shape
rows describe the step at all, and — per MLForge's coupling argument — if the
climb is allocator-side it *caps* whatever #1 buys: after the update re-host the
climb's share grows to ~30% of a ~1.95 s step, so it can't stay a footnote.

**(c) Verification protocol** — one Colab session, no production-code changes:
1. Instrument the A/B driver (one patch to `scripts/colab_step_ab.py`): per-step
   `torch.cuda.memory_stats()` (reserved, `num_alloc_retries`, fragmentation),
   per-step nvidia-smi clock/power (the console sampler MLForge already uses),
   and a host/device split (cuda events) per step. Run both backends, NSTEPS=300.
2. **Decisive falsifier — mid-run intervention**: at step 150, `torch.cuda.empty_cache()`
   in one arm, no-intervention paired control; compare steps 140–160 pre/post.
   Allocator fragmentation predicts a step-time drop back toward the step-1 rate
   and shrinking `reserved`; no drop + flat memory stats closes the allocator
   class in one reading.
3. Decide from the two traces: monotone clock decline with flat memory →
   power/clock state (recoverable = 0 in software; the fix is measurement
   hygiene — clock-matched windows + the A/A null arm, which bench_protocol
   already has). Memory growth / retries / empty_cache response with flat clocks
   → allocator (software fix: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   or a pre-sizing policy — a one-line env change, then re-run the whole A/B
   under it). Neither + growing host time → Python refs (gc stats next).
4. torch.profiler (profile_memory=True) on the same session for the
   kernel-by-kernel device sum; nsys if available for the host/device split.
   Pair the session with the perfgate baseline regen (same protocol).

**Risk**: none for the measurement — instrumentation is driver-side, and
`empty_cache()` is semantics-neutral. The only way this goes wrong is skipping it
and attributing the climb by reasoning, which is the failure mode the brief
explicitly forbids.

### #3 — CUDA-graph capture (gpu_util.py): verify the ceiling, don't build yet

**(a)** Verified this session: `train_step_cudagraph` (train_gigatoken.py:81) is
a plain eager Python loop — no graph anywhere in it; host dispatch measured at
1.9 ms/step across 175 launches (AUDIT §10) [measured]; the cublas phase showed
100% SM-busy at all 24 live samples, 57–69 W cap [measured, coarse]. So the
launch-overhead ceiling is ~2 ms/step = 0.06% [derived] — while device-bound.

**(b)** ≤2 ms/step if device-bound; potentially large only if #2 lands
host/allocator-bound. Graphs also cut allocator churn via a graph-private pool —
which only pays in that same world.

**(c)** Gated on #2's host/device split; if built, whole-step A/B vs eager under
bench_protocol. Note the open contradiction to settle then: AUDIT §9 marks
whole-step graphing BLOCKED (in-place counter mutation, dynamic control flow)
while tests/test_graph_replay_parity.py passes (capture/mutate/replay bit-exact).

### #4 — Persistent fp16 weight shadow: no (free-rider only)

**(a)** Decode = 4.61 ms/step = 0.16% of the step [derived from measured
per-shape rows]; ~120 MB VRAM cost (fc1/fc2 8 MB each + head 103 MB) against
~4.7 GB headroom [measured headroom].

**(b)** Ceiling 0.16% — no.

**(c)** None needed. If #1's flip epilogue is built anyway, a shadow written on
flip stays coherent (flips are the only W mutation) and rides free — but that is
a footnote to #1, not a slot of its own.

### #5 — Fuse the CE/loss-head backward: no

**(a)** Loss + head scaling ≈ 24 ms/step (AUDIT §2.1, 5 launches) [measured,
older]; grad_logits materialization (1.64 GB fp16) is most of it (~10–15 ms at
T4 bandwidth) [derived].

**(b)** Ceiling ≈ 0.4–0.5% — 10× below the 5% ship bar; and dY has TWO consumers
(head dX AND head update), so fusion forces recomputation or retention.

**(c)** None needed.

## 3. Don't-bother list — the evidence that kills each

1. **Idle-GAP recovery / host-overhead elimination** — GPU measured 100%
   SM-busy at 24 live samples, power-capped 57–69 W of 70 W, clocks 720–1110 MHz
   (§29); host dispatch measured 1.9 ms/step (AUDIT §10). No observable idle gap
   → ceiling ~0.06%. Caveat [reality-checker]: nvidia-smi 100% is coarse
   (residency, not issue-rate; sampled), so this kills the *dispatch-gap* class
   only — which is exactly why #2 exists before anything is declared dead.
2. **Further custom fwd/dX tuning** — the cuBLAS route is bit-exact at all three
   shapes (probe 2635d53, max_abs 0.0) and 1.85× whole-step; fwd/dX already sit
   at the roofline (223 ms measured vs ~200 ms floor at 33 TFLOPS); §26 CLOSED.
3. **Weight-decode path** — 0.105 ms at fc1, 1.022 ms at head (Phase 0); two
   decodes per layer per step = 4.61 ms/step = 0.16%.
4. **CUDA graphs as a build item** — see #3: ceiling 2 ms while device-bound,
   AUDIT §9 blocker, conditional on #2.
5. **CE/loss fusion, fp16 shadow standalone** — see #4/#5: ceilings 0.4–0.5% and
   0.16%.
6. **fc1 cublas fwd tuning** — 44 ms at 18.74 TFLOPS vs fc2's 35.1 (K=1024 is a
   worse cuBLAS tile); ceiling ~0.7% [derived].
7. **All previously closed tuning spaces** — §26 CLOSED: counter-word (3-strike),
   compiler flags (byte-identical SASS), tile/occupancy/counter re-proposals.

## 4. Sequence and gates

Half the cublas step (~1.43 s) has never been measured inside the cublas
backend; of the measured half, 76% is the custom update kernel, which the same
VM already runs 7.4–8.5× faster through cuBLAS. And the run climbs +25%/+36.5%
from step 1 to plateau in both backends — a cost sitting on top of every number
we have. Sequence: **#2 first** (one instrumented 300-step session that settles
both the unattributed gap and the climb, and gates #3), **then #1** (the −33%
heavy item, sign-parity gate before any perf claim), **#3 only if #2 says
host-bound**. If #2 lands on the allocator, the climb fix (~590 ms/step) and #1
(~1,000 ms/step) stack — a ~1.95 s step → ~1.35 s [derived]. Perfgate regen +
the three parked probes stay parked until #2 lands — they all assume the
per-shape rows describe the whole step, which is precisely what #2 tests.
