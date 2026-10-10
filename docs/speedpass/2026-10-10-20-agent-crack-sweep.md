---
tags: [project/ultimate-ai-model, topic/cuda, topic/speedpass, topic/correctness]
date: 2026-10-10
status: active
parent: docs/speedpass
---

# 2026-10-10 — 20-Agent Exhaustive Crack Sweep (C13)

Mission: 20 independent read-only investigations into every remaining crack in the
trainer at `357c33b` (post-C12). Orchestrated in 3 waves (3 salvaged from the dead
session's ZIP, 7 wave-1, 10 wave-2 after a model-pool correction). All 20 delivered.

This file is the deduplicated, contradiction-resolved synthesis. Raw payloads:
17 wave-1/2 JSON reports (persisted at `/tmp/sweep2/` during the session; re-pullable
via `agent://<id>` for ~5 min after settlement).

## Executive summary

The sweep **inverts the campaign's working priority**. Kernel micro-tuning is dead
(confirmed for the nth time, now with roofline arithmetic), but two **structural
non-tile levers** survive everything falsified, and the **correctness infrastructure
has a hole bigger than any perf item**: the production update kernel has *zero
direct semantic tests* — the suite that "covers" it silently validates a different
algorithm (v3) at different dims, and the only probe gate is old-vs-new parity,
which any shared bug survives.

Biggest single finding: **fwd head runs at 3% of TC peak (2.0 TFLOPS), dX at 5%,
update at 1.8% of DRAM roofline and 12% of TC peak** — all far from every roofline.
There is no bandwidth wall, no occupancy wall, no flop wall. The machine is idle
waiting on barriers. The gap is scheduling structure, and dX is the same-hardware
existence proof that 1.65x is reachable.

## Results: all 20 investigations

| # | Investigation | Verdict | One-line result |
|---|---|---|---|
| 1 | Correctness audit (salvaged) | BUG (latent) | Odd-`in_features` tail handler unreachable → last W column never trains; int16 counter wrap; dX epilogue refactor verified correct but rests on elided `__syncwarp` |
| 2 | Parity audit | BUG (test-suite) | `test_update_tc_flips_bits` never reaches the TC kernel (N=4 → scalar fallback); v3 can silently occupy the v2 slot; threshold≥32768 int16 truncation cliff |
| 3 | Convergence | BUG (theoretical) | No damping/annealing exists at all: noise coordinates = flip-every-~2T-steps limit cycle; late-training SNR drop worsens it |
| 4 | Counter semantics | BUG (latent) | int16 wrap at ±32768 → spurious flip; no clamp anywhere; 1-line fix (~0.1% cost). Reset-to-0 consistent; no cross-race (separate arrays) |
| 5 | Weight-flip gate | BUG (gate) | Probe gate vacuous in the strong sense: old-vs-new parity only, `flips>0` not `flips==expected`; wrong-column/sign bugs shared by both arms pass |
| 6 | CUDA memory behavior | PROMISING | fwd/dX gap = 2x iterations × fixed barrier drain (bytes PROVABLY symmetric: 26.3B halfs, 206M HMMA each); update at 1.8% DRAM roofline; fwd spill epilogue caps occ at 3/SM (dX-pattern fix → 4-6) |
| 7 | Generated assembly | PROMISING | `decode_ternary` kLUT compiles to LOCAL-MEMORY traffic (STL+LDL.S8) in all 3 kernels; fwd loop = 430 instr/K16 vs dX 223 (1.9x overhead); 16x W-word LDG redundancy confirmed in SASS |
| 8 | Compiler flags | NO-FINDING | -O2/-O3/vectorization → byte-identical SASS everywhere; `--use_fast_math` on update only changes FSETP→FSETP.FTZ (semantic, zero gain). Flag space closed |
| 9 | Numerical semantics (salvaged) | BUG (minor) | `pack_tensor` (half-even) vs `pack_row` (`roundf`, half-away) disagree at exact .5; TC trio bitwise deterministic; fused path is not (fp16 atomicAdd) |
| 10 | Shape/tail (salvaged) | FALSIFIED | VOCAB 50272→50304 = net regression (+0.019%); premise wrong (fwd grid identical either way; dX/update have no tail) |
| 11 | half2 opportunities | PROMISING | Top-2: fwd X load `:118`, dX dY load `:82` (both hot-loop, guard pattern already shipped on update); stores negligible; realistic 2-3.5% e2e combined, needs 10-trial protocol |
| 12 | Counter 4-wide | FALSIFIED | Counters are flat int16[] (not packed); one CAS can't express 4 mixed-direction transitions; CAS traffic = 0.25% of step — irrelevant |
| 13 | Fwd/dX bottlenecks | PROMISING | fwd warp-private spill epilogue (20→8KB, occ 2-3→4-6) + n-outer rasterization swizzle (X DRAM 25.7GB→32MB) — both non-tile changes, ~3-5% e2e each |
| 14 | E2E profiling | MEASUREMENT-NOISE | dX 965ms is residual-DERIVED, never measured; isolated sums say 1010ms (45ms unaccounted); 'other 208ms' loosest of 3 estimates; Amdahl: fwd-gap closed = −10.2% e2e (only >7% candidate) |
| 15 | Benchmark methodology | BUG (protocol) | 3-trial min-of-3 → ±4-6% CI; a 1% win is UNDETECTABLE; fc2 +3.28% fully consistent with noise; F1 (-14%) survives scrutiny; fix = 10 trials + median + sleep barriers |
| 16 | Memory allocation/layout | NO-FINDING | Hidden casts ~0.16% e2e (inside noise); fwd/dX W addressing identical (L2 reuse possible but temporal distance kills it); probe clones symmetric → no fc2 bias; alignment clean |
| 17 | Alternative update algos | 2×PROMISING, rest dead | Split-K falsified (sign(Σ)≠Σ sign); fusion blocked (deps); persistent kernel dead (occ already 100%); register-only dW accumulation = 5-15% kernel candidate; deferred flips = experimental branch only |
| 18 | Adversarial testing | BUG (coverage) | B=1/K=1/N=1, threshold=0/32767, counter saturation, W ±1 boundaries, all-zero/all-same-sign grads — never tested anywhere; prod-dim paths unexercised by design |
| 19 | Test infrastructure | PROMISING | Gate validates v3 semantics while prod runs v2 (dims 16-63 dispatch trap); every CUDA test silently returns on CPU (66 passed ≠ 66 ran); zero sanitizers ever; no perf gate exists; build_kernels.py omits all *_32/v3/fused kernels |
| 20 | Red team | PROMISING | Roofline confirms latency-bound (1.8% DRAM, 12% TC) but lever class = prefetch/pipelining, never probed; C12 rejection statistically UNDECIDABLE but moot (0.7% ceiling < 1% bar); single best hypothesis = cp.async/double-buffer on update's 1024-iter loop |

## Verified bugs, ranked by severity

1. **#2/#19 (test-suite, HIGH, live now):** the production update kernel's flip/
   direction/reset semantics have ZERO direct test coverage. The gate tests run at
   dims 16-63 which dispatch to **v3** (a different algorithm: magnitude-scaled
   delta) while production runs **v2**. Any future update change (C12 retry,
   register-dW, pipelining) ships with no semantic gate. The one direct test
   (`test_update_dimensional_bug.py`) is excluded from `colab_pytest.py`.
2. **#3 (convergence, HIGH, by design gap):** no annealing/threshold schedule/counter
   decay exists. Late training = oscillation risk. `reset_counter()` exists, never
   called. Needs the flip-rate-trajectory test before any long run is trusted.
3. **#1/#4 (latent, MEDIUM):** odd-`in_features` tail unreachable (last column frozen
   — dormant at even prod dims); int16 counter wrap (dormant at threshold=32;
   becomes live at threshold≥32768 where truncation makes ALL negative counters
   flip); `pack_tensor`/`pack_row` rounding disagreement at exact .5.
4. **#15 (protocol, MEDIUM, live now):** every timing decision made at ±4-6% CI.
   Both shippings (F1, dX epilogue) survive; several falsifications (C9a's +1.9-2.7%,
   C12's spread) are within their own noise — some "falsified" verdicts are actually
   "undecided, below bar".

## Contradiction resolved

Three agents attacked the fwd-dX gap independently. FwdBwd2's byte-count
(W-reload 12.9GB vs 0.9GB) is overturned by the two exact accountings (CudaMem2,
RedTeam2): dX re-reads W per (b,k) pair symmetric to fwd's per-(m,n) — **26.3B
halfs and 206M HMMA in BOTH kernels**. The surviving mechanism (2-of-3 majority,
arithmetic-checked): fwd runs **2.0x the outer-loop iterations** (K16 vs N32
step) and in these single-buffered lockstep kernels (0 LDGSTS in all SASS), the
**fixed per-iteration barrier/latency drain dominates**. Occupancy (fwd 3 CTAs/SM
vs dX 5) compounds it. Actionables converge and compose: warp-private spill
epilogue (occupancy) + rasterization swizzle (L2 locality) — both non-tile changes,
both absent from every falsified probe.

## Remaining untested hypotheses, ranked by expected value

| Rank | Hypothesis | Source | Mechanism | Ceiling (e2e) | Cost | Go/No-Go |
|---|---|---|---|---|---|---|
| 1 | **fwd warp-private spill epilogue** (dX pattern) | #6/#13 | smem 20→8KB → occ 3→4-6 CTAs/SM; latency hiding in 64-iter K-loop | 3-5% realistic (fwd = 26% of step) | small (mirror of shipped dX refactor) + 1 Colab cycle | **GO** |
| 2 | **fwd n-outer rasterization swizzle** (grid-stride) | #13/#20 | co-resident CTAs share the same X tile (25.7GB→~32MB X DRAM) + W L2 locality | 2-5% e2e (capped ~80ms of 831 head) | 3-line kernel change, bit-identical | **GO** (same Colab cycle as #1) |
| 3 | **update cp.async/double-buffer pipelining** | #20 (with #6/#7 corroboration) | 1024 serial iters at 2200 cycles exposed latency each; prefetch i+1 during mma_i | 3-7% e2e (update = 36% of step) | 1-2h + 1 cycle; real occ tension (8→6 blk/SM) | **GO** (one probe; falsifier pre-registered) |
| 4 | **Bundle: kLUT→branchless + W-word load dedup-reg + half2 X/dY loads** | #7/#11 | 430→~250 instr/K16 on fwd; ~960 redundant LDG issues removed; L1 request halving | 1-3% e2e bundled | medium; MUST use 10-trial protocol | **GO bundled** (single fwd variant, one cycle) |
| 5 | **Benchmark protocol upgrade** (10 trials, median, sleep barriers, A/A null) | #15 | CI ±4-6% → ±1.5% | 0% direct; enables 1-3% detection | ~30 lines | **GO first** — prerequisite for 1-4 |
| 6 | register-only dW accumulation (update) | #17 | avoids smem dW round-trip | 1.3-5% kernel | prototype risk: reg pressure | HOLD — after #3 verdict |
| 7 | deferred flips (batched counter flush) | #17 | changes flip timing semantics | ~1.3% e2e | + convergence validation | **NO-GO** for main branch (experimental only, needs #3's test first) |

**Falsified / closed by this sweep:** counter 4-wide (#12), split-K update (#17),
VOCAB 50304 (#10), compiler flag space (#8), persistent update kernel (#17),
allocator/layout levers (#16), host/hidden-copy levers (#16, confirms C9b),
CAS-traffic levers (#6/#12 concur).

## Correctness/convergence test plan (GPU-gated, one Colab session)

1. **Reference-transition gate** (~20 lines, from #5): seed W=0, counter=T-1 at
   known (n,k); craft X/dY for known-sign dW; assert bit-exact W transition AND
   counter reset against a Python-computed expectation. 4 cases: (grad±) × (W=-1,0,+1).
2. **Fix the dispatch trap** (#2/#19): add `test_update_dimensional_bug.py` +
   `test_update64_dimensional_bug.py` to `colab_pytest.py` TEST_MODULES; add a
   hard-fail session fixture when CUDA is present but any prod kernel fails to load.
3. **Boundary tests** (#18): B=1/K=1/N=1 across trio; threshold=0 and 32767;
   all-zero and all-same-sign gradients on update/dX; counter at ±32767; B=17 vs
   B=16-with-zeroed-17th-row parity.
4. **Counter clamp** (#4): one-line saturating clamp before threshold check in
   v2_32 (+ the dead v3_32 for hygiene); re-run parity gate.
5. **Sanitizer one-shot** (#19): memcheck on fused, initcheck on v2_32 at odd
   dims (N=40,K=48), racecheck on v2_32, synccheck on dX epilogue. ~15 min.
6. **Flip-rate trajectory test** (#3): 1k synthetic steps with true-gradient+noise
   mix; assert flip rate decays as noise share drops; catches the limit cycle.
7. **Perf gate** (#19 design): `colab_perfgate.py` — 3 fixed shapes, warmup 10,
   10 trials, median, vs committed `tests/perf_baseline_t4.json`, WARN >2%, FAIL >5%.

## Benchmark plan

Per #15: 10 paired trials + median (not min), warmup 10, `torch.cuda._sleep`
between trials, single-process interleaved A/B/A/B, plus one **A/A null arm** to
empirically calibrate σ_diff before believing any delta. Decision rule:
`|mean Δ| > max(2·σ_null, 1%)`. Ship bar unchanged: >1% e2e. Expected CI: ±1.5%.

## Go/No-Go summary

- GO (order matters): ① protocol upgrade → ② fwd epilogue + swizzle (one cycle,
  parity + 10-trial) → ③ update pipelining probe (pre-registered falsifier) →
  ④ fwd instruction bundle (kLUT + W-dedup-reg + half2) if ② shows head room.
- GO in parallel (no GPU needed): test-plan items 1-2 (reference gate + dispatch
  fix) — these close the correctness hole that every future kernel change needs.
- NO-GO: deferred flips, split-K, counter 4-wide, flag changes, VOCAB bump,
  any tile/occupancy/regsweep re-proposal (now closed by roofline + SASS, not
  just by null results).
- RECLASSIFY: C12 falsified → **MEASUREMENT-NOISE (undecided)** — moot at 0.7%
  ceiling; do not re-measure.

## The single best next action

**Ship the test-suite fixes (items 1-2 of the test plan) and the benchmark
protocol upgrade together — they need zero GPU time to write and one Colab
session to validate — then run the fwd epilogue + swizzle probe as the first
10-trial measurement.** The correctness hole is the only finding that makes
every other result unsafe to act on; the fwd probe is the highest-EV perf item
(3-5% e2e realistic, mechanism triangulated by three independent agents, and
its fix is a mirror of an already-shipped, already-validated refactor).
