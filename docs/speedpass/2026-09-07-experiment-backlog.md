# Experiment Backlog — Ultimate Trainer @ 2026-09-07

**Status**: Discovery only. No production kernels modified. Branch
`chore/speedpass` remains at `e7c4521` (all docs/sessions).

**Scope**: PackedTernaryLinear WMMA kernels (fwd TC64, bwd dX TC64, update
TC32) + SubQSA fused combine + sparse ternary, all targeting Modal T4 sm_75.

**Roadmap source**: `2026-09-07-deep-audit.md` → executed in
`2026-09-07-session12-roadmap-execution.md` → `2026-09-07-session12b-ccombined...`
(combined fwd test). All measured findings final.

---

## Current bottleneck model

Three production WMMA kernels each sit ~2.5–3 GFLOP/ms on T4 (T4 TC peak
≈ 65 GFLOP/ms ⇒ ~4 % utilisation). All three share the same skeleton: strict
serial `load-to-smem → __syncthreads → MMA(acc-2D-fixed) → __sync` ×
reduction_iters. This sync chain has been tested at three independent slices
(16, 32, 64) across two different kernels; every widening landed at ≥20 KB
smem ⇒ occupancy collapsed, and the fixed 2KB/frag pipeline constraints have
been falsified as the dominating bottleneck, **except for forward K16 vs
K32 collision between the smem-limited change and the separate smem-store
transposed-write slowdown loops**.

**What is proven**:

- sync count is not the limiter on fwd (halving syncs at K32 lost 31% even
  with headroom-matched occupancy, 5 blocks/SM)
- dX K32 sync-halving won because dX never writes transposed into smem
- update TC32 at 8 blocks/SM and 100% occupancy — both further occupancy and
  less sync both lose
- CPU/dispatch/memory movement = 0 % of step time

The three kernels sit in a latency well defined by the load→smem→MMA chain;
each experiment has only moved the bottleneck, not reduced serialized latency.

## Falsified hypotheses (do NOT revisit)

| Experiment | Measured result | What it actually proves |
|---|---|---|
| fwd K32 | +31 % at 5 blocks/SM | fwd smem-store transpose bugs dominate; not occupancy |
| fwd K32 + 4 KB spill (12b) | +32 % head, 5 blk/SM | the fwd store is layout-blocked, not occupancy-starved |
| fwd 16 KB spill removal (E4-only) | +2 % at 7 blocks/SM | occupancy at K16 is saturated; not helpful |
| dX K64 | +20 % at 3 blocks/SM | smem budget cliff at 20 KB |
| update tile dedup | +18 % slower | duplicate loads absorbed by L1/L2; redundant-is-fine |
| update kSub=4 | 2.3× slower | occupancy collapse at 20 KB smem |
| update K32 (kK2) | <2 % gains in isolated | update is truly at a sync chain local optimum |
| SubQSA Phase 8 coalesce | not measured (fwd not in speedpass path) | can't measure → shouldn't tune |
| Compiler flags -O3 fast_math | 0 % dX/upd | no effect |
| Half-bit decode LUT | 1.06× head on fwd scratch probe | decode is not the bottleneck |
| Fusing dX+update | invalid on dataflow | dX is *not* consumed by update |
| GPU residency | 0 % e2e | already 0 memcpy/step |

## Tier A — high confidence, short, isolated experiments

| # | Kernel | Hypothesis | Mechanism | Expected effect | Risk |
|---|---|---|---|---|---|
| A1 | fwd TC64 | W_smem store transpose is the failing K32 path. Re-write W tile row-major (`W_smem[r][c]`), and mirror b_frag `ldm`. Probe at K16 first, then apply K32. | Change 6 lines in W fill loop; update b_frag load path. | −10..−30 % kernel e2e head | medium — fragment layout change |
| A2 | all three | spill[4][256] float store allows __half2float conversions to be the only global-write double step. Direct float store per-frag (register→global) slices the need for a smem slot at all. | `wmma::store_matrix_sync` never required; c_frag → float convert in-register, direct write to Y. | −3..−8 % kernel | low |
| A3 | update | kSub=2 (not 4), unrolling only the X/dY load rollout, while keeping 16-batch inner loop, i.e. 2 blocks/loop instead of 4. | smem 8→12 KB, regs +~6, sync same count. | −1..−3 % kernel | low |

## Tier B — plausible, requires measurement

| # | Kernel | Hypothesis | Mechanism | Expected effect | Risk |
|---|---|---|---|---|---|
| B1 | fwd TC64 | The W-smem store bank-conflict stride of 64 was patched at K32 to stride-64 (max conflict). Single-index layout: change to W_smem[n][k] row-major (r = row = n innermost) then b_frag from W_smem[r][ks*16] becomes... swap ldm roles between a_frag and b_frag. | Design b_frag as `col_major`, load at `W_smem[n][kK2]` with `ldm = kK2`. | if A1 passes, fwd K32 becomes profitable | high — fragile layout logic against the 113b40b fix |
| B2 | dX TC64 | K64 took dX to 3 blocks/SM (20 KB); keep K32 tiles but add a second r0-block per warp, (2 outer loops * K32), giving 4 MMAs/sync at 12 KB + improved dY reuse, without hitting the smem drawbridge. | split fragment blocks differently, same load buffers, 2 outer-step structure | −3..−8 % head on dX | medium |
| B3 | update | Vectorized half2 loads (already applied) may leave 32-bit transactions. Look for 8-bit/16-bit remaining strided loads in the actual packed W (uint32) load path and pack them to uint4. Only if SASS shows scalar lds on the dX. | ncu or SASS diff; 2-bit pack currently 4 lanes × 32-bit — should already be coalesced | −1..−3 % kernel | low–medium |
| B4 | fwd TC64 | fuse W decode into the dY load (only decode the W you actually need) — decode16 × 16 pack in `int2` registers then push a combined half2 into smem vectorized. Remove `decode_ternary(int)`→`__int2half_rn(t)` scalar step. | replace per-element decode+convert with 8-lane table lookup + vectorized half2 store. | −3..−6 % kernel | low–medium |

## Tier C — speculative architecture experiments

| # | Kernel | Hypothesis | Mechanism | Expected effect | Risk |
|---|---|---|---|---|---|
| C1 | fwd TC64 | Warp specialization: 1 warp produces tiles, 3 warps MMA; remove cross-warp sync entirely for producer. | split block into producer (dY)/consumer; requires cp.async → T4 sm_75 cannot, so value doubtful. | unknown | high — blocked at HW |
| C2 | fwd TC64 | Fold the SubQSA-style decode-table into a fragment mode — fully decode a [32×16] W tile into W_smem with *one* half2 store per 16-word→2-tiles-worth row. | Register-level LUT → 8 halfs → half2 stores. | theoretical −10 % on decode step | high, thick redesign |
| C3 | all | Skip accumulator-conversion spills entirely: accumulate fp16 in the MMA, convert per-frag in-register (if sm_75 WMM supports half acc). | `wmma::fragment<accumulator,…,half>` — sm_75 WMMA requires float acc; rejected *at design*, no test. | −5 % norm store | impossible at HW |

## Tier D — moonshots

| # | Kernel | Hypothesis | Mechanism | Risk |
|---|---|---|---|---|
| D1 | fwd TC64 | Row-major W_smem layout (no transpose). Reload + fragment-setting change. Full rewrite. | Reimplement from dX's row-major pattern. | high — lost the 3× session tried it |
| D2 | update TC32 | Different batch-splitting strategy: split-K across *warps* with atomicAdd into dW_float_smem instead of smem tiles → same semantics, less serial loop. | distribute batch slices per warp; atomicAdd float smem. | medium–high — atomics on smem |
| D3 | head layers only | For head (out=50,272), negligible tail; replace global sync path with 2D tile rasterization into a 2-stage pipelined direct global→fragment mapping. | skips rematerializing through smem entirely. | very high |

## Dead ends (do NOT revisit)

| Experiment | Evidence | Why it can't work |
|---|---|---|
| fwd/dX more sync halving (K64) | dX K64 +20 % at 20 KB smem | reviving requires exit smem budget cap |
| fwd/dX occupancy > 5 blocks/SM | fwd E4 7 blocks/SM +2 %; fwd E1+12b at 5 blocks/SM +31 %/+32 % | latency is *not* occupancy-bound at this range |
| fwd W decode LUT/half-bit | 1.06× scratch probe | decode is ~6 % of work |
| SubQSA Phase 8 coalesce | fwd+nnot even in speedpass profile baseline | optimizing unmeasured path |
| Persistent kernel | sm_75 kernel-launch is not in the bottleneck | zero CUDA memcpy already |
| cudaMemcpy / GPU residency | measured 0 % gain | the whole pipeline is GPU-resident |
| Compiler flags | measured -O3+fast_math change nothing | already maxed |
| update tile dedup | overlap loads are absorbed in L1 | duplicated traffic is free on T4 |
| update K32 (kK2 on 32×32) | from audit session-2 findings | update is fully occupied |
| fp16 (half acc) accumulator | sm_75 requires float WMMA acc | HW |
| update 64×64 WMMA bug fix | already fixed `1baa097`, but bench-neutral | documented — don't re-test, don't wire |

## Recommended test sequence

1. **A1** — immediate probe of the hypothesis that has *strongest direct evidence*: fwd W_smem transposed-store fixes. If A1 moves fwd (any direction), then:
2. **B1** — after A1, apply K32 widening; fwd is currently by far the biggest-ticket item (head layer = 51.7 %).
3. **A2** — remove the W_smem spill indirect store and store fragment outputs directly.
4. **B2** — dX K32 × (outer unroll 2 = 4 MMAs/sync at 12 KB).
5. **A3** — update kSub=2 at the same smem budget.

Every one of these has a measurable hypothesis, a clear failure mode, and is
isolated. Stop each experiment as soon as a single measurable result
confirms or falsifies the hypothesis.

## Unknowns requiring dedicated experiments

| Question | Experiment |
|---|---|
| Could exact extra smem absorb K32 without smem-store transposition? | A1: none, then attempt |
| What's the *specific* smem latency + bank conflict cost on T4? | micro-bench on pattern (write-then-read transposed tiles) |
| Is the global dX pipeline (autograd hook return) costing anything? | instrument `PackedTernaryLinearFn.backward` directly with cuda events |
| Does the update kernel's counter/flip phase ever become hot when batch is the only scale? | run B=8,192 vs B=16,384 |
| dX bwd K64 + spill re-write change — would spill elimination absorb K64 too? | combine the E4 spill-fix (4 KB) with dX K64 tiles (20 KB) → 24 KB — probably still not enough; test if B2 works |

## Newly discovered bottlenecks

1. fwd still has the dominant step cost (1,371 ms × 13 calls 37.6 % of GPU time)
2. fwd dX, forward both at ~2.5–3.2 GFLOP/ms — this consistency across different loop shapes suggests a common bound = load-store serialization on the smem tiles, not compute, not occupancy, not sync count
3. Head layer = 51.7 % of total — at 50,272 outputs the N×K grid is huge; scattering the K padded N dim to K multiples ((50272+64−1)//64×64 = 50272 the de-facto multiple 64 hits N ends fine but float padding is essential)
4. The entire branchable fused (subqsa_combine) path is currently dead in the speedpass benchmark — no evidence it's in the e2e loop at all

## Key insight (new this session)

**The fwd kernels, dX kernels, and update kernels all share a common load-store
pattern chain that none of the previous A.B. benchmarks ever isolated
independently, and every experiment that harmed one class of overhead
(occupancy, sync rate, latency chain) collapsed under the complementary
interaction: latency chain + layout transpose drive independent penalties.**
