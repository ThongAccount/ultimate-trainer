# Step attribution inside the cublas backend + ranked heavy-optimization targets — 2026-10-20

Author: Researcher (analysis only, no code changes). Inputs: HANDOFF §26/§28/§29,
AUDIT.md, docs/speedpass/2026-10-11-t4-underutilization.md, source reads
(cublas_backend.py, gpu_util.py, train_gigatoken.py, modal_bench_update_base.py,
tests/bench_update_v2.py, counterword findings), plus a web check on the
T4/cuBLAS roofline claims (T4 sm_75: 65 TFLOPS FP16 tensor-core peak, ~320 GB/s
HBM2 — consistent with the campaign's own numbers).

## 1. Attribution table — where the 2,942.8 ms cublas step goes

Per-shape measured rows × per-step counts (13 linear layers: 6×fc1, 6×fc2, 1×head;
B_eff=16384).

| Bucket | Per-shape ms (fc1 / fc2 / head) | Step ms | % of 2942.8 | Provenance |
|---|---|---|---|---|
| **Update v2_32 (custom, unchanged by cublas backend)** | 46.5 / 46.3 / 596.4 | **1,153.2** | **39.1%** | Phase 0 roofline, T4 @1155 MHz, cuda-event median-of-20 (a3d2117/eb08d515) |
| cublas fwd | 7.33 / 3.92 / 51.74 | 119.2 | 4.0% | probe 2635d53, same VM |
| cublas dX | 3.97 / 4.32 / 54.02 | 103.8 | 3.5% | probe 2635d53 |
| decode (2-bit→fp16) | 0.105 / 0.109 / 1.022, ×2 (fwd+dX each decode) | 4.6 | 0.2% | Phase 0; cublas_backend.py calls decode_into on both paths |
| SubQSA/attention | — | ~45 | 1.5% | AUDIT §2.1 (custom era, Modal) |
| LayerNorm+embed | — | ~38 | 1.3% | AUDIT §2.1 |
| GELU | — | ~26 | 0.9% | AUDIT §2.1 |
| CE loss (50272 vocab) | — | ~24 | 0.8% | AUDIT §2.1 |
| **Identified subtotal** | | **~1,513.8** | **51.4%** | |
| **UNATTRIBUTED** | | **~1,429** | **48.6%** | no measurement exists inside the cublas step |

Caveats on provenance: the glue rows (~133 ms) are custom-era Modal numbers; the
A/B ran on a hosted T4 power-capped at 57–69 W of 70 W with SM clocks 720–1110
MHz (Phase 0 ran uncapped at 1155 MHz) — small rows barely move with clock state,
the GEMM rows scale ~linearly, so the update share inside the cublas step is a
floor, not a ceiling. The update rows predate `91ec193`, which fixed the custom-op
wrapper that the whole-step training path (and bench_update_v2.py) was calling;
they were measured healthy via the Phase 0 probe, but re-confirming them on the
fixed op path costs one Colab session alongside the perfgate regen.

Roofline cross-check: total GEMM work/step = 10.0 TFLOP (fwd+dX 6.67 + dW 3.34).
At the measured cuBLAS class (22.6–35 TFLOPS) the floor is ~300–330 ms; measured
GEMM time is 1,376 ms (223 fwd/dX + 1,153 update). The update kernel runs at
~2.9 TFLOPS = 4.4% of the T4's 65 TFLOPS peak and 1.8% of the DRAM roofline
(Phase 0) — simultaneously far from bandwidth and at 100% occupancy, i.e.
barrier-latency-bound (AUDIT §6.2: 1,024 reduction iterations × 2 barriers, one
16×16×16 MMA per warp per sync window).

## 2. Ranked targets — measurement, expected win, risk

### #1 Re-host the update's reduction on cuBLAS; keep v2_32 counter/flip semantics as a thin epilogue
- **Measurement that justifies it**: update = 1,153.2 ms = 39.1% of the step
  (rows above); the same shapes on the same VM via cuBLAS dW = 5.47 / 5.68 /
  80.65 ms = 147.6 ms total (7.4–8.5× faster per shape). Occupancy, bandwidth and
  flop walls are all absent (Phase 0 + AUDIT §4), and the tuning space around the
  current kernel is CLOSED (§26: compiler flags byte-identical SASS; tile /
  occupancy / counter re-proposals closed; counter-word variant parked under the
  3-strike rule — do not resurrect it).
- **Expected win with its basis**: update 1,153 → ~150–165 ms (measured cublas dW
  rows + a sign/counter/flip epilogue that moves ~110 MB ≈ 0.4 ms) → step
  ~1,960–1,975 ms, **−33%**, i.e. 3.37× vs the 6,619 ms baseline (from 1.85×).
  Basis is measured, not modeled: every component of the replacement has a
  same-VM measurement already.
- **Risk**: sign parity. Needs cublasGemmEx with fp32-in/fp32-out (a small C++
  shim — torch.matmul on fp16 inputs returns fp16, which rounds dW and can flip
  near-zero signs). Summation order differs from WMMA fp32 fragments → elements
  near zero can disagree between the two paths. The counter-word failure was a
  sign-parity-class failure whose documented diagnosis (accumulator/fragment
  wiring in a copied WMMA loop — unconfirmed) is deleted by construction when the
  front-end is cuBLAS. Gate: v2_32-vs-Python-reference sign comparison on the
  transition matrix (tests/test_update_semantics.py pattern) + an explicit
  sign-disagreement count on production shapes + the already-queued flip-rate
  trajectory test before any long run. No counter-semantics change.

### #2 Attribute the unattributed ~1,429 ms — measurement item, highest information value
- **Measurement that justifies it**: identified kernels sum to ~1.51 s vs the
  measured 2.943 s step; AUDIT §2.1's full-step table is custom-era synthesis on
  a different VM and clock state; and the custom arm's per-step time ROSE during
  the run (4.0 → 5.5 s), which is growth — fragmentation or clock/thermal drift,
  both consistent with the measured power-cap wobble (57–69 W, 720–1110 MHz).
- **Expected win**: unknown until measured; ceiling 1.43 s. It gates whether #1
  and #3 are the whole story, and whether the per-shape rows describe the step at
  all.
- **Method** (one Colab session, ~30 min, no code): torch.profiler on 50 cublas
  steps with profile_memory=True and per-step nvidia-smi clock/power logging +
  torch.cuda.memory_stats() (alloc retries, fragmentation); nsys if available
  (it does not need the gated hardware counters) for the host/device split.
  Compare the profiler's device-time sum against the per-shape rows at matched
  clock state. Do it together with the perfgate baseline regen.
- **Risk**: none (measurement only).

### #3 CUDA-graph capture (gpu_util.py) — verify the ceiling, don't build yet
- **Verified this session (not repeated from the brief)**: `train_step_cudagraph`
  (train_gigatoken.py:81) is a plain eager Python loop — no graph anywhere in it;
  host dispatch is measured at 1.9 ms/step across 175 launches (AUDIT §10) =
  0.06% of the step; the cublas phase sampled 100% SM-busy at all 24 live samples
  (57–69 W cap) → no idle-dispatch gap was observable at 3 s sampling.
- **What that does and doesn't prove**: it bounds the launch-overhead win to
  ~2 ms, but nvidia-smi "100% SM" is coarse (residency, not issue-rate; sampled),
  so sub-3 s idle gaps and host-side stalls are NOT excluded — the decisive
  number is #2's host/device split. Graphs also remove allocator churn via a
  graph-private pool, which only pays if #2 lands on fragmentation. Note the
  open contradiction: AUDIT §9 marks whole-step graphing BLOCKED (in-place
  counter mutation, dynamic control flow) while tests/test_graph_replay_parity.py
  passes (capture/mutate/replay bit-exact) — settle it only after attribution.
- **Expected win**: ≤2 ms/step if device-bound (the likely world); large only in
  the allocator/host-bound world #2 might reveal.
- **Risk**: currently zero (verification cost only); building before #2 risks
  the AUDIT §9 blocker.

### #4 Persistent fp16 weight shadow (kill the decode pass) — dead as a standalone
- **Measurement that justifies the verdict**: decode runs 2× per layer per step
  (cublas_backend.py forward() and backward_dx() each call decode_into) =
  4.61 ms/step = **0.16%** ceiling; ~120 MB VRAM (fc1/fc2 8 MB each + head
  103 MB) against ~4.7 GB headroom — affordable but pointless at 0.16%.
- It rides along FREE inside #1's flip epilogue if ever wanted (flips are the
  only W mutation, so a shadow written on flip stays coherent). Do not spend a
  slot on it standalone.

### #5 Fuse the CE/loss-head backward — dead
- **Measurement that justifies the verdict**: loss + head scaling ≈ 24 ms/step
  (AUDIT §2.1, 5 launches); grad_logits materialization (1.64 GB fp16) is most of
  it (~10–15 ms at T4 bandwidth). Fusion ceiling ≈ 0.4–0.5% — 10× below the 5%
  ship bar — and dY has two consumers (head dX AND head update), so fusion forces
  recomputation or retention.

### Pre-empted "no" list
fc1 cublas fwd tuning (44 ms at 18.74 TFLOPS vs fc2's 35.1 → 0.7% ceiling);
compiler-flag space (CLOSED §26, byte-identical SASS); tile/occupancy/counter
re-proposals (CLOSED §26); counter-word resurrection (3-strike, parked).

## 3. Bottom line
Half of the cublas step (1.43 s) has never been measured inside the cublas
backend; of the measured half, 76% is the custom update kernel, which the same
VM already runs 7.4–8.5× faster through cuBLAS. Sequence: #2 (one profiling
session, settles the other half and gates #3) → #1 (the −33% heavy item, gated
on sign parity) → #3 only if #2 says host-bound. Everything else is ≤0.5%.
