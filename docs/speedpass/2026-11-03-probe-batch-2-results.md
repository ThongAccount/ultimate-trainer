# Probe batch 2 results (2026-11-03)

Session `ut-probe2c`, branch `feat/probe-batch-2` @ `69241bc` (expect-commit
matched), bundle `kernels-probe-batch-2` (CI run 41691777179, green; bundle
commit 4c0581c — the branch delta after that is tests/-only, warned + ignored
per the ABI contract). Pre-registration: `2026-11-03-probe-batch-2-prereg.md`.
Artefacts: `results/probe_batch2.69241bc7.{json,log,*.log}` (provenance block
present; verdict `fail` because flip_b3/b4/perfgate_regen returned rc=1 — all
three are RESULTS, not plumbing failures).

Gates: smoke PASS (both variants bit-exact at B=128,K=64,N=64 — after two
compile/indexing fixes: kK4 hoisted to file scope; uint4 loads one 16 B
word-group per W row), pytest 103 passed, perfgate_check rc=0.

## #10 dX K64 unroll-2 (kK2=32→64, 16 MMAs/sync) — CLOSED NEGATIVE, not shipped
- parity fc1/head: EXACT vs production kernel
- fc1: A=35.21ms B=46.24ms **+31.35%** (σ_null 1.42%) → LOSS
- head: A=507.92ms B=571.72ms **+12.56%** (σ_null 0.67%) → LOSS
- Same direction as the closed "dX K64" dead end; the wider smem window
  (32 KB) drops occupancy and costs more than the saved syncs. Do not re-propose.

## #11 fwd uint4 W loads (K-window 64, one 16 B load per row) — CLOSED NEGATIVE, not shipped
- parity fc1/head: EXACT
- fc1: A=19.99ms B=80.26ms **+301.40%** (σ_null 2.95%) → LOSS
- head: A=319.41ms B=917.91ms **+187.38%** (σ_null 0.54%) → LOSS
- Same dead end as "fwd K-widening" (+31%): the 64-wide K window destroys the
  load/MMA overlap the K16 loop lives on. Scalar 32-bit W loads stay.

## #12 layer-0 clone cost — CLOSED (negligible)
- `X.detach().clone()` [16352,1024] fp16: **0.312 ms = 0.053%** of the 593.98 ms
  shipped step; [16352,4096]: 1.182 ms = 0.199%. Not a target.

## #15 small-dims fwd fallback parity — PASS
- TC64 kernel vs fp32 reference, max rel err: 64³ 4.87e-4, 48³ 4.88e-4,
  16×1024×4096 4.88e-4, 80×96×112 4.88e-4, 16352×1024×4096 9.54e-4 — all < 5e-3.
  Fallback path is correct at small/edge dims; no action.

## #6 fused CE + scale — MEASURED, decision deferred
- At logits [16352,50272] fp16: eager CE **24.88 ms** (σ_null 0.06%),
  `logits*scale → CE` **38.72 ms** (the scale pass alone ≈ 14 ms),
  torch.compile CE **21.70 ms** (−12.7% vs eager).
- A custom fused CE+scale kernel could at best save the ~14 ms scale pass but
  breaks the bit-stable gate → would need a loss-trajectory gate +
  re-baseline. Not shipped; candidate for a future branch only with that gate.

## #17 flip-budget B{3,4} sensitivity — MEASURED, budgets 3/4 FAIL the gate
- `colab_flip_rate_gate.py --steps 400`: B=3 noisy-window flip rate 0.0090,
  B=4 → 0.0107, both ≫ threshold 0.000244 (0.25/T², T=32); verdict FAIL on
  custom and cublas backends. Flip budget stays **off** on the shipped path.

## perfgate
- check rc=0. regen rc=1: head/update +7.54% vs baseline 700.43 ms,
  fc1/update WARN +4.96% — this branch changes no `kernels/` file, so this is
  VM noise (same class as the r1 slow-VM gate FAIL); the regen baseline is
  deliberately NOT committed.

## List A disposition
#10, #11, #17 measured-negative/closed; #12 negligible; #15 PASS; #6 measured,
deferred by gate policy; #18 closed by protocol (fc1/fwd 25.8–27.1 ms under
ab_median). All 18 List A items now have a measured disposition.
