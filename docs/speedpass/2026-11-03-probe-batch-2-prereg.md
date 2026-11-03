# Pre-registration: probe batch 2 (2026-11-03)

Branch `feat/probe-batch-2` from master `0876ba6`. Committed BEFORE any
Colab allocation. One T4 session (`ut-probe2`) runs `scripts/colab_probe_batch2.py`
with `--expect-commit` pinned to this branch's HEAD. Production kernels come
from the prebuilt bundle via `UT_PREBUILT_TAG=kernels-probe-batch-2` (dispatched
from this branch; the branch changes no `kernels/` file, but the bundle tag is
still pinned per the standing rule so no stale `.so` can shadow anything).

## Items, hypotheses, decision rules

| # | Item | File(s) | Hypothesis | Ship bar |
|---|------|---------|-----------|----------|
| 10 | dX K32→K64 unroll-2 (16 MMAs/sync) | `tests/dx_unroll2.cu`, `tests/probe_dx_unroll2.py` | −3..−8% head dX if sync/ldg overhead dominates | delta ≤ −max(2σ_null, 1.0)% on BOTH fc1+head, bit-exact parity vs prod |
| 11 | fwd uint4 W loads (K-window 64) | `tests/fwd_uint4.cu`, `tests/probe_fwd_uint4.py` | −1..−3% fwd (W loads are scalar 32-bit today) | same bar; if ≥ +2% on both shapes, closed negative (adjacent to the closed fwd K-widening dead end) |
| 12 | layer-0 clone cost | `tests/probe_misc.py` §clone | clone at [16352,1024] fp16 is measurable but <1% of the 593.98 ms shipped step | report only (measurement item) |
| 15 | small-dims fwd fallback parity | `tests/probe_misc.py` §small_dims | TC64 kernel correct at small/edge dims (64³, 48³, 16×1024×4096, 80×96×112, 16352×1024×4096) | max rel err < 5e-3 vs fp32 ref on every shape |
| 6 | fused CE + scale probe | `tests/probe_misc.py` §fused_ce | eager CE vs torch.compile vs (logits·scale→CE) at [16352,50272] fp16; quantifies the ~33 ms opportunity | report only; shipping any fused CE breaks the bit-stable gate → needs loss-trajectory gate + re-baseline, NOT decided here |
| 17 | flip-budget B{3,4} sensitivity | `scripts/colab_flip_rate_gate.py --steps 400 --flip-budget 3|4` | flip rate vs budget curve at B=3,4 alongside the existing B=5 sweep | gate PASS/PASS criteria as pre-registered in the flip-rate gate; report only |

## Protocol

- Shapes: B=16352; fc1 K=1024 N=4096; head K=1024 N=50272. Parity = bit-exact
  `torch.equal` vs the production kernel compiled from `kernels/` on the same VM.
- Timing: `bench_protocol.ab_median` interleaved A/B, null arm first
  (σ_null from A/A), ship threshold max(2σ_null, 1.0)%.
- Step order (binding): smoke → pytest (`colab_pytest.TEST_MODULES`) →
  dx_unroll2 → fwd_uint4 → misc → flip_b3 → flip_b4 → perfgate check+regen.
- rc: 0 = gates passed (probe win/loss is a RESULT inside the logs),
  1 = gate failed, 2 = commit-mismatch (INVALID, never committed).
- Artefacts: `results/probe_batch2.<tag>.{json,log,*.log}` with provenance
  block (artefact gate hard-rejects results/*.json without it).

## What this batch does NOT decide

- No production dispatch is touched; nothing here ships to master unless a
  probe clears its bar AND a separate branch + perfgate confirms e2e.
- #6 fused CE is measurement-only this round.
- #18 (fc1 fwd 4.15 vs 7.33 ms conflict) is closed by protocol: current
  ab_median measures fc1/fwd 25.8–27.1 ms; the old numbers were pre-protocol
  artefacts.
