# Pre-registration: flip-count batching/deferral (2026-11-24)

Committed on `feat/flip-count-batching` **before any Colab allocation**.
Thresholds and decision rules below are binding; they are not changed after
measurement.

## Scope

One T4 session runs the flip-count batching validation:
1. `smoke` — `tests/test_packed_ternary.py` (parity-only plumbing check)
2. `pytest` — correctness gate, `colab_pytest.TEST_MODULES`
3. `flip_count_parity` — bit-exact parity of the batched flip-count variant
   vs the current per-step flip-count kernel. Parity must cover:
   - W trajectory (bit-identical packed weights after N steps)
   - V/M/C gate values (counter, flip_count, and flip budget semantics)
   - Loss trajectory (bit-identical loss values)
4. `flip_count_ab` — same-VM whole-step A/B, 10-trial paired median vs
   measured shipped baseline 627.6 ms with flip_budget=5

No production file is modified outside the flip-count change.

## Hypothesis

Flip counting costs ~11 ms/step (~1.9%) when flip_budget > 0. The cost comes
from the extra memory reads/writes for `flip_count` in the update epilogue
kernel. Batching or deferring the flip-count updates could reduce this cost.

## Mechanism

Current: each thread in the update epilogue reads `flip_count[idx]` to check
if the weight is frozen, and increments `flip_count[idx]` when a flip occurs.

Variant options (to be determined during implementation):
- **A. Deferred frozen check**: skip the `flip_count` read when the counter
  is far from the threshold (most weights won't flip)
- **B. Batched flip-count writes**: accumulate flips in shared memory and
  write to global memory less frequently
- **C. Reduced-precision flip_count**: use int8 instead of int16 for
  flip_count (halves the memory traffic)

## Pre-registered predictions

| Item | Prediction | Falsifier / close rule |
|------|-----------|----------------------|
| flip-count batching | ~11 ms saved on shipped step with flip_budget=5 | win/loss judged per bench_protocol rule below |
| parity | bit-exact W trajectory, V/M/C values, loss | FAIL if not bit-exact |
| pytest | 103 passed | any failure = FAIL |

## Decision rule (bench_protocol, mandatory)

All timings use `tests/bench_protocol.py::ab_median`: 10-trial paired
interleaved median, warmup 10, `torch.cuda._sleep` clock settle, A/A null arm
on the first shape for sigma_null. A variant counts as a win/loss only when
|Δ| > max(2*sigma_null, 1%).

## What this batch does NOT claim

- Flip-count batching is NOT a change to the flip budget semantics.
- The flip budget safety mechanism must be preserved exactly.
- Single-seed timing numbers support no convergence claim.

## Artefacts

`results/flip_count.<tag>.{json,log}` plus per-step logs on the VM,
downloaded with `colab download` before `colab stop`. rc: 0 = all gates
pass, 1 = a gate failed (valid result), 2 = commit-mismatch refusal
(invalid; never commit rc=2 artefacts).
