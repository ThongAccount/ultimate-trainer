# Pre-registration: skip unused gradients (2026-11-24)

Committed on `feat/skip-unused-grads` **before any Colab allocation**.
Thresholds and decision rules below are binding; they are not changed after
measurement.

## Scope

One T4 session runs the skip-unused-grads validation:
1. `smoke` — `tests/test_packed_ternary.py`
2. `pytest` — correctness gate, `colab_pytest.TEST_MODULES`
3. `parity` — bit-identical W trajectory (packed weights, counter, flip_count)
   vs master after N steps
4. `ab` — same-VM whole-step A/B vs shipped baseline 627.6 ms

## Hypothesis

The ternary update kernel mutates W in-place during backward — there is no
`optimizer.step()`. The embedding and LayerNorm parameters are standard
`nn.Module` parameters whose gradients are computed by autograd but never
used. Skipping the gradient computation for these modules saves ~10 ms/step
(~1.5%) of backward compute.

Additionally, the layer-0 dX (gradient w.r.t. the embedding output) is
computed but never used — there is no layer before layer 0.

## Mechanism

1. **Embedding**: freeze `model.embed.weight.requires_grad = False` and
   detach the embedding output before the first layer. The embedding is
   a lookup table — its gradient is sparse and unused.
2. **LayerNorm**: freeze `norm.weight.requires_grad = False` and
   `norm.bias.requires_grad = False` for all norms. LayerNorm gradients
   are dense and unused.
3. **Layer-0 dX**: the backward through fc1 of layer 0 computes dX w.r.t.
   the embedding output. If the embedding is detached, this dX is not
   computed.

## Pre-registered predictions

| Item | Prediction | Falsifier / close rule |
|------|-----------|----------------------|
| skip unused grads | ~10 ms saved on shipped step | win/loss judged per bench_protocol rule |
| parity | bit-identical W trajectory | FAIL if W diverges |
| pytest | 103 passed | any failure = FAIL |

## Decision rule (bench_protocol, mandatory)

All timings use `tests/bench_protocol.py::ab_median`: 10-trial paired
interleaved median, warmup 10, `torch.cuda._sleep` clock settle, A/A null arm
on the first shape for sigma_null. A variant counts as a win/loss only when
|Δ| > max(2*sigma_null, 1%).

## What this does NOT claim

- The embedding and LayerNorm parameters are NOT updated by the ternary
  kernel — they are frozen at their initial values. This is correct because
  there is no optimizer to update them.
- If an optimizer is later added, this optimization must be reverted.

## Artefacts

`results/skip_grads.<tag>.{json,log}` plus per-step logs on the VM.
rc: 0 = all gates pass, 1 = a gate failed, 2 = commit-mismatch refusal.
