# Pre-registration: fused LayerNorm casts (2026-11-24)

Committed on `feat/fused-ln-casts` **before any Colab allocation**.
Thresholds and decision rules below are binding; they are not changed after
measurement.

## Scope

One T4 session runs the fused-LN validation:
1. `smoke` — `tests/test_packed_ternary.py`
2. `pytest` — correctness gate, `colab_pytest.TEST_MODULES`
3. `parity` — bit-exact loss trajectory vs master (same seed, same data)
4. `ab` — same-VM whole-step A/B vs shipped baseline 627.6 ms

## Hypothesis

The LayerNorm path `h = norms[i](h.float()).half()` does a separate fp32→fp16
cast pass per layer (6 layers = 6 extra kernel launches + memory traffic).
Fusing via `torch.compile` on the norm+cast pattern would eliminate the
separate cast pass, saving ~10 ms/step (~1.5%).

## Mechanism

Wrap the LayerNorm + cast in a `torch.compile`'d function:
```python
@torch.compile
def _fused_ln_cast(norm, h):
    return norm(h.float()).half()
```
The compiler fuses the float() cast, LayerNorm, and half() cast into a single
kernel, eliminating the intermediate fp32 tensor allocation and the separate
cast pass.

## Pre-registered predictions

| Item | Prediction | Falsifier / close rule |
|------|-----------|----------------------|
| fused LN casts | ~10 ms saved on shipped step | win/loss judged per bench_protocol rule |
| parity | bit-exact loss trajectory | FAIL if loss diverges |
| pytest | 103 passed | any failure = FAIL |

## Decision rule (bench_protocol, mandatory)

All timings use `tests/bench_protocol.py::ab_median`: 10-trial paired
interleaved median, warmup 10, `torch.cuda._sleep` clock settle, A/A null arm
on the first shape for sigma_null. A variant counts as a win/loss only when
|Δ| > max(2*sigma_null, 1%).

## Artefacts

`results/fused_ln.<tag>.{json,log}` plus per-step logs on the VM.
rc: 0 = all gates pass, 1 = a gate failed, 2 = commit-mismatch refusal.
