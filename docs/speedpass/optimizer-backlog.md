# Optimizer backlog — shipped cuBLAS arm

Last updated: 2026-11-24

## Closed items

### aten::mm precision swap — CLOSED (hardware limitation)

T4 (sm_75, Turing) has no TF32, FP8, or FP4 tensor core modes. The shipped
GEMMs already run at FP16 on tensor cores (~75% of T4's 65 TFLOPS peak).
The update dW uses FP16 inputs with FP32 output (cublasGemmEx) — FP32 output
is required for sign safety in the flip gate. No precision swap is possible
on this hardware.

### Scale-alpha fold — CLOSED (parity FAIL)

Branch: `feat/scale-alpha-fold` @ `9e68af0`
Result: parity FAIL at 16352x1024x4096. Pre-scaling W16 before accumulation
vs post-scaling the output produces different fp16 rounding for non-power-of-2
scale values. Not bit-exact; candidate closed.

### Custom-arm kernel backlog — CLOSED permanently

All custom-arm kernel optimization candidates measured negative (A2 +1.03%,
B2 +18.2%, B4 +5.6%, A3 +23.4%). Masters debate closed all permanently.
The custom arm remains only as a bit-exact reference.

### Flip-count batching/deferral — CLOSED (measured-negative)

Branch: `feat/flip-count-batching` @ `d884ad5`
Result: **+2.46% slower** (494.423 ms vs 482.576 ms master, flip_budget=5).
Branch divergence from the deferred flip_count check costs more than the
memory traffic savings. Candidate closed; not merged.

### Fused LayerNorm casts — CLOSED (NOISE)

Branch: `feat/fused-ln-casts` @ `9bb8671`
Result: **−0.41%** (465.697 ms vs 467.636 ms master) — below the 1% threshold.
torch.compile overhead offsets the cast fusion savings. Not merged.

### Skip unused gradients — MERGED (WIN)

Branch: `feat/skip-unused-grads` @ `8a7d40c`, merged to master @ `ae6b64d`
Result: **−3.07%** (453.284 ms vs 467.636 ms master) — **14.352 ms saved**.
Bit-identical loss trajectory. 103/103 pytest passed.

## Backlog exhausted

All optimizer items from the 2026-11-22 list have been processed. The shipped
cuBLAS arm step time improved from 627.6 ms to **453.3 ms** (−27.8%) through:
- P0 cuBLAS migration (−81.33% vs custom)
- I1 fused CE+scale (−11.714 ms)
- Skip unused gradients (−14.352 ms)
