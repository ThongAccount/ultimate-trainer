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

## Active items

### 1. Fused LayerNorm casts — NOT STARTED

Target: ~10 ms/step (~1.5%)
Mechanism: fuse the float()→half() casts in the LayerNorm path

### 2. Skip unused gradients — NOT STARTED

Target: ~10 ms/step (~1.5%)
Mechanism: skip layer-0 dX and embedding/LN gradient paths that are never used
Requires: bit-identical W trajectory
