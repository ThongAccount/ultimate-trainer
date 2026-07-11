# CUDA Kernel Fusion Design: BitNet b1.58 × SubQSA

**Date**: 2026-07-11
**Status**: Draft
**Tags**: `project/ultimate-ai-model`, `decision`, `cuda`, `kernel-design`

## 1. Objective

Fuse BitNet b1.58 (ternary weights, Int8 activations) with SubQSA (3-branch sparse attention) via custom CUDA C++ kernels. The PyTorch architecture is already implemented in `ultimate_trainer/`; this spec covers the CUDA kernel layer that replaces critical computation paths for performance.

## 2. Data Type Strategy

| Storage | Type | Rationale |
|---------|------|-----------|
| Master weights (optimizer) | FP32 | AdamW needs FP32 for epsilon/denominator precision |
| QKV/O forward projection | Int8→FP16 | BitLinear outputs Int8; upcast to FP16 for attention safety |
| Attention intermediates | FP16 | Softmax-safe, FlashAttention-compatible |
| Compression MLP weights | FP16 | Small learned projections; no benefit from quantization |
| Gate MLP weights | FP16 | Always kept in FP (never quantized) |
| Attention scores | FP32 inside softmax | Accumulated in FP16, softmax in FP32 for numerical safety |

**Principle**: Forward compute in FP16 with Int8 quantization boundaries. Only master weights and optimizer state live in FP32.

## 3. Kernel Specifications

### Kernel 1: `fused_compressed_attn_kernel`

**Role**: Replace `k.unfold()` + MLP compression (`phi_k`, `phi_v`) — the memory bottleneck.

**Problem**: `torch.unfold` materializes a `(B, H, n_blocks, l*D)` intermediate. For T=4096, l=32, d=16, D=128, n_blocks≈254 → 20 heads × 254 blocks × 4096 = ~80M elements of intermediate storage.

**Approach**: Grid = `(B * H * n_blocks)` — one thread block per head per compressed block. Each block:
1. Computes strided loads from K/V source tensor (no unfold)
2. Evaluates phi_k / phi_v MLP (2-layer: Linear→SiLU→Linear) in registers
3. Writes compressed K_cmp, V_cmp to global memory

**Shared memory**: ~10 KB per block — MLP weight tiles + K/V segment.

**Backward**: Each block's backward pass loads dy, re-computes forward MLP, distributes gradients to source K/V positions via atomicAdd (since blocks overlap with stride < block_len).

### Kernel 2: `fused_selective_attn_kernel`

**Role**: Replace score aggregation → top-K → block gather → causal masked attention.

**Approach**: Grid = `(B, H)`. Two-phase per head:
1. **Phase 1 (shared memory)**: Load aggregate scores `(B, H, n_sel)` into shared memory. Run tournament-style top-K selection (radix-select in registers + shared memory exchange). Writes `topk` indices.
2. **Phase 2 (streaming)**: For each query position `t`, load selected K/V blocks, compute online softmax (Flash Attention algorithm), applying causal mask via position comparison (no explicit mask tensor). Accumulate output in registers.

**Causal masking strategy**: Reconstruct original positions of selected keys as `orig_pos = top_idx * lp + offset`. For each query position `t`, only keys with `orig_pos <= t` contribute. No mask tensor materialized.

**Shared memory**: Score array `(n_sel)` + top-K indices `(K)` + selected block pointers. ~2 KB.

### Kernel 3: `block_sparse_ternary_matmul`

**Role**: Extend existing `ternary_matmul.cu` with block-sparse support.

**Existing kernel**: Dense ternary matmul (tile-based, BM=32, BN=32, BK=32), on-the-fly weight quantization to {-1, 0, +1}, add/sub-only.

**New feature**: Input bitmask indicating which `(N//BN × K//BK)` blocks are active. A `(0)` bit means the entire inner K-loop for that output tile is skipped and the result tile is zeroed.

**Block mask source**: In SubQSA usage, the top-K selection produces a per-query-head block mask. For grouped queries (GQA), the mask is shared across the KV-head group.

**Implementation**: In the existing kernel's inner loop, check `block_mask[tile_n, tile_k]` before entering the BK loop. If 0, skip to next tile.

### Kernel 4: `subqsa_combine_kernel`

**Role**: Fuse gate MLP → sigmoid → normalize → weighted blend → RMSNorm → BitLinear O.

**Approach**: Grid = `(B, T // tile)`. Each thread block processes a tile of tokens:
1. Load `gate_mlp` weights in tiles, compute logits: `Linear_1(x) → SiLU → Linear_2`
2. Per-head sigmoid + L1 normalization of 3 branch weights
3. Weighted blend: `g_cmp * o_cmp + g_slc * o_slc + g_win * o_win`
4. In-register RMSNorm on the blended output
5. BitLinear O projection via tile-based ternary matmul (reuses existing kernel infrastructure)

## 4. Backward Pass Strategy

| Operation | Backward Strategy |
|-----------|------------------|
| BitLinear (ternary STE) | Existing `cuda_ternary.py` `TernaryMatmulFn.backward` — identity through quant, FP32 grad to master weight |
| Compression MLP | Standard autograd chain through MLP (differentiable) |
| Top-K selection | STE — top-K indices are treated as constants in backward; gradient flows through the SDPA softmax to selected K/V |
| Sliding window SDPA | Standard dot-product attention backward (handled by our kernel or FlashAttention) |
| Gate blending | Fully differentiable (sigmoid + linear combination) |
| RMSNorm | Standard RMSNorm backward |

**Key insight**: Top-K selection is non-differentiable, but the gradient path is:
```
loss → gating → o_slc → SDPA softmax → scores(q, k_sel) → k_sel → k_blocks → (selected by top_K)
```
The top-K indices are computed from `raw_scores_cmp` which is **detached** from the selection computation — the gradient flows through `o_slc` to the attended K/V values, not through the selection itself. Standard STE for top-K.

## 5. CUDA Pipeline Architecture

Each kernel follows this structure:

```
kernel_name.cu        → CUDA C++ implementation (device functions + kernel launches)
kernel_name.py        → Python JIT wrapper via torch.utils.cpp_extension.load_inline
                        - autograd.Function (forward/backward)
                        - input validation
                        - CPU fallback
```

Build pipeline reuses the same pattern as `kernels/ternary/ternary_matmul.cu` + `kernels/ternary/ternary_matmul.py`.

## 6. Training Infra Additions

### LR Scheduler
- Cosine decay with linear warmup (same as `1bit_trainer/train.py`)
- `get_cosine_schedule_with_warmup(warmup, max_steps, min_lr_ratio=0.1)`
- Applied in `UltimateTrainer.train()` between optimizer.step() and zero_grad()

### DDP Support
- Detect `LOCAL_RANK`/`WORLD_SIZE` env vars
- Initialize NCCL process group
- Wrap `UltimateModel` in `DistributedDataParallel`
- `DistributedSampler` on DataLoader
- Broadcast initial parameters from rank 0

### Staged Context Extension
- Config: `context_stages = ((4096, 200), (8192, 100), (32768, 50))`
- During training, when `global_step` crosses the threshold, update `model.max_seq_len` and rebuild DataLoader with new sequence length
- LR scheduler continues across stages (no reset)

### Quantized Training Path
- Forward: FP16 intermediates, ternary weights via CUDA kernels, Int8 activation quantization in BitLinear
- Backward: FP32 master weight update (AdamW), STE for weight quantization, standard gradient for everything else
- No full-precision fallback in the hot path — kernels are the primary compute

## 7. File Layout

```
kernels/
├── compressed_attn/
│   ├── compressed_attn_kernel.cu    ← CUDA C++ implementation
│   ├── compressed_attn.py            ← Python wrapper + autograd.Function
│   └── test_compressed_attn.py       ← Unit tests
├── selective_attn/
│   ├── selective_attn_kernel.cu
│   ├── selective_attn.py
│   └── test_selective_attn.py
├── block_sparse_ternary/
│   ├── block_sparse_ternary.cu       ← Extends existing ternary_matmul.cu
│   ├── block_sparse_ternary.py
│   └── test_block_sparse_ternary.py
├── subqsa_combine/
│   ├── subqsa_combine_kernel.cu
│   ├── subqsa_combine.py
│   └── test_subqsa_combine.py
└── (existing files stay as-is)
```

Each kernel directory is independent — no cross-kernel dependencies at the CUDA level.

## 8. Integration Points

| Existing File | Change |
|---|---|
| `ultimate_trainer/subqsa.py` | Add `use_cuda_kernels` flag; when True, dispatch to CUDA kernels instead of PyTorch branches |
| `ultimate_trainer/bitlinear.py` | Keep as-is (already dispatches to ternary_matmul CUDA kernel when available) |
| `ultimate_trainer/config.py` | Add `use_cuda_kernels: bool = False` to `UltimateModelConfig` |
| `ultimate_trainer/model.py` | Pass `use_cuda_kernels` through to `SubQSAAttention` |
| `ultimate_trainer/train.py` | Add LR scheduler, DDP, staged context extension |

## 9. Testing Strategy

Each kernel has:
- **Unit test** (in its directory): correctness vs PyTorch reference at small sizes
- **Integration test** (in `tests/`): plug into SubQSAAttention and verify loss decreases on smoke data
- **Numerical test**: gradient check (`torch.autograd.gradcheck`) on the autograd.Function

All tests runnable via:
```bash
python -m pytest kernels/compressed_attn/test_compressed_attn.py -v
```
