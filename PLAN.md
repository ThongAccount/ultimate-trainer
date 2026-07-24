# Ultimate AI Model — Project Plan

> **Current direction:** Packed ternary discrete-optimization CUDA stack.
> No FP32/BF16 master weights. Weights always ternary {-1,0,+1}.

## Phases

### Phase 1 — PackedTernaryTensor ✅
- `packed_ternary.cuh`: struct, LUT decode, pack16/unpack16, state machine, atomic helpers
- Python wrappers: pack_tensor, unpack_tensor
- Tests: encode/decode, state machine, tensor roundtrip

### Phase 2A — Forward GEMM (correctness) ✅
- v1-v4 kernels, each incrementally optimized
- Correctness verified against F.linear at atol=1e-3
- OOB bounds fix applied

### Phase 2B — Tensor Core WMMA ✅
- TC kernel: wmma::mma_sync, 16×16×16 tiles, 149 GFLOPS peak
- Shared-memory W + X tiles, float accumulation
- Auto-dispatch for batch ≥ 16

### Phase 3 — Backward + Update ✅
- `gemm_backward_dx.cu`: dX = W^T @ dY (column-wise)
- `gemm_update.cu`: fused dW → sign → int16 counter → bit flip
- 2D grid parallelization (each thread = 1 weight)
- No dW tensor materialized

### Phase 4 — Trainable Module ✅
- `PackedTernaryLinearFn`: autograd.Function (forward + backward + update)
- `PackedTernaryLinear`: nn.Module with bias, save/load, reset
- `from_pretrained_linear()`: convert existing nn.Linear
- 7 integration tests passing

### Phase 5 — Performance (IN PROGRESS)
- [x] 2D grid parallelization for update kernel
- [ ] Tiled batch reduction (shared memory) for update kernel
- [ ] TC backward kernel (dX reuse TC WMMA)
- [ ] Auto-dispatch polish (seamless fallback for small dims)

### Phase 6 — Convergence (NEXT)
- [ ] Small model test (4 layers, 256 hidden) with PackedTernaryLinear
- [ ] Verify counter-based optimizer converges on real data
- [ ] Compare loss curves against AdamW baseline

### Phase 7 — Integration (FUTURE)
- [ ] Gigatoken tokenizer backend (optional Rust dep)
- [ ] INT4 quantization for broader Tensor Core coverage
- [ ] torch.compile support for autograd.Function
- [ ] Multi-GPU (DDP) support for PackedTernaryLinear

## Known Issues

1. Update kernel is 5-10× slower than AdamW (9.9 vs 71.7 avg GFLOPS)
2. TC kernel requires batch ≥ 16
3. No convergence verification yet
4. INT4 path unexplored (would enable TC on small batch)
