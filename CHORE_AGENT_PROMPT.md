# Task: Pure-PyTorch counterpart of the CUDA ternary stack

You are working in a CUDA C++ training stack for LLMs with 2-bit packed
ternary weights {-1,0,+1} and a counter-based optimizer (no AdamW). The CUDA
source is INCOMPLETE — some kernels are stubs, some have drifted from the
Python wrappers. Your job: build a complete, verified **pure-PyTorch**
replacement (no CUDA C++, no Triton) from the CUDA code as spec.

## Repo
- CUDA C++ spec: `kernels/packed_ternary/*.cu`, `kernels/packed_ternary/packed_ternary.cuh`
- Python loader/wrappers (authoritative for signatures): `kernels/packed_ternary/pack_forward.py`, `pack_update.py`, `custom_ops.py`, `packed_linear.py`
- Ground truth helpers: `pack_tensor`/`unpack_tensor`/`ref_linear` in `kernels/packed_ternary/__init__.py` + `pack_forward.py`
- Tests to mirror: `tests/test_*.py`

## Storage contract (must match EXACTLY)
- `W_packed`: int32 `[N, stride_words]`, `stride_words = ceil(K/16)`.
  One word = 16 ternary codes, 2 bits each, little-endian:
  `word = Σ code_i << (2*i)`,  code 0→0, 1→+1, 2→−1.
- `counter`: int16 `[N, K]`.
- `X`, `dY`, `Y`: FP16. All accumulation in FP32. Outputs `.half()`.

## Ops to replicate (from gemm_update.cu + gemm_forward_tc.cu etc.)
1. Forward:  `Y[b,n] = Σ_k X[b,k] * w[n,k]` where w is decoded ternary.
2. Backward dX: `dX[b,k] = Σ_n dY[b,n] * w[n,k]`.
3. Update (per weight r,c):  NOTE: CUDA is DESCENT
   - `dW[r,c] = Σ_b dY[b,r] * X[b,c]`  (never materialized in CUDA; compute in FP32)
   - `counter[r,c] -= sign(dW)`  (gradient DESCENT — matches CUDA kernels)
   - if `counter[r,c] >  threshold`: ternary value += 1 (−1→0→+1), reset counter
   - if `counter[r,c] < −threshold`: ternary value −= 1 (+1→0→−1), reset counter
   - flip writes the 2-bit code back into the packed word in place.
4. Fused dX+update entry point (see gemm_fused_backward_update.cu) — a single
   call doing both is fine even if internally two torch ops.

## Deliverables — TORCH ONLY
### A) `kernels/packed_ternary/torch_impl.py`
- `unpack_ternary(packed, rows, cols) -> int8 [N,K]` — vectorized, no Python loops over rows/cols.
- `ternary_forward`, `ternary_backward_dx`, `ternary_update` (+ fused).
- An `autograd.Function` equivalent of `PackedTernaryLinearFn` that runs the
  update inside `backward()` (preserve the clone/requires_grad trick — see
  packed_linear.py) so the module trains without an optimizer.
- CPU + CUDA both work (pure torch ops only).
- Do NOT create triton_impl.py; do NOT call the CUDA extensions.

### B) `tests/test_torch_impl.py`
- Assert against ground truth: `unpack_tensor` for decode; `ref_linear` for
  forward; autograd-through-dequantized-weights for dX; a hand-rolled update
  reference (mask math on decoded weights + re-pack).
- Cover: aligned (64,64,64), (32,128,128), (128,96,96), ragged (17,33,65);
  FP16 tolerance < 1e-2. Also verify update resets counters where flipped and
  leaves others unchanged.

## Known traps (from prior porting)
- Bit-twiddle in int64 masked with `& 0xFFFFFFFF` (int32 shifts sign-extend).
- LUT gathers need long-index dtype: `_LUT[code]` with `code` in torch.long.
- Word index is `k//16`, bit pos `2*(k%16)` — a single mis-mapped column
  invalidates everything (it silently produced 3× outputs in the CUDA port).
- Re-packing the whole W each step is fine (correctness target, not perf).
- `sign(dW)` is ±1 → int16 counter cannot overflow for sane thresholds.
- Keep the `.clone().requires_grad_(True)` + `ctx.X_saved` pattern from
  packed_linear.py — removing it triples autograd overhead / corrupts state.

## Hard requirements
- Read the .cu/.cuh first. Where CUDA and wrappers disagree, the Python
  wrapper + this prompt win (they are the contract).
- Never call the CUDA extensions from your impls. They are the spec, not deps.
- Do not modify the CUDA files or the training scripts.
- Report: which kernels were stubs vs drifted, and any semantic gap you had to
  infer (e.g. exact threshold comparison, counter reset timing).

## Acceptance
`python tests/test_torch_impl.py` passes on CPU and GPU; a 200-step
Shakespeare run with a module that uses your torch impl converges to
loss < 3.0 (counter optimizer active, no AdamW).
