# Task: TritonJIT counterpart of the CUDA ternary stack

You are working in a CUDA C++ training stack for LLMs with 2-bit packed
ternary weights {-1,0,+1} and a counter-based optimizer (no AdamW). The CUDA
source is INCOMPLETE — some kernels are stubs, some have drifted from the
Python wrappers. Your job: build a complete, verified **Triton JIT**
replacement (`triton.jit` kernels only — no custom CUDA C++, no
torch.utils.cpp_extension) from the CUDA code as spec.

## Repo
- CUDA C++ spec: `kernels/packed_ternary/*.cu`, `kernels/packed_ternary/packed_ternary.cuh`
- Python loader/wrappers (authoritative for signatures): `kernels/packed_ternary/pack_forward.py`, `pack_update.py`, `custom_ops.py`, `packed_linear.py`
- Ground truth helpers: `pack_tensor`/`unpack_tensor`/`ref_linear` in `kernels/packed_ternary/__init__.py` + `pack_forward.py`
- **Pure-torch reference impl (already in this branch): `kernels/packed_ternary/torch_impl.py`** — your correctness oracle for the update semantics and pack layout. Do not rewrite it; import it in tests.
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
4. Fused dX+update entry point (see gemm_fused_backward_update.cu).

## Deliverables — TRITON ONLY
### A) `kernels/packed_ternary/triton_impl.py`
- `triton.jit` kernels for fwd/bwd/update with the same public function
  signatures as `torch_impl.py` (`ternary_forward`, `ternary_backward_dx`,
  `ternary_update`, fused).
- Forward/backward: `tl.dot` on tiles; decode the packed W tile in-kernel with
  shifts/where (`(word >> 2*(k%16)) & 3` → {0,1,-1} → fp16).
- Update: **tile by (N, word)** — grid dim 2 = KWORDS, each program owns one
  whole 16-code word, so the flip is a plain `tl.store`, NO atomics.
- Match storage contract exactly; mind alignment (words are 4B, counter 2B).
- CPU is NOT required — CUDA only.

### B) `tests/test_triton_impl.py`
- Forward/dX vs ground truth (`ref_linear`, autograd-through-dequantized W);
  update vs `torch_impl.ternary_update` as the reference (bit-exact on W_packed
  and counter after one step).
- Cover: (64,64,64), (32,128,128), (128,96,96), ragged (17,33,65); FP16
  tolerance < 1e-2 for GEMMs; update must be bit-exact.

## Known traps (from prior porting)
- Bit-twiddle in int64/uint32 space; int32 shifts sign-extend in torch —
  in Triton use `tl.where` and mask on the packed word; beware `(w >> sh)` on
  signed int32.
- Word index is `k//16`, bit pos `2*(k%16)` — a single mis-mapped column
  invalidates everything.
- `tl.dot` needs all GEMM dims ≥ 16 and power-of-2 tiles; mask loads with
  `other=0` at ragged edges; pad K tiles to multiples of 16 so a word never
  spans two programs.
- `tl.trans` for the `[N,K]` accumulation orientation; accumulate in fp32.
- Counter reset must write `0` exactly where flipped, keep elsewhere.

## Hard requirements
- Read the .cu/.cuh + torch_impl.py first. Where CUDA and wrappers disagree,
  the Python wrapper + this prompt + torch_impl.py win (they are the contract).
- Never call the CUDA extensions from your impls. They are the spec, not deps.
- Do not modify the CUDA files, the training scripts, or torch_impl.py.
- Report: which kernels were stubs vs drifted, and any semantic gap you had to
  infer (e.g. exact threshold comparison, counter reset timing).

## Acceptance
`python tests/test_triton_impl.py` passes on the GPU host (Triton is GPU-only);
a 200-step Shakespeare run with a module that uses your triton impl converges
to loss < 3.0 (counter optimizer active, no AdamW).
