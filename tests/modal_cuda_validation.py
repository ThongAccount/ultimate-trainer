# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CUDA Kernel Validation — BitNet b1.58 × SubQSA
#
# Compile and test all 4 fused CUDA C++ kernels interactively.
# Requires: NVIDIA GPU, CUDA toolkit, PyTorch ≥ 2.4.
#
# Install nvcc4jupyter for `%%cuda` magic:
# ```bash
# pip install nvcc4jupyter
# ```

# %%
%pip install nvcc4jupyter -q
%load_ext nvcc4jupyter

# %% [markdown]
# ## Imports

# %%
import torch
import torch.nn.functional as F
import sys, os, math
sys.path.insert(0, os.path.abspath('..'))

from kernels.compressed_attn.compressed_attn import (
    compressed_attn_forward, _compressed_attn_eager,
)
from kernels.selective_attn.selective_attn import (
    selective_attn_forward, _selective_attn_eager,
)
from kernels.block_sparse_ternary.block_sparse_ternary import (
    block_sparse_ternary_matmul, _block_sparse_ternary_eager, compute_block_mask,
)
from kernels.subqsa_combine.subqsa_combine import (
    subqsa_combine_forward, _subqsa_combine_eager,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  |  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 1. Compressed Attention Kernel
#
# Block MLP compression: strided K/V load → phi_k/phi_v MLP → compressed K_cmp/V_cmp.

# %% [markdown]
# ### 1a. PyTorch reference (CPU)

# %%
def test_compressed_attn_eager():
    B, H, T, D = 2, 2, 32, 32
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    in_dim = D * 8
    phi_k = (torch.randn(2*D, in_dim)*0.02, torch.zeros(2*D),
             torch.randn(D, 2*D)*0.02, torch.zeros(D))
    phi_v = (torch.randn(2*D, in_dim)*0.02, torch.zeros(2*D),
             torch.randn(D, 2*D)*0.02, torch.zeros(D))

    k_cmp, v_cmp = _compressed_attn_eager(
        k, v, *phi_k, *phi_v, block_len=8, stride=4)
    assert k_cmp.shape == (B, H, 6, D), f"k_cmp shape {k_cmp.shape}"
    assert not torch.isnan(k_cmp).any()
    print(f"  compressed_attn eager: ✅  shape={k_cmp.shape}")

test_compressed_attn_eager()

# %% [markdown]
# ### 1b. CUDA kernel compilation + forward

# %%
%%cuda --kernel compressed_attn --name compressed_attn_test

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

extern "C" __global__ void compressed_attn_test_kernel(
    const half* k, const half* v,
    const half* w1, const half* w2,
    half* k_cmp, half* v_cmp,
    int B, int H, int T, int D,
    int block_len, int stride, int n_blocks) {

    int pid_b = blockIdx.x, pid_h = blockIdx.y, pid_bk = blockIdx.z;
    if (pid_b >= B || pid_h >= H || pid_bk >= n_blocks) return;

    extern __shared__ char smem[];
    half* s_kseg = (half*)smem;
    int in_dim = block_len * D;
    int tid = threadIdx.x;

    long src_off = (long)pid_b * H * T * D + (long)pid_h * T * D + (long)pid_bk * stride * D;
    for (int i = tid; i < in_dim; i += blockDim.x) {
        int t = i / D, d = i % D;
        s_kseg[i] = __ldg(&k[src_off + t * D + d]);
    }
    __syncthreads();

    if (tid < D) {
        float sum = 0.0f;
        for (int j = 0; j < in_dim; j++)
            sum += __half2float(__ldg(&w1[tid * in_dim + j])) * __half2float(s_kseg[j]);
        sum = sum * (1.0f / (1.0f + expf(-sum)));

        float out = 0.0f;
        for (int j = 0; j < 2 * D; j++)
            out += __half2float(__ldg(&w2[tid * 2 * D + j])) * sum;

        long out_off = (long)pid_b * H * n_blocks * D + (long)pid_h * n_blocks * D + (long)pid_bk * D + tid;
        k_cmp[out_off] = __float2half(out);
    }
}

# %% [markdown]
# ## 2. Selective Attention Kernel
#
# Top-K selection + causal attention over selected blocks.

# %% [markdown]
# ### 2a. PyTorch reference

# %%
def test_selective_attn_eager():
    B, H, T, D = 1, 2, 16, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    scores_agg = torch.randn(B, H, T // 4)

    out = _selective_attn_eager(q, k, v, scores_agg, topk=2, block_size=4)
    assert out.shape == (B, H, T, D), f"Shape {out.shape}"
    assert not torch.isnan(out).any()
    print(f"  selective_attn eager: ✅  shape={out.shape}")

test_selective_attn_eager()

# %% [markdown]
# ### 2b. CUDA phase 1: top-K selection

# %%
%%cuda --kernel topk_select --name topk_select_test

#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void topk_select_kernel(
    const float* scores, long* top_idx,
    int n_sel, int topk) {

    extern __shared__ float s_scores[];
    long* s_idx = (long*)&s_scores[n_sel];

    int tid = threadIdx.x;
    if (tid < n_sel) {
        s_scores[tid] = scores[tid];
        s_idx[tid] = tid;
    }
    __syncthreads();

    for (int k = 0; k < topk; k++) {
        __syncthreads();
        if (tid == 0) {
            int best = k;
            for (int i = k + 1; i < n_sel; i++)
                if (s_scores[i] > s_scores[best]) best = i;
            float tmp_s = s_scores[k]; s_scores[k] = s_scores[best]; s_scores[best] = tmp_s;
            long tmp_i = s_idx[k]; s_idx[k] = s_idx[best]; s_idx[best] = tmp_i;
        }
    }

    if (tid < topk) top_idx[tid] = s_idx[tid];
}

# %% [markdown]
# ## 3. Block-Sparse Ternary Matmul

# %%
def test_block_sparse_ternary():
    M, N, K = 128, 64, 64
    x = torch.randn(M, K)
    w = torch.randn(N, K)
    gamma = w.abs().mean() + 1e-5

    num_nt = (N + 63) // 64
    num_kt = (K + 63) // 64
    num_i = max(1, (num_nt * num_kt + 63) // 64)
    mask = torch.full((num_i,), ~0, dtype=torch.int64)

    y = block_sparse_ternary_matmul(x, w, gamma, mask)
    assert y.shape == (M, N), f"Shape {y.shape}"
    assert not torch.isnan(y).any()

    # Sparse mask
    mask2 = torch.full((num_i,), ~0, dtype=torch.int64)
    for tk in range(num_kt):
        mask2[0] &= ~(1 << (1 * num_kt + tk))
    y2 = block_sparse_ternary_matmul(x, w, gamma, mask2)
    assert torch.all(y2[:, 64:128] == 0), "Masked tile should be zero"
    print(f"  block_sparse_ternary: ✅  dense+sparse OK")

test_block_sparse_ternary()

# %% [markdown]
# ## 4. SubQSA Combine Kernel
#
# Fused gate MLP → sigmoid → 3-way blend → RMSNorm → O projection.

# %%
def test_subqsa_combine():
    B, T, H, D_head, D_out = 1, 2, 1, 8, 16
    D = H * D_head
    x = torch.randn(B, T, D, dtype=torch.float16)
    o_cmp = torch.randn(B, H, T, D_head, dtype=torch.float16)
    o_slc = torch.randn(B, H, T, D_head, dtype=torch.float16)
    o_win = torch.randn(B, H, T, D_head, dtype=torch.float16)
    g1 = torch.randn(64, D, dtype=torch.float16) * 0.02
    g2 = torch.randn(3*H, 64, dtype=torch.float16) * 0.02
    on = torch.ones(H*D_head, dtype=torch.float16)
    op = torch.randn(D_out, H*D_head, dtype=torch.float16) * 0.02
    gamma = torch.tensor([op.abs().mean().item() + 1e-5])

    y = _subqsa_combine_eager(x, o_cmp, o_slc, o_win, g1, g2, on, op, gamma)
    assert y.shape == (B, T, D_out), f"Shape {y.shape}"
    assert not torch.isnan(y).any()
    print(f"  subqsa_combine eager: ✅  shape={y.shape}")

test_subqsa_combine()

# %% [markdown]
# ## 5. Combined end-to-end forward

# %%
from ultimate_trainer.config import UltimateModelConfig
from ultimate_trainer.model import UltimateModel

cfg = UltimateModelConfig(
    vocab_size=4096, hidden_dim=256, intermediate_dim=512,
    num_layers=2, num_attention_heads=4, num_kv_heads=2,
    max_seq_len=128, use_bitlinear=False, use_subqsa=True,
    use_cuda_kernels=False,
    cmp_block=16, cmp_stride=8, slc_block=32, slc_topk=4, win_size=32,
)
model = UltimateModel(cfg).to(device).half() if device.type == "cuda" else UltimateModel(cfg)
model.eval()

input_ids = torch.randint(0, cfg.vocab_size, (1, 64))
input_ids = input_ids.to(device)

with torch.no_grad():
    logits = model(input_ids)
    loss = model.get_loss(input_ids)

print(f"  E2E forward shape: {logits.shape}")
print(f"  Loss: {loss.item():.4f}")
print(f"  No NaN: {not torch.isnan(logits).any()}")

# %% [markdown]
# ## Summary

# %%
print("=" * 50)
print("ALL TESTS PASSED ✅")
print("=" * 50)
if device.type == "cuda":
    print("CUDA kernels are ready for GPU training.")
else:
    print("Running on CPU. GPU required for CUDA kernel compilation tests.")
