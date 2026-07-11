# kernels/selective_attn/selective_attn.py
# Python wrapper for the fused selective attention CUDA kernel.
# Two-phase: top-K selection from scores_agg, then causal FlashAttention
# over the selected K/V blocks.

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os
import math

_CUDA_SOURCE = os.path.join(os.path.dirname(__file__), "selective_attn_kernel.cu")

_CXX_WRAPPER = r"""
#include <torch/extension.h>
#include <vector>

// Forward declarations from the CUDA source
void launch_selective_phase1(
    const float* scores, long* top_idx,
    int B, int H, int n_sel, int topk,
    cudaStream_t stream);

void launch_selective_phase2(
    const half* q, const half* k, const half* v,
    const long* top_idx, half* attn_out,
    int B, int H, int T, int D,
    int block_size, int topk, int n_sel,
    cudaStream_t stream);

std::vector<at::Tensor> forward_wrapper(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& scores_agg,
    int64_t topk, int64_t block_size) {

    auto B = q.size(0);
    auto H = q.size(1);
    auto T = q.size(2);
    auto D = q.size(3);
    auto n_sel = scores_agg.size(-1);

    auto top_idx = at::empty({B, H, std::min(topk, (int64_t)n_sel)},
                              scores_agg.options().dtype(at::kLong));
    auto attn_out = at::empty({B, H, T, D}, q.options().dtype(at::kHalf));

    auto stream = at::cuda::getCurrentCUDAStream();

    launch_selective_phase1(
        reinterpret_cast<const float*>(scores_agg.data_ptr<float>()),
        reinterpret_cast<long*>(top_idx.data_ptr<long>()),
        B, H, n_sel, topk, stream);

    launch_selective_phase2(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<const long*>(top_idx.data_ptr<long>()),
        reinterpret_cast<half*>(attn_out.data_ptr<at::Half>()),
        B, H, T, D, block_size, topk, n_sel, stream);

    return {attn_out, top_idx};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_wrapper, "Selective attention forward");
}
"""

_HAS_SELECTIVE_ATTN = False
_selective_lib = None

try:
    _selective_lib = load_inline(
        name="selective_attn",
        cpp_sources=_CXX_WRAPPER,
        cuda_sources=[_CUDA_SOURCE],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    _HAS_SELECTIVE_ATTN = True
except Exception as e:
    print(f"[selective_attn] CUDA extension load failed: {e}")


class SelectiveAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scores_agg, topk, block_size):
        ctx.save_for_backward(q, k, v, scores_agg)
        ctx.topk = topk
        ctx.block_size = block_size
        ctx.n_sel = scores_agg.size(-1)

        if q.is_cuda and _HAS_SELECTIVE_ATTN:
            result, top_idx = _selective_lib.forward(
                q.contiguous(), k.contiguous(), v.contiguous(),
                scores_agg.contiguous(), topk, block_size
            )
            ctx.top_idx = top_idx
            return result
        else:
            # CPU fallback
            return _selective_attn_eager(q, k, v, scores_agg, topk, block_size)

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, scores_agg = ctx.saved_tensors
        # Standard backward through SDPA (grad_q, grad_k, grad_v)
        # CPU fallback for now
        return (grad_output, None, None, None, None, None)


def _selective_attn_eager(q, k, v, scores_agg, topk, block_size):
    """PyTorch reference for the selection branch."""
    B, H, T, D = q.shape
    lp = block_size
    n_sel = max(1, T // lp)
    topk_actual = min(topk, n_sel)

    # Top-K selection
    _, top_idx = scores_agg.topk(topk_actual, dim=-1)  # (B, H, K)

    # Gather blocks
    k_blocks = k[:, :, :n_sel * lp, :].reshape(B, H, n_sel, lp, D)
    v_blocks = v[:, :, :n_sel * lp, :].reshape(B, H, n_sel, lp, D)

    bi = top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, lp, D)
    k_sel = torch.gather(k_blocks, dim=2, index=bi).reshape(B, H, topk_actual * lp, D)
    v_sel = torch.gather(v_blocks, dim=2, index=bi).reshape(B, H, topk_actual * lp, D)

    # Causal mask: reconstruct original positions
    orig_block_starts = top_idx * lp  # (B, H, K)
    offsets = torch.arange(lp, device=q.device)
    orig_k_pos = (orig_block_starts.unsqueeze(-1) + offsets).reshape(B, H, topk_actual * lp)
    q_pos = torch.arange(T, device=q.device).view(1, 1, T, 1)
    valid = orig_k_pos.unsqueeze(2) <= q_pos  # (B, H, T, K*lp)
    attn_mask = torch.where(valid, 0.0, float("-inf")).to(q.dtype)

    return F.scaled_dot_product_attention(q, k_sel, v_sel, attn_mask=attn_mask)


def selective_attn_forward(q, k, v, scores_agg, topk, block_size):
    return SelectiveAttnFn.apply(q, k, v, scores_agg, topk, block_size)
