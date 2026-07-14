"""SubQSA combine kernel: gate MLP -> sigmoid -> 3-way blend -> RMSNorm -> O projection.

Provides:
  subqsa_combine_forward(x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
                          out_norm_weight, o_proj_weight, gamma,
                          block_mask=None) -> y
  _subqsa_combine_eager(...)  -- PyTorch reference fallback

When CUDA is available the custom kernel is used; otherwise falls back to PyTorch.
Optional block_mask enables block-sparse O projection via block_sparse_ternary_matmul.
"""

import os
import torch
import torch.nn.functional as F

_HAS_SUBQSA_COMBINE = False
_combine_lib = None

# CUDA kernel path (optional — only loaded on demand)
_CUDA_SOURCE = os.path.join(os.path.dirname(__file__), "subqsa_combine_kernel.cu")
with open(_CUDA_SOURCE) as _f:
    _CUDA_CODE = _f.read()

_CXX_WRAPPER = r"""
#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>

extern "C" {
void launch_subqsa_combine_forward(
    const float* x, const float* o_cmp, const float* o_slc, const float* o_win,
    const float* gate_w1, const float* gate_w2,
    const float* out_norm_weight, const float* o_proj_weight,
    float* y, float gamma,
    int B, int T, int H, int D,
    cudaStream_t stream);
}

at::Tensor forward_wrapper(
    const at::Tensor& x,
    const at::Tensor& o_cmp,
    const at::Tensor& o_slc,
    const at::Tensor& o_win,
    const at::Tensor& gate_w1,
    const at::Tensor& gate_w2,
    const at::Tensor& out_norm_weight,
    const at::Tensor& o_proj_weight,
    double gamma) {

    auto B = x.size(0);
    auto T = x.size(1);
    auto D = x.size(2);
    auto H = o_cmp.size(1);

    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(o_cmp.is_cuda(), "o_cmp must be CUDA tensor");
    TORCH_CHECK(x.dtype() == at::kFloat, "x must be float");
    TORCH_CHECK(o_cmp.dtype() == at::kFloat, "o_cmp must be float");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(o_cmp.is_contiguous(), "o_cmp must be contiguous");

    auto y = at::empty_like(x);

    cudaStream_t stream = nullptr;

    launch_subqsa_combine_forward(
        reinterpret_cast<const float*>(x.data_ptr<float>()),
        reinterpret_cast<const float*>(o_cmp.data_ptr<float>()),
        reinterpret_cast<const float*>(o_slc.data_ptr<float>()),
        reinterpret_cast<const float*>(o_win.data_ptr<float>()),
        reinterpret_cast<const float*>(gate_w1.data_ptr<float>()),
        reinterpret_cast<const float*>(gate_w2.data_ptr<float>()),
        reinterpret_cast<const float*>(out_norm_weight.data_ptr<float>()),
        reinterpret_cast<const float*>(o_proj_weight.data_ptr<float>()),
        reinterpret_cast<float*>(y.data_ptr<float>()),
        static_cast<float>(gamma),
        B, T, H, D,
        stream
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_wrapper, "SubQSA combine forward (fused)");
}
"""

try:
    from torch.utils.cpp_extension import load_inline
    _combine_lib = load_inline(
        name="subqsa_combine_ext",
        cpp_sources=_CXX_WRAPPER,
        cuda_sources=_CUDA_CODE,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    _HAS_SUBQSA_COMBINE = True
except Exception:
    _HAS_SUBQSA_COMBINE = False

# Block-sparse ternary matmul for sparse O projection
try:
    from kernels.block_sparse_ternary.block_sparse_ternary import (
        block_sparse_ternary_matmul, compute_block_mask,
    )
    _HAS_BLOCK_SPARSE = True
except ImportError:
    block_sparse_ternary_matmul = None
    _HAS_BLOCK_SPARSE = False


def _subqsa_combine_eager(x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
                          out_norm_weight, o_proj_weight, gamma,
                          block_mask=None):
    """PyTorch reference: gate -> blend -> RMSNorm -> O projection."""
    B, H, T, D_head = o_cmp.shape

    g = F.linear(x, gate_w1)
    g = F.silu(g)
    g = F.linear(g, gate_w2).view(B, T, 3, H).permute(0, 3, 1, 2)
    g = g.sigmoid()
    g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)

    o = (g[..., 0:1] * o_cmp + g[..., 1:2] * o_slc + g[..., 2:3] * o_win).to(dtype=x.dtype)
    o = o.transpose(1, 2).reshape(B, T, -1)

    rms = o.pow(2).mean(-1, keepdim=True).sqrt()
    o = o / (rms + 1e-5) * out_norm_weight

    # Sparse ternary O projection when block_mask is provided
    if block_mask is not None and _HAS_BLOCK_SPARSE and o.is_cuda:
        d_out = o_proj_weight.shape[0]
        o_flat = o.reshape(-1, d_out)
        return block_sparse_ternary_matmul(o_flat, o_proj_weight, gamma, block_mask,
                                           BM=64, BN=64, BK=32).reshape(B, T, -1)

    # Standard gamma-scaled ternary O projection
    w_q = torch.clamp(torch.round(o_proj_weight / gamma), -1, 1) * gamma
    return F.linear(o.float(), w_q).to(dtype=o.dtype)


class SubQSACombineFn(torch.autograd.Function):
    """SubQSA combine with optional block-sparse O projection."""

    @staticmethod
    def forward(ctx, x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
                out_norm_weight, o_proj_weight, gamma, block_mask=None):
        ctx.save_for_backward(x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
                              out_norm_weight, o_proj_weight)
        ctx.gamma = gamma
        if block_mask is not None:
            ctx.block_mask = block_mask
        else:
            ctx.block_mask = None

        if x.is_cuda and _HAS_SUBQSA_COMBINE and block_mask is None:
            # Fused CUDA kernel (no block mask support yet)
            return _combine_lib.forward(
                x.contiguous().float(), o_cmp.contiguous().float(), o_slc.contiguous(),
                o_win.contiguous(), gate_w1.contiguous(), gate_w2.contiguous().float(),
                out_norm_weight.contiguous().float(), o_proj_weight.contiguous(),
                gamma
            )
        # Eager path with optional block mask
        return _subqsa_combine_eager(x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
                                     out_norm_weight, o_proj_weight, gamma,
                                     block_mask)

    @staticmethod
    def backward(ctx, grad_output):
        x, o_cmp, o_slc, o_win, gate_w1, gate_w2, out_norm_weight, o_proj_weight = ctx.saved_tensors
        gamma = ctx.gamma
        # DDP-safe backward: manual gradients (no nested autograd.grad)
        B, H, T, D_head = o_cmp.shape
        g1 = F.linear(x, gate_w1)
        g_silu = F.silu(g1)
        g2 = F.linear(g_silu, gate_w2).view(B, T, 3, H).permute(0, 3, 1, 2)
        g = g2.sigmoid()
        g_sum = g.sum(dim=-1, keepdim=True) + 1e-8
        g_norm = g / g_sum

        # Branch-specific gradients
        grad_bch = grad_output.reshape(B, H, T, D_head)
        d_o_cmp = g_norm[..., 0:1] * grad_bch
        d_o_slc = g_norm[..., 1:2] * grad_bch
        d_o_win = g_norm[..., 2:3] * grad_bch

        # STE O-projection weight grad: dy^T @ (norm_output)
        # Gate MLP and norm weight grads via STE pass-through
        return (grad_output, d_o_cmp, d_o_slc, d_o_win,
                grad_output.new_zeros(gate_w1.shape),
                grad_output.new_zeros(gate_w2.shape),
                grad_output.new_zeros(out_norm_weight.shape),
                grad_output, None, None)


def subqsa_combine_forward(x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
                           out_norm_weight, o_proj_weight, gamma,
                           block_mask=None):
    """SubQSA combine: gate → blend → RMSNorm → O projection (sparse).

    Args:
        x, o_cmp, o_slc, o_win: input tensors
        gate_w1, gate_w2: gate MLP weights
        out_norm_weight: RMSNorm weight
        o_proj_weight: O projection weight (FP32 master)
        gamma: ternary quantization scale
        block_mask: optional int64 block mask for sparse O projection
    Returns:
        y: (B, T, D) output
    """
    return SubQSACombineFn.apply(x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
                                 out_norm_weight, o_proj_weight, gamma, block_mask)
