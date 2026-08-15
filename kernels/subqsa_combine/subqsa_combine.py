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
#include <cuda_fp16.h>

extern "C" {
void launch_subqsa_combine_forward(
    const half* x, const half* o_cmp, const half* o_slc, const half* o_win,
    const half* gate_w1, const half* gate_w2,
    const half* out_norm_weight, const float* o_proj_weight,
    half* y, float gamma,
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

    auto yh = at::empty_like(xh);

    cudaStream_t stream = nullptr;

    // Kernel computes in half precision; inputs arrive as float.
    auto xh = x.to(at::kHalf);
    auto o_cmph = o_cmp.to(at::kHalf);
    auto o_slch = o_slc.to(at::kHalf);
    auto o_winh = o_win.to(at::kHalf);
    auto gw1h = gate_w1.to(at::kHalf);
    auto gw2h = gate_w2.to(at::kHalf);
    auto onwh = out_norm_weight.to(at::kHalf);

    launch_subqsa_combine_forward(
        reinterpret_cast<const half*>(xh.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(o_cmph.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(o_slch.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(o_winh.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(gw1h.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(gw2h.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(onwh.data_ptr<at::Half>()),
        reinterpret_cast<const float*>(o_proj_weight.data_ptr<float>()),
        reinterpret_cast<half*>(yh.data_ptr<at::Half>()),
        static_cast<float>(gamma),
        B, T, H, D,
        stream
    );

    return yh.to(at::kFloat);
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
except Exception as _e:
    _HAS_SUBQSA_COMBINE = False
    # Surface load failure (don't hide — parity tests need to know why)
    print(f"[subqsa_combine] CUDA extension load failed: {_e}", flush=True)

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

    # Sparse ternary O projection when block_mask is provided.
    # The block_sparse CUDA kernel only handles BM=BN=BK=16 (shared mem
    # is fixed 16x16). Production calls it with BN=64 -> OOB/garbage.
    # Until the kernel supports larger tiles, fall back to the same
    # dense gamma-scaled ternary matmul then zero masked N-tiles.
    w_q = torch.clamp(torch.round(o_proj_weight / gamma), -1, 1) * gamma
    y = F.linear(o.float(), w_q).to(dtype=o.dtype)
    if block_mask is not None:
        d_out = o_proj_weight.shape[0]
        BN = 64  # N-tile size used by compute_block_mask in production
        num_n_tiles = (d_out + BN - 1) // BN
        num_k_tiles = (d_out + 15) // 16  # BK=16 for K-tile granularity
        y2d = y.view(-1, d_out)
        for tn in range(num_n_tiles):
            for tk in range(num_k_tiles):
                bit = tn * num_k_tiles + tk
                if not (block_mask[bit // 64] & (1 << (bit % 64))):
                    y2d[:, tn * BN:(tn + 1) * BN] = 0.0
    return y


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
        block_mask = ctx.block_mask

        # Full differentiable recompute: gate MLP, blend, RMSNorm and O-proj
        # all receive real gradients (no zeros for gate_w1/gate_w2/norm weights).
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            o_cmp_in = o_cmp.detach().requires_grad_(True)
            o_slc_in = o_slc.detach().requires_grad_(True)
            o_win_in = o_win.detach().requires_grad_(True)
            gw1 = gate_w1.detach().requires_grad_(True)
            gw2 = gate_w2.detach().requires_grad_(True)
            onw = out_norm_weight.detach().requires_grad_(True)
            opw = o_proj_weight.detach().requires_grad_(True)

            y = _subqsa_combine_eager(x_in, o_cmp_in, o_slc_in, o_win_in,
                                      gw1, gw2, onw, opw, gamma, block_mask)
            grads = torch.autograd.grad(
                y, (x_in, o_cmp_in, o_slc_in, o_win_in, gw1, gw2, onw, opw),
                grad_output, allow_unused=True)

        grads = list(grads)
        # STE for the ternary O projection: round()/clamp() have zero
        # autograd gradient, so compute d_y^T @ blended input manually.
        # Blended input = o after RMSNorm (before O projection).
        with torch.no_grad():
            g1 = F.linear(x.detach(), gate_w1.detach())
            g2 = F.linear(F.silu(g1), gate_w2.detach())
            B, H, T, D_head = o_cmp.shape
            g = g2.view(B, T, 3, H).permute(0, 3, 1, 2).sigmoid()
            g_norm = g / (g.sum(dim=-1, keepdim=True) + 1e-8)
            blended = (g_norm[..., 0:1] * o_cmp.detach()
                       + g_norm[..., 1:2] * o_slc.detach()
                       + g_norm[..., 2:3] * o_win.detach())
            blended = blended.transpose(1, 2).reshape(B, T, -1)
            rms = blended.pow(2).mean(-1, keepdim=True).sqrt()
            blended = blended / (rms + 1e-5) * out_norm_weight.detach()
            d_opw = torch.mm(
                grad_output.reshape(-1, grad_output.shape[-1]).t(),
                blended.reshape(-1, blended.shape[-1]).float())
        if o_proj_weight.numel() > 0:
            grads[-1] = d_opw.to(o_proj_weight.dtype)
        else:
            grads[-1] = torch.zeros_like(o_proj_weight)

        # None -> zero for non-differentiable inputs (block_mask, gamma)
        return (*grads, None, None)


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
