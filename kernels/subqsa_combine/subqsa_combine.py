"""SubQSA combine kernel: gate MLP → sigmoid → 3-way blend → RMSNorm → O projection."""

import torch
import torch.nn.functional as F
import math

_HAS_SUBQSA_COMBINE = False
_combine_lib = None


def _subqsa_combine_eager(x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
                          out_norm_weight, o_proj_weight, gamma):
    """PyTorch reference: gate → blend → RMSNorm → O projection."""
    B, H, T, D = o_cmp.shape

    g = F.linear(x, gate_w1)
    g = F.silu(g)
    g = F.linear(g, gate_w2).view(B, T, 3, H).permute(0, 3, 1, 2)
    g = g.sigmoid()
    g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)

    o = (g[..., 0:1] * o_cmp + g[..., 1:2] * o_slc + g[..., 2:3] * o_win).to(dtype=x.dtype)
    o = o.transpose(1, 2).reshape(B, T, -1)

    rms = o.pow(2).mean(-1, keepdim=True).sqrt()
    o = o / (rms + 1e-5) * out_norm_weight

    w_q = torch.clamp(torch.round(o_proj_weight / gamma), -1, 1) * gamma
    return F.linear(o, w_q)


class SubQSACombineFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
                out_norm_weight, o_proj_weight, gamma):
        ctx.save_for_backward(x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
                              out_norm_weight, o_proj_weight)
        ctx.gamma = gamma
        return _subqsa_combine_eager(x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
                                     out_norm_weight, o_proj_weight, gamma)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output, None, None, None, None, None, None, None, None)


def subqsa_combine_forward(x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
                           out_norm_weight, o_proj_weight, gamma):
    return SubQSACombineFn.apply(x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
                                 out_norm_weight, o_proj_weight, gamma)
