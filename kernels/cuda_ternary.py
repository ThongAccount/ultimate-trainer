"""
PyTorch wrapper for the CUDA C++ ternary matmul kernel (ternary_matmul.cu).

Provides a torch.autograd.Function so the fused ternary matmul can be used
during training (forward = add/sub only, backward = STE through weights,
proper gradient through activations via backward_dx_ternary).

Usage:
    from kernels.cuda_ternary import TernaryMatmulFn, HAS_CUDA_KERNEL

    if HAS_CUDA_KERNEL:
        y = TernaryMatmulFn.apply(x, weight, gamma, bias)
    else:
        y = F.linear(x, weight_ternary, bias)   # fallback
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_HAS_CUDA_KERNEL = False
_ternary_lib = None  # loaded shared library

# ── Compile / load the CUDA kernel ─────────────────────────────────────

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
_CU_SOURCE = os.path.join(_KERNEL_DIR, "ternary", "ternary_matmul.cu")

try:
    from torch.utils.cpp_extension import load

    _ternary_lib = load(
        name="ternary_matmul_cuda",
        sources=[_CU_SOURCE],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    HAS_CUDA_KERNEL = True
except Exception:
    HAS_CUDA_KERNEL = False


# ── Helper: compute gamma ──────────────────────────────────────────────

def compute_gamma(weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """γ = mean(|W|) + eps — same as kernels.ternary_matmul.compute_gamma."""
    return weight.abs().mean() + eps


# ── autograd.Function ──────────────────────────────────────────────────

class TernaryMatmulFn(torch.autograd.Function):
    """Fused ternary matmul with on-the-fly weight quantization.

    Forward:  y = x @ Q(W)^T   (add/sub only, no multiplications)
    Backward: dx = dy @ Q(W)   (STE for W, proper grad for x, dy)

    Uses the CUDA kernel from kernels/ternary/ternary_matmul.cu when
    available; falls back to eager PyTorch otherwise.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor,
                gamma: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:      (..., K) activations
            weight: (N, K) FP32 master weights
            gamma:  scalar, mean(|W|) + eps
            bias:   optional (N,) bias
        Returns:
            y: (..., N) output
        """
        # Save for backward
        ctx.save_for_backward(x, weight, gamma)
        if bias is not None:
            ctx.bias = bias
        else:
            ctx.bias = None

        # Flatten batch dims
        *dims, K = x.shape
        M = x.view(-1, K).shape[0]
        N = weight.shape[0]
        x_2d = x.reshape(M, K)

        if HAS_CUDA_KERNEL:
            # CUDA kernel path
            y = torch.empty(M, N, device=x.device, dtype=torch.float32)
            gamma_scalar = gamma.item() if isinstance(gamma, torch.Tensor) else gamma
            _ternary_lib.forward_ternary_matmul(
                x_2d.contiguous(), weight.contiguous(), y,
                gamma_scalar, M, N, K,
            )
        else:
            # Eager fallback: quant + matmul
            w_q = torch.clamp(torch.round(weight / gamma), -1.0, 1.0)
            y = F.linear(x_2d, w_q)

        if bias is not None:
            y = y + bias

        return y.reshape(*dims, N)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Computes:
            dx = dy @ Q(W)        (gradient w.r.t. x)
            dw = None              (STE — no gradient through W quant)
            d_gamma = None
            dbias = dy.sum(dim=...)
        """
        x, weight, gamma = ctx.saved_tensors
        bias = ctx.bias

        *dims, N = grad_output.shape
        M = grad_output.view(-1, N).shape[0]
        K = weight.shape[1]
        dy_2d = grad_output.reshape(M, N)

        if HAS_CUDA_KERNEL:
            # CUDA backward kernel: dx = dy @ Q(W)
            dx = torch.empty(M, K, device=x.device, dtype=torch.float32)
            gamma_scalar = gamma.item() if isinstance(gamma, torch.Tensor) else gamma
            _ternary_lib.backward_dx_ternary(
                dy_2d.contiguous(), weight.contiguous(), dx,
                gamma_scalar, M, N, K,
            )
        else:
            # Eager fallback: STE backward (no gradient through weight quant)
            with torch.no_grad():
                w_q = torch.clamp(torch.round(weight / gamma), -1.0, 1.0)
            # dx = dy @ W_q — same add/sub trick
            dx = F.linear(dy_2d, w_q.t())

        dx = dx.reshape(*dims, K)

        # STE for weight: gradient passes through as if no quantization
        dw = grad_output.reshape(-1, N).t() @ x.reshape(-1, K)

        d_gamma = None

        dbias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1))) if bias is not None else None

        return dx, dw, d_gamma, dbias


# ── Convenience module ─────────────────────────────────────────────────

def fused_ternary_linear(x: torch.Tensor, weight: torch.Tensor,
                          gamma: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """Drop-in for F.linear with fused ternary quantization.

    During training: uses TernaryMatmulFn (autograd-aware).
    During eval: uses CUDA kernel or eager fallback with cached ternary.
    """
    if weight.requires_grad or x.requires_grad:
        return TernaryMatmulFn.apply(x, weight, gamma, bias)
    else:
        # Eval / inference path
        if HAS_CUDA_KERNEL:
            return TernaryMatmulFn.apply(x, weight, gamma, bias)
        w_q = torch.clamp(torch.round(weight / gamma), -1.0, 1.0)
        return F.linear(x, w_q, bias)
