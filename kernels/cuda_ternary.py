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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_HAS_CUDA_KERNEL = False
_forward_fn = None
_backward_fn = None

# ── Compile / load the CUDA kernel (using load_inline like other SubQSA kernels) ──

_CUDA_SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ternary", "ternary_matmul.cu"
)

try:
    from torch.utils.cpp_extension import load_inline

    with open(_CUDA_SOURCE) as f:
        cuda_source = f.read()

    _lib = load_inline(
        name="ternary_matmul_cuda_ext",
        cpp_sources=r"""
        #include <cuda_runtime.h>
        #include <vector>
        #include <torch/extension.h>

        extern "C" {
            void forward_ternary_matmul(
                const float* x, const float* w, float* y,
                float gamma, int M, int N, int K,
                cudaStream_t stream);
            void backward_dx_ternary(
                const float* dy, const float* w, float* dx,
                float gamma, int M, int N, int K,
                cudaStream_t stream);
        }

        torch::Tensor forward_wrapper(
            torch::Tensor x, torch::Tensor w, double gamma) {
            TORCH_CHECK(x.is_cuda() && w.is_cuda(), "Inputs must be CUDA tensors");
            TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "Inputs must be contiguous");
            TORCH_CHECK(x.dim() == 2, "x must be 2D (M, K)");
            TORCH_CHECK(w.dim() == 2, "w must be 2D (N, K)");
            TORCH_CHECK(x.size(1) == w.size(1), "x and w must have same K dimension");
            TORCH_CHECK(x.dtype() == torch::kFloat32 && w.dtype() == torch::kFloat32,
                        "Inputs must be float32");

            int M = x.size(0);
            int N = w.size(0);
            int K = x.size(1);
            auto y = torch::empty({M, N}, x.options());

            forward_ternary_matmul(
                x.data_ptr<float>(), w.data_ptr<float>(),
                y.data_ptr<float>(),
                static_cast<float>(gamma),
                M, N, K, at::cuda::getCurrentCUDAStream());

            return y;
        }

        torch::Tensor backward_dx_wrapper(
            torch::Tensor dy, torch::Tensor w, double gamma) {
            TORCH_CHECK(dy.is_cuda() && w.is_cuda(), "Inputs must be CUDA tensors");
            TORCH_CHECK(dy.is_contiguous() && w.is_contiguous(), "Inputs must be contiguous");
            TORCH_CHECK(dy.dim() == 2, "dy must be 2D (M, N)");
            TORCH_CHECK(w.dim() == 2, "w must be 2D (N, K)");
            TORCH_CHECK(dy.size(1) == w.size(0), "dy(N) must match w(N)");
            TORCH_CHECK(dy.dtype() == torch::kFloat32 && w.dtype() == torch::kFloat32,
                        "Inputs must be float32");

            int M = dy.size(0);
            int N = dy.size(1);
            int K = w.size(1);
            auto dx = torch::empty({M, K}, dy.options());

            backward_dx_ternary(
                dy.data_ptr<float>(), w.data_ptr<float>(),
                dx.data_ptr<float>(),
                static_cast<float>(gamma),
                M, N, K, at::cuda::getCurrentCUDAStream());

            return dx;
        }
        """,
        cuda_sources=cuda_source,
        functions=["forward_wrapper", "backward_dx_wrapper"],
        verbose=False,
        extra_cuda_cflags=["-DBUILD_AS_SHARED"],
    )

    _forward_fn = _lib.forward_wrapper
    _backward_fn = _lib.backward_dx_wrapper
    _HAS_CUDA_KERNEL = True
except Exception:
    _HAS_CUDA_KERNEL = False

HAS_CUDA_KERNEL = _HAS_CUDA_KERNEL


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

        if _HAS_CUDA_KERNEL and x.is_cuda:
            # CUDA kernel path: computes y_raw = x @ Q(W)^T where Q(W) ∈ {-1,0,+1}
            y = torch.empty(M, N, device=x.device, dtype=torch.float32)
            gamma_scalar = gamma.item() if isinstance(gamma, torch.Tensor) else gamma
            y = _forward_fn(x_2d.contiguous(), weight.contiguous(), gamma_scalar)
            # Scale by gamma: y = gamma * (x @ Q(W)^T) → ternary weights are {-γ, 0, +γ}
            y = y * gamma_scalar
        else:
            # Eager fallback: quant + matmul with gamma-scaled ternary weights
            w_q = torch.clamp(torch.round(weight / gamma), -1.0, 1.0)
            w_ternary = w_q * gamma  # {-γ, 0, +γ}
            y = F.linear(x_2d, w_ternary)

        if bias is not None:
            y = y + bias

        return y.reshape(*dims, N)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Computes:
            dx = dy @ Q(W)        (gradient w.r.t. x)
            dw = dy^T @ x         (STE — no gradient through W quant)
            d_gamma = None
            dbias = dy.sum(dim=...)
        """
        x, weight, gamma = ctx.saved_tensors
        bias = ctx.bias

        *dims, N = grad_output.shape
        M = grad_output.view(-1, N).shape[0]
        K = weight.shape[1]
        dy_2d = grad_output.reshape(M, N)

        if _HAS_CUDA_KERNEL and x.is_cuda:
            # CUDA backward kernel: dx_raw = dy @ Q(W)  then scale by gamma
            # Since forward = gamma * (x @ Q(W)^T), backward = gamma * (dy @ Q(W))
            dx = _backward_fn(dy_2d.contiguous(), weight.contiguous(), gamma.item())
            dx = dx * gamma.item()
        else:
            # Eager fallback: STE backward (no gradient through weight quant)
            with torch.no_grad():
                w_q = torch.clamp(torch.round(weight / gamma), -1.0, 1.0)
            # dx = dy @ (gamma * W_q) — gamma-scaled ternary
            dx = F.linear(dy_2d, (w_q * gamma).t())

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

    Uses gamma-scaled ternary weights {-γ, 0, +γ} so output magnitudes
    match FP linear outputs.

    During training: uses TernaryMatmulFn (autograd-aware).
    During eval: uses CUDA kernel or eager fallback with cached ternary.
    """
    if weight.requires_grad or x.requires_grad:
        return TernaryMatmulFn.apply(x, weight, gamma, bias)
    else:
        if _HAS_CUDA_KERNEL:
            return TernaryMatmulFn.apply(x, weight, gamma, bias)
        w_q = torch.clamp(torch.round(weight / gamma), -1.0, 1.0)
        return F.linear(x, w_q * gamma, bias)
