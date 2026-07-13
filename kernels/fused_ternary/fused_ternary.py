"""FP16 TensorCore ternary matmul for BitNet b1.58.

Strategy:
  w_fp16 = quantize_ternary_fp16(W_fp32, gamma)  # CUDA kernel, element-wise
  y = torch.matmul(x_fp16, w_fp16.t())            # cuBLAS FP16 TensorCore HMMA

This eliminates the FP32 weight-materialization step and lets cuBLAS FP16
TensorCores handle the matmul (~2-3x faster than FP32 on H100).
"""

import os
import torch
import torch.nn.functional as F

_HAS_FUSED_TERNARY = False
_lib = None

# ── Compile the quantize kernel ─────────────────────────────────────────

_CU_SOURCE = os.path.join(os.path.dirname(__file__), "fused_ternary_kernel.cu")

try:
    from torch.utils.cpp_extension import load_inline

    with open(_CU_SOURCE) as f:
        cuda_source = f.read()

    _lib = load_inline(
        name="fused_ternary_extension",
        cpp_sources=r"""
        #include <cuda_runtime.h>
        #include <torch/extension.h>

        extern "C" void launch_quantize_ternary_fp16(
            const float*, __half*, float, int);

        torch::Tensor quantize_ternary_fp16_wrapper(
            torch::Tensor w, double gamma) {
            TORCH_CHECK(w.is_cuda(), "w must be CUDA tensor");
            TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
            TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");

            int num_elements = w.numel();
            auto w_out = torch::empty_like(w, torch::TensorOptions()
                .dtype(torch::kHalf).device(w.device()));

            launch_quantize_ternary_fp16(
                w.data_ptr<float>(),
                reinterpret_cast<__half*>(w_out.data_ptr()),
                static_cast<float>(gamma),
                num_elements);

            return w_out;
        }
        """,
        cuda_sources=cuda_source,
        functions=["quantize_ternary_fp16_wrapper"],
        verbose=False,
    )
    _HAS_FUSED_TERNARY = True
except Exception:
    _HAS_FUSED_TERNARY = False


# ── FP16 quantize fallback (CPU) ────────────────────────────────────────

def _quantize_ternary_fp16_eager(w, gamma):
    """PyTorch fallback: ternary quantize, output FP16."""
    w_q = torch.clamp(torch.round(w / gamma), -1.0, 1.0)
    return (w_q * gamma).half()


# ── autograd.Function ───────────────────────────────────────────────────

class FusedTernaryFn(torch.autograd.Function):
    """FP16 ternary matmul with on-the-fly quantization + STE backward.

    Forward:  quantize FP32 weights to FP16 ternary, then matmul in FP16
    Backward: dx = dy @ W_fp16_ternary  (re-quantized), dw = dy^T @ x (STE)
    """

    @staticmethod
    def forward(ctx, x, weight, gamma, bias=None):
        """
        Args:
            x:      (..., K) activations (any dtype — cast internally)
            weight: (N, K) FP32 master weights
            gamma:  scalar tensor, mean(|weight|) + eps
            bias:   optional (N,) bias
        Returns:
            y: (..., N) output (in input dtype)
        """
        x_dtype = x.dtype
        input_device = x.device

        # Quantize weights to FP16 ternary
        if _HAS_FUSED_TERNARY and x.is_cuda:
            w_fp16 = _lib.quantize_ternary_fp16_wrapper(
                weight.contiguous().float(), float(gamma))
        else:
            w_fp16 = _quantize_ternary_fp16_eager(weight, gamma).to(input_device)

        # Save for backward: x (as FP16), w_fp16, gamma
        x_fp16 = x.contiguous().to(torch.float16)
        ctx.save_for_backward(x_fp16, w_fp16, gamma)
        ctx.bias = bias is not None
        ctx.x_shape = x.shape

        # FP16 matmul via cuBLAS TensorCore HMMA
        *dims, K = x.shape
        x_2d = x_fp16.reshape(-1, K)
        y = torch.matmul(x_2d, w_fp16.t()).to(x_dtype)

        if bias is not None:
            y = y + bias.to(device=input_device, dtype=x_dtype)

        return y.reshape(*dims, -1)

    @staticmethod
    def backward(ctx, grad_output):
        x_fp16, w_fp16, gamma = ctx.saved_tensors
        has_bias = ctx.bias
        *dims, N = grad_output.shape
        K = x_fp16.shape[-1]
        dy = grad_output.reshape(-1, N)
        x_2d = x_fp16.reshape(-1, K)

        # dx = dy @ w_fp16  (FP16 TensorCore matmul)
        dx = torch.matmul(dy.to(torch.float16), w_fp16).to(grad_output.dtype)
        dx = dx.reshape(*dims, K)

        # dw = dy^T @ x  (STE — identity through ternary quant)
        dw = torch.mm(dy.t(), x_2d.to(grad_output.dtype))

        d_gamma = None
        dbias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1))) if has_bias else None

        return dx, dw, d_gamma, dbias


# ── Public API ──────────────────────────────────────────────────────────

def fused_ternary_forward(x, weight, gamma, bias=None):
    """FP16 TensorCore ternary matmul.

    Args:
        x:      (..., K) activations
        weight: (N, K) FP32 master weights
        gamma:  scalar, mean(|weight|) + eps
        bias:   optional (N,) bias
    Returns:
        y: (..., N) output (same dtype as input)
    """
    return FusedTernaryFn.apply(x, weight, gamma, bias)


def quantize_ternary_fp16(w, gamma):
    """Quantize FP32 master weights to FP16 ternary {-gamma, 0, +gamma}.

    Useful for standalone use (e.g., eval cache refresh).
    """
    if _HAS_FUSED_TERNARY and w.is_cuda:
        return _lib.quantize_ternary_fp16_wrapper(w.contiguous().float(), float(gamma))
    return _quantize_ternary_fp16_eager(w, gamma)
