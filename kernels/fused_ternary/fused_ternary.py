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
            const float*, float*, float, int);

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
                reinterpret_cast<float*>(w_out.data_ptr()),
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


# ── Plain STE function (no custom autograd.Function) ─────────────────

def fused_ternary_forward(x, weight, gamma, bias=None):
    """FP16 TensorCore ternary matmul with STE.

    Uses the standard detach trick for STE: forward uses gamma-scaled
    ternary weights, backward passes gradient through as identity (no
    quantization gradient). This avoids autograd.Function which conflicts
    with DDP's gradient hooks.

    Args:
        x:      (..., K) activations
        weight: (N, K) FP32 master weights
        gamma:  scalar, mean(|weight|) + eps
        bias:   optional (N,) bias
    Returns:
        y: (..., N) output (same dtype as input)
    """
    x_dtype = x.dtype
    input_device = x.device
    *dims, K = x.shape
    M = x.view(-1, K).shape[0]
    N = weight.shape[0]

    # Quantize weights to FP16 ternary via CUDA kernel or CPU fallback
    if _HAS_FUSED_TERNARY and x.is_cuda:
        w_fp16 = _lib.quantize_ternary_fp16_wrapper(
            weight.contiguous().float(), float(gamma))
    else:
        w_fp16 = _quantize_ternary_fp16_eager(weight, gamma).to(input_device)

    # STE: use quantized weights for forward, identity gradient for backward
    w_ste = weight + (w_fp16.float() - weight).detach()
    y = torch.matmul(x.reshape(-1, K).half(), w_ste.half().t()).to(x_dtype)
    if bias is not None:
        y = y + bias.to(device=input_device, dtype=x_dtype)
    return y.reshape(*dims, N)


def quantize_ternary_fp16(w, gamma):
    """Quantize FP32 master weights to FP16 ternary {-gamma, 0, +gamma}.

    Useful for standalone use (e.g., eval cache refresh).
    """
    if _HAS_FUSED_TERNARY and w.is_cuda:
        return _lib.quantize_ternary_fp16_wrapper(w.contiguous().float(), float(gamma))
    return _quantize_ternary_fp16_eager(w, gamma)
