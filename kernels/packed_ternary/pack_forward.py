"""
pack_forward.py — Packed ternary × FP16 forward GEMM via CUDA.

Provides:
    packed_ternary_forward(W_packed, X_fp16) → Y_fp16

The weight tensor W_packed is a (out_features, stride_words) int64 tensor
packed with 16 ternary values per word (see pack_tensor).

Reference (for testing):
    Y_fp16 = F.linear(X_fp16, W_fp16_dequantised)
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

_HAS_FORWARD_KERNEL = False
_forward_fn = None

HERE = os.path.dirname(os.path.abspath(__file__))
CUH_PATH = os.path.join(HERE, "packed_ternary.cuh")
CU_PATH  = os.path.join(HERE, "gemm_forward.cu")

# ── Compile the CUDA kernel ──────────────────────────────────────────────────

def _load_forward_kernel():
    global _HAS_FORWARD_KERNEL, _forward_fn
    if _HAS_FORWARD_KERNEL:
        return

    try:
        from torch.utils.cpp_extension import load_inline

        with open(CUH_PATH) as f:
            cuh_source = f.read()
        with open(CU_PATH) as f:
            cu_source = f.read()

        # Combine: header first, then kernel (remove the #include so it
        # doesn't try to resolve a path at compile time).
        combined = cuh_source + "\n" + cu_source.replace(
            '#include "packed_ternary.cuh"', ""
        )

        _lib = load_inline(
            name="packed_ternary_forward_ext",
            cpp_sources=r"""
            #include <cuda_runtime.h>
            #include <torch/extension.h>

            extern "C" {
                void launch_packed_ternary_forward(
                    const uint32_t* W,
                    const void*     X,
                    void*           Y,
                    int batch_size,
                    int in_features,
                    int out_features,
                    int stride_words,
                    cudaStream_t stream);
            }

            torch::Tensor forward_wrapper(
                torch::Tensor W,
                torch::Tensor X)
            {
                TORCH_CHECK(W.is_cuda() && X.is_cuda(), "W and X must be CUDA tensors");
                TORCH_CHECK(W.dim() == 2, "W must be 2D (out_features, stride_words)");
                TORCH_CHECK(X.dim() == 2, "X must be 2D (batch, in_features)");

                int batch_size   = X.size(0);
                int in_features  = X.size(1);
                int out_features = W.size(0);
                int stride_words = W.size(1);

                auto Y = torch::empty({batch_size, out_features},
                                      torch::dtype(torch::kFloat16).device(X.device()));

                launch_packed_ternary_forward(
                    reinterpret_cast<const uint32_t*>(W.data_ptr<int32_t>()),
                    X.data_ptr<at::Half>(),
                    Y.data_ptr<at::Half>(),
                    batch_size,
                    in_features,
                    out_features,
                    stride_words,
                    nullptr  // default stream
                );

                return Y;
            }

            PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                m.def("forward", &forward_wrapper, "Packed ternary × FP16 forward");
            }
            """,
            cuda_sources=[combined],
            verbose=False,
            extra_cuda_cflags=["-O2"],
        )

        _forward_fn = _lib.forward
        _HAS_FORWARD_KERNEL = True

    except Exception as e:
        print(f"[packed_ternary_forward] Failed to load CUDA kernel: {e}")
        _HAS_FORWARD_KERNEL = False


def has_forward_kernel() -> bool:
    if not _HAS_FORWARD_KERNEL:
        _load_forward_kernel()
    return _HAS_FORWARD_KERNEL


def packed_ternary_forward(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Packed ternary GEMM forward.

    Args:
        W: ``(out_features, stride_words)`` int64 packed tensor.
        X: ``(batch, in_features)`` FP16 tensor on CUDA.

    Returns:
        ``(batch, out_features)`` FP16 tensor.
    """
    if not _HAS_FORWARD_KERNEL:
        _load_forward_kernel()
    if not _HAS_FORWARD_KERNEL:
        raise RuntimeError("Packed ternary forward kernel not available")

    return _forward_fn(W.contiguous(), X.contiguous())


# ── Reference (pure PyTorch, for testing) ───────────────────────────────────

def ref_linear(W_packed: torch.Tensor, X: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Reference ``F.linear`` on dequantised weights.

    This is the **ground truth** the CUDA kernel must match.
    """
    from . import unpack_tensor

    rows = W_packed.shape[0]
    cols_estimate = W_packed.shape[1] * 16  # upper bound on in_features

    # Dequantise packed weights → FP32 → FP16 on the same device as X
    W_fp32 = unpack_tensor(W_packed, rows, cols_estimate, gamma=gamma)
    W_fp16 = W_fp32[:, :X.size(1)].to(device=X.device, dtype=torch.float16).contiguous()

    return F.linear(X, W_fp16)
