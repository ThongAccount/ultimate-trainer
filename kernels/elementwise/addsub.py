"""
addsub.py — Python wrapper for GPU-accelerated vector add/sub kernels.

Provides:
  vec_add(a, b)           → c  (element-wise a + b)
  vec_sub(a, b)           → c  (element-wise a - b)
  vec_add_sub(a, b)       → add, sub  (fused a+b and a-b, single kernel launch)

All functions accept and return CUDA torch.Tensors (float32).
If the CUDA extension is unavailable, falls back to plain PyTorch.
"""

import os
import torch
from typing import Tuple

# ── JIT-compile the CUDA kernel via PyTorch's load_inline ──────────────

_CUDA_SOURCE = os.path.join(os.path.dirname(__file__), "addsub.cu")

_extension_loaded = False
_vec_add = None
_vec_sub = None
_vec_add_sub = None


def _load_extension():
    global _extension_loaded, _vec_add, _vec_sub, _vec_add_sub
    if _extension_loaded:
        return True

    try:
        from torch.utils.cpp_extension import load_inline

        with open(_CUDA_SOURCE) as f:
            cuda_source = f.read()

        # Parse out just the kernel + extern "C" functions
        module = load_inline(
            name="addsub_extension",
            cpp_sources="""
            #include <cuda_runtime.h>
            #include <cstddef>

            extern "C" {
                void vec_add(const float* a, const float* b, float* c, size_t N, cudaStream_t stream);
                void vec_sub(const float* a, const float* b, float* c, size_t N, cudaStream_t stream);
                void vec_add_sub(const float* a, const float* b, float* add, float* sub, size_t N, cudaStream_t stream);
            }

            torch::Tensor vec_add_wrapper(torch::Tensor a, torch::Tensor b) {
                TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors");
                TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "Inputs must be contiguous");
                TORCH_CHECK(a.numel() == b.numel(), "Inputs must have same number of elements");
                TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "Inputs must be float32");

                auto c = torch::empty_like(a);
                auto N = a.numel();
                vec_add(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), N, at::cuda::getCurrentCUDAStream());
                return c;
            }

            torch::Tensor vec_sub_wrapper(torch::Tensor a, torch::Tensor b) {
                TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors");
                TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "Inputs must be contiguous");
                TORCH_CHECK(a.numel() == b.numel(), "Inputs must have same number of elements");
                TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "Inputs must be float32");

                auto c = torch::empty_like(a);
                auto N = a.numel();
                vec_sub(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), N, at::cuda::getCurrentCUDAStream());
                return c;
            }

            std::vector<torch::Tensor> vec_add_sub_wrapper(torch::Tensor a, torch::Tensor b) {
                TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors");
                TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "Inputs must be contiguous");
                TORCH_CHECK(a.numel() == b.numel(), "Inputs must have same number of elements");
                TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "Inputs must be float32");

                auto add = torch::empty_like(a);
                auto sub = torch::empty_like(a);
                auto N = a.numel();
                vec_add_sub(a.data_ptr<float>(), b.data_ptr<float>(),
                            add.data_ptr<float>(), sub.data_ptr<float>(), N, at::cuda::getCurrentCUDAStream());
                return {add, sub};
            }
            """,
            cuda_sources=cuda_source,
            functions=["vec_add_wrapper", "vec_sub_wrapper", "vec_add_sub_wrapper"],
            verbose=False,
        )

        _vec_add = module.vec_add_wrapper
        _vec_sub = module.vec_sub_wrapper
        _vec_add_sub = module.vec_add_sub_wrapper
        _extension_loaded = True
        return True

    except Exception as e:
        print(f"[addsub] CUDA extension load failed ({e}), falling back to PyTorch")
        return False


# ── Public API ──────────────────────────────────────────────────────


def vec_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """c = a + b (element-wise, GPU-accelerated)."""
    assert a.device == b.device, "Inputs must be on the same device"
    assert a.shape == b.shape, "Inputs must have the same shape"

    if a.is_cuda and _load_extension():
        return _vec_add(a.contiguous(), b.contiguous())

    return a + b


def vec_sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """c = a - b (element-wise, GPU-accelerated)."""
    assert a.device == b.device, "Inputs must be on the same device"
    assert a.shape == b.shape, "Inputs must have the same shape"

    if a.is_cuda and _load_extension():
        return _vec_sub(a.contiguous(), b.contiguous())

    return a - b


def vec_add_sub(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused a+b and a-b in a single kernel launch.

    Returns (add, sub) where add = a + b, sub = a - b.
    When using CUDA, this streams a and b from HBM only once (vs twice for
    separate add/sub calls), halving memory traffic.
    """
    assert a.device == b.device, "Inputs must be on the same device"
    assert a.shape == b.shape, "Inputs must have the same shape"

    if a.is_cuda and _load_extension():
        add, sub = _vec_add_sub(a.contiguous(), b.contiguous())
        return add, sub

    return a + b, a - b
