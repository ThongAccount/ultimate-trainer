"""Block-sparse ternary matmul: dense ternary matmul with block-skip bitmask.

CUDA kernel with eager PyTorch CPU fallback.
"""

import os
import torch
import torch.nn.functional as F

_HAS_CUDA = False
_block_sparse_lib = None

# ── Compile CUDA kernel ────────────────────────────────────────────────

_CUDA_SOURCE = os.path.join(os.path.dirname(__file__), "block_sparse_ternary.cu")

try:
    from torch.utils.cpp_extension import load_inline

    with open(_CUDA_SOURCE) as f:
        cuda_source = f.read()

    # Append host-side launch wrapper for the kernel (CUDA C++, compiled by nvcc)
    cuda_source += r"""
    #include <cuda_runtime.h>
    #include <cuda_fp16.h>
    #include <cstdint>

    extern "C" void launch_block_sparse_ternary(
        const float* x_f32, const float* w, float* y_f32,
        const uint64_t* block_mask, float gamma,
        int M, int N, int K, int BM, int BN, int BK, int num_k_tiles) {

        dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
        block_sparse_ternary_kernel<<<grid, 256>>>(
            reinterpret_cast<const half*>(x_f32),
            w,
            reinterpret_cast<half*>(y_f32),
            block_mask, gamma, M, N, K, BM, BN, BK, num_k_tiles);
    }
    """

    _block_sparse_lib = load_inline(
        name="block_sparse_ternary_extension",
        cpp_sources=r"""
        #include <cuda_runtime.h>
        #include <vector>
        #include <torch/extension.h>

        extern "C" void launch_block_sparse_ternary(
            const float*, const float*, float*,
            const uint64_t*, float,
            int, int, int, int, int, int, int);

        torch::Tensor block_sparse_ternary_wrapper(
            torch::Tensor x, torch::Tensor w,
            torch::Tensor block_mask, double gamma,
            int BM, int BN, int BK) {

            int M = x.size(0);
            int N = w.size(0);
            int K = x.size(1);
            TORCH_CHECK(x.is_cuda() && w.is_cuda(), "Inputs must be CUDA tensors");
            TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "Inputs must be contiguous");
            TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");

            int num_k_tiles = (K + BK - 1) / BK;
            auto y = torch::empty({M, N}, torch::TensorOptions()
                .dtype(torch::kFloat32).device(x.device()));

            launch_block_sparse_ternary(
                x.data_ptr<float>(),
                w.data_ptr<float>(),
                y.data_ptr<float>(),
                reinterpret_cast<const uint64_t*>(block_mask.data_ptr()),
                static_cast<float>(gamma),
                M, N, K, BM, BN, BK, num_k_tiles);

            return y;
        }
        """,
        cuda_sources=cuda_source,
        functions=["block_sparse_ternary_wrapper"],
        verbose=False,
    )
    _HAS_CUDA = True
except Exception:
    _HAS_CUDA = False


def compute_block_mask(top_idx, n_sel, block_size, num_n_tiles, num_k_tiles):
    """Convert top-K indices to a block-sparse bitmask.

    Each selected block's K-range is fully active. Non-selected blocks are masked out.
    """
    num_tiles = num_n_tiles * num_k_tiles
    num_ints = max(1, (num_tiles + 63) // 64)
    mask = torch.zeros(num_ints, dtype=torch.int64, device=top_idx.device)
    for i in range(top_idx.size(-1)):
        tile_n = top_idx[0, 0, i].item()
        for tk in range(num_k_tiles):
            bit_pos = tile_n * num_k_tiles + tk
            mask[bit_pos // 64] |= 1 << (bit_pos % 64)
    return mask


def _block_sparse_ternary_eager(x, weight, gamma, block_mask, BM=64, BN=64):
    """PyTorch reference with block mask."""
    w_q = torch.clamp(torch.round(weight / gamma), -1, 1)
    y = F.linear(x.float(), w_q.float())
    N = weight.size(0)
    num_n_tiles = (N + BN - 1) // BN
    num_k_tiles = (x.size(1) + 63) // 64
    for tn in range(num_n_tiles):
        for tk in range(num_k_tiles):
            bit = tn * num_k_tiles + tk
            if not (block_mask[bit // 64] & (1 << (bit % 64))):
                y[:, tn*BN:(tn+1)*BN] = 0
    return y.to(x.dtype)


class BlockSparseTernaryFn(torch.autograd.Function):
    """Block-sparse ternary matmul with STE backward."""

    @staticmethod
    def forward(ctx, x, weight, gamma, block_mask, BM=64, BN=64, BK=32):
        ctx.save_for_backward(x, weight, gamma, block_mask)
        ctx.BM = BM
        ctx.BN = BN
        ctx.BK = BK

        # ── CUDA path ──
        if _HAS_CUDA and x.is_cuda:
            y = _block_sparse_lib.block_sparse_ternary_wrapper(
                x.contiguous().float(),
                weight.contiguous().float(),
                block_mask.contiguous().long(),
                float(gamma),
                BM, BN, BK,
            )
            return y.to(x.dtype)

        # ── CPU fallback ──
        return _block_sparse_ternary_eager(x, weight, gamma, block_mask, BM, BN)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, gamma, block_mask = ctx.saved_tensors
        # STE backward: dx = dy @ Q(W), dw = dy^T @ x (identity through quant)
        with torch.no_grad():
            w_q = torch.clamp(torch.round(weight / gamma), -1, 1)
        dx = grad_output @ w_q.to(grad_output.dtype)
        dw = grad_output.reshape(-1, grad_output.shape[-1]).t() @ x.reshape(-1, x.shape[-1])
        return dx, dw, None, None, None, None, None


def block_sparse_ternary_matmul(x, weight, gamma, block_mask, BM=64, BN=64, BK=32):
    """Block-sparse ternary matmul.

    Args:
        x: (M, K) activations
        weight: (N, K) FP32 master weights
        gamma: scalar, mean(|weight|) + eps
        block_mask: (num_u64,) int64 bitmask
        BM, BN, BK: tile sizes
    Returns:
        y: (M, N) output
    """
    return BlockSparseTernaryFn.apply(x, weight, gamma, block_mask, BM, BN, BK)
