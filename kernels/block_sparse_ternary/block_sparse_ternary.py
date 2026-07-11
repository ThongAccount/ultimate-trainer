"""Block-sparse ternary matmul: dense ternary matmul with block-skip bitmask."""

import torch
import torch.nn.functional as F
import math

_HAS_BLOCK_SPARSE = False
_block_sparse_lib = None


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
    @staticmethod
    def forward(ctx, x, weight, gamma, block_mask):
        ctx.save_for_backward(x, weight, gamma, block_mask)
        return _block_sparse_ternary_eager(x, weight, gamma, block_mask)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, gamma, block_mask = ctx.saved_tensors
        w_q = torch.clamp(torch.round(weight / gamma), -1, 1)
        dx = grad_output @ w_q.to(grad_output.dtype)
        return dx, None, None, None


def block_sparse_ternary_matmul(x, weight, gamma, block_mask):
    return BlockSparseTernaryFn.apply(x, weight, gamma, block_mask)
