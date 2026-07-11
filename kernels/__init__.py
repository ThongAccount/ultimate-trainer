"""CUDA-accelerated kernels for the Ultimate model."""

from . import elementwise
from . import ternary

try:
    from kernels.compressed_attn.compressed_attn import compressed_attn_forward, _HAS_COMPRESSED_ATTN
except:
    compressed_attn_forward = None
    _HAS_COMPRESSED_ATTN = False

try:
    from kernels.selective_attn.selective_attn import selective_attn_forward, _HAS_SELECTIVE_ATTN
except:
    selective_attn_forward = None
    _HAS_SELECTIVE_ATTN = False

try:
    from kernels.block_sparse_ternary.block_sparse_ternary import block_sparse_ternary_matmul, compute_block_mask, _HAS_BLOCK_SPARSE
except:
    block_sparse_ternary_matmul = None
    compute_block_mask = None
    _HAS_BLOCK_SPARSE = False

try:
    from kernels.subqsa_combine.subqsa_combine import subqsa_combine_forward, _HAS_SUBQSA_COMBINE
except:
    subqsa_combine_forward = None
    _HAS_SUBQSA_COMBINE = False
