"""CUDA-accelerated kernels for the Ultimate model.

Import failures are logged with their exception so that a broken kernel
extension is debuggable instead of silently degrading to CPU fallbacks.
"""

import logging

logger = logging.getLogger(__name__)

from . import elementwise
from . import ternary
from . import fused_ternary

try:
    from kernels.compressed_attn.compressed_attn import compressed_attn_forward, _HAS_COMPRESSED_ATTN
except Exception as e:  # noqa: BLE001 — log & degrade, do not crash import
    logger.error("[kernels] compressed_attn import failed: %r", e)
    compressed_attn_forward = None
    _HAS_COMPRESSED_ATTN = False

try:
    from kernels.selective_attn.selective_attn import selective_attn_forward, _HAS_SELECTIVE_ATTN
except Exception as e:  # noqa: BLE001
    logger.error("[kernels] selective_attn import failed: %r", e)
    selective_attn_forward = None
    _HAS_SELECTIVE_ATTN = False

try:
    from kernels.block_sparse_ternary.block_sparse_ternary import (
        block_sparse_ternary_matmul, compute_block_mask, _HAS_CUDA as _HAS_BLOCK_SPARSE_CUDA,
    )
    _HAS_BLOCK_SPARSE = _HAS_BLOCK_SPARSE_CUDA
except Exception as e:  # noqa: BLE001
    logger.error("[kernels] block_sparse_ternary import failed: %r", e)
    block_sparse_ternary_matmul = None
    compute_block_mask = None
    _HAS_BLOCK_SPARSE = False

try:
    from kernels.subqsa_combine.subqsa_combine import subqsa_combine_forward, _HAS_SUBQSA_COMBINE
except Exception as e:  # noqa: BLE001
    logger.error("[kernels] subqsa_combine import failed: %r", e)
    subqsa_combine_forward = None
    _HAS_SUBQSA_COMBINE = False
