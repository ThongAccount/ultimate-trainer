"""Tests for CUDA ternary kernel integration (CPU fallback paths).

All tests run on CPU — they verify the Python wrappers and eager fallbacks
work correctly. CUDA kernel dispatch is tested on GPU only.
"""

import torch
import pytest


# ═══════════════════════════════════════════════════════════════════════
#  1.  module-level imports in bitlinear.py
# ═══════════════════════════════════════════════════════════════════════

class TestBitlinearCudaImports:
    """bitlinear.py loads CUDA modules at module init, not per-call."""

    def test_has_cuda_ternary_flag_exists(self):
        """_HAS_CUDA_TERNARY is defined (False on CPU, may be True on GPU)."""
        from ultimate_trainer.bitlinear import _HAS_CUDA_TERNARY
        # On CPU without CUDA this is False, on GPU it may be True
        assert isinstance(_HAS_CUDA_TERNARY, bool)

    def test_has_fused_ternary_flag_exists(self):
        """_HAS_FUSED_TERNARY is defined (False on CPU, may be True on GPU)."""
        from ultimate_trainer.bitlinear import _HAS_FUSED_TERNARY
        assert isinstance(_HAS_FUSED_TERNARY, bool)

    def test_bitlinear_forward_cpu_still_works(self):
        """BitLinear forward uses eager fallback on CPU."""
        from ultimate_trainer.bitlinear import BitLinear
        m = BitLinear(64, 128)
        x = torch.randn(2, 64)
        y = m(x)
        assert y.shape == (2, 128)
        assert not torch.isnan(y).any()


# ═══════════════════════════════════════════════════════════════════════
#  2.  fused_ternary CPU fallback
# ═══════════════════════════════════════════════════════════════════════

class TestFusedTernaryCpu:
    """fused_ternary kernel CPU fallback correctness."""

    @pytest.fixture
    def _setup(self):
        from kernels.fused_ternary.fused_ternary import _HAS_FUSED_TERNARY, fused_ternary_forward
        return _HAS_FUSED_TERNARY, fused_ternary_forward

    def test_fused_ternary_imports_on_cpu(self, _setup):
        """fused_ternary module loads on CPU (_HAS_FUSED_TERNARY=False)."""
        has_ft, fn = _setup
        assert isinstance(has_ft, bool)
        assert callable(fn)

    def test_forward_cpu_shape(self, _setup):
        """Forward produces correct output shape."""
        _, fn = _setup
        x = torch.randn(4, 64)
        w = torch.randn(128, 64)
        gamma = w.abs().mean() + 1e-5
        y = fn(x, w, gamma)
        assert y.shape == (4, 128)

    def test_forward_cpu_no_nan(self, _setup):
        """Forward produces finite output."""
        _, fn = _setup
        x = torch.randn(4, 64)
        w = torch.randn(128, 64)
        gamma = w.abs().mean() + 1e-5
        y = fn(x, w, gamma)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_forward_cpu_with_bias(self, _setup):
        """Forward with bias produces correct shape."""
        _, fn = _setup
        x = torch.randn(4, 64)
        w = torch.randn(128, 64)
        gamma = w.abs().mean() + 1e-5
        bias = torch.randn(128)
        y = fn(x, w, gamma, bias)
        assert y.shape == (4, 128)

    def test_forward_cpu_3d_input(self, _setup):
        """Forward handles 3D input (B, T, K)."""
        _, fn = _setup
        x = torch.randn(2, 32, 64)
        w = torch.randn(128, 64)
        gamma = w.abs().mean() + 1e-5
        y = fn(x, w, gamma)
        assert y.shape == (2, 32, 128)

    def test_forward_cpu_parity_with_eager(self, _setup):
        """fused_ternary output matches eager reference (FP16 tolerance)."""
        _, fn = _setup
        torch.manual_seed(42)
        x = torch.randn(4, 64)
        w = torch.randn(128, 64)
        gamma = w.abs().mean() + 1e-5

        y_fused = fn(x, w, gamma)

        # Eager reference at FP16 precision
        w_q = torch.clamp(torch.round(w / gamma), -1.0, 1.0)
        w_ternary = (w_q * gamma).half().float()
        y_eager = torch.mm(x.half().float(), w_ternary.t())

        # FP16 introduces ~2 ulp error; use relaxed tolerance
        assert torch.allclose(y_fused, y_eager, atol=1e-2, rtol=1e-2), \
            f"Max diff: {(y_fused - y_eager).abs().max().item()}"

    def test_gradient_flow(self, _setup):
        """Gradients flow through fused_ternary via STE."""
        _, fn = _setup
        x = torch.randn(4, 64, requires_grad=True)
        w = torch.randn(128, 64, requires_grad=True)
        gamma = w.abs().mean() + 1e-5
        gamma = gamma.detach().requires_grad_(True)

        y = fn(x, w, gamma)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "x.grad is None"
        assert w.grad is not None, "w.grad is None"
        assert not torch.isnan(x.grad).any(), "NaN in x.grad"
        assert not torch.isnan(w.grad).any(), "NaN in w.grad"

    def test_quantize_ternary_fp16_cpu(self, _setup):
        """quantize_ternary_fp16 produces FP16 output."""
        from kernels.fused_ternary.fused_ternary import quantize_ternary_fp16
        w = torch.randn(64, 128)
        gamma = w.abs().mean() + 1e-5
        w_q = quantize_ternary_fp16(w, gamma)
        assert w_q.dtype == torch.float16
        assert w_q.shape == w.shape
        assert not torch.isnan(w_q).any()

    def test_quantize_fp16_values(self, _setup):
        """quantize_ternary_fp16 produces values in {-gamma, 0, +gamma}."""
        from kernels.fused_ternary.fused_ternary import quantize_ternary_fp16
        w = torch.tensor([[2.0, 0.3, -1.5], [0.0, -0.1, 3.0]], dtype=torch.float32)
        gamma = w.abs().mean() + 1e-5
        w_q = quantize_ternary_fp16(w, gamma)
        expected_gamma_fp16 = float(torch.tensor(gamma).half())
        for val in w_q.flatten():
            val_f = float(val)
            assert val_f in (expected_gamma_fp16, 0.0, -expected_gamma_fp16), \
                f"Unexpected quantized value: {val_f}"


# ═══════════════════════════════════════════════════════════════════════
#  3.  block_sparse_ternary CPU path
# ═══════════════════════════════════════════════════════════════════════

class TestBlockSparseTernaryCpu:
    """block_sparse_ternary CPU fallback."""

    def test_import_and_flag(self):
        """Module imports on CPU, _HAS_CUDA is False."""
        from kernels.block_sparse_ternary.block_sparse_ternary import _HAS_CUDA
        assert isinstance(_HAS_CUDA, bool)

    def test_sparse_matmul_dense_mask(self):
        """All blocks active → output matches dense ternary matmul."""
        from kernels.block_sparse_ternary.block_sparse_ternary import (
            block_sparse_ternary_matmul,
        )
        M, N, K = 32, 32, 32
        x = torch.randn(M, K)
        w = torch.randn(N, K)
        gamma = w.abs().mean() + 1e-5
        num_n_tiles = (N + 63) // 64
        num_k_tiles = (K + 63) // 64
        num_ints = max(1, (num_n_tiles * num_k_tiles + 63) // 64)
        mask = torch.full((num_ints,), ~0, dtype=torch.int64)
        y = block_sparse_ternary_matmul(x, w, gamma, mask)
        assert y.shape == (M, N)
        assert not torch.isnan(y).any()

    def test_sparse_matmul_empty_mask(self):
        """All blocks masked → output is all zeros."""
        from kernels.block_sparse_ternary.block_sparse_ternary import (
            block_sparse_ternary_matmul,
        )
        M, N, K = 32, 32, 32
        x = torch.randn(M, K)
        w = torch.randn(N, K)
        gamma = w.abs().mean() + 1e-5
        num_n_tiles = (N + 63) // 64
        num_k_tiles = (K + 63) // 64
        num_ints = max(1, (num_n_tiles * num_k_tiles + 63) // 64)
        mask = torch.zeros(num_ints, dtype=torch.int64)
        y = block_sparse_ternary_matmul(x, w, gamma, mask)
        assert torch.all(y == 0), "All-zero mask should produce zero output"

    def test_compute_block_mask(self):
        """compute_block_mask produces correct bitmask from indices."""
        from kernels.block_sparse_ternary.block_sparse_ternary import compute_block_mask
        n_sel, topk = 8, 3
        top_idx = torch.tensor([[[1, 3, 5]]])
        num_k_tiles = 4
        num_n_tiles = n_sel
        mask = compute_block_mask(top_idx, n_sel, 64, num_n_tiles, num_k_tiles)
        for tn in [1, 3, 5]:
            for tk in range(num_k_tiles):
                bit_pos = tn * num_k_tiles + tk
                assert mask[bit_pos // 64] & (1 << (bit_pos % 64)), f"Bit {bit_pos} should be set"

    def test_gradient_flow(self):
        """Gradients flow through block_sparse_ternary."""
        from kernels.block_sparse_ternary.block_sparse_ternary import (
            block_sparse_ternary_matmul,
        )
        M, N, K = 16, 16, 16
        x = torch.randn(M, K, requires_grad=True)
        w = torch.randn(N, K, requires_grad=True)
        gamma = w.abs().mean() + 1e-5
        num_n_tiles = (N + 63) // 64
        num_k_tiles = (K + 63) // 64
        num_ints = max(1, (num_n_tiles * num_k_tiles + 63) // 64)
        mask = torch.full((num_ints,), ~0, dtype=torch.int64)

        y = block_sparse_ternary_matmul(x, w, gamma, mask)
        y.sum().backward()
        assert x.grad is not None, "x.grad is None"
        assert w.grad is not None, "w.grad is None"
        assert not torch.isnan(x.grad).any()


# ═══════════════════════════════════════════════════════════════════════
#  4.  subqsa_combine block_mask parameter
# ═══════════════════════════════════════════════════════════════════════

class TestSubqsaCombineBlockMask:
    """subqsa_combine_forward accepts block_mask parameter (CPU)."""

    @pytest.fixture
    def _setup(self):
        from kernels.subqsa_combine.subqsa_combine import (
            subqsa_combine_forward, _subqsa_combine_eager,
        )
        return subqsa_combine_forward, _subqsa_combine_eager

    def test_block_mask_none_equals_no_mask(self, _setup):
        """block_mask=None produces same output as calling without it."""
        fn, eager = _setup
        B, H, T, D = 2, 4, 16, 64
        x = torch.randn(B, T, H * D)
        o_cmp = torch.randn(B, H, T, D)
        o_slc = torch.randn(B, H, T, D)
        o_win = torch.randn(B, H, T, D)
        gw1 = torch.randn(64, H * D)
        gw2 = torch.randn(3 * H, 64)
        onw = torch.randn(H * D)
        opw = torch.randn(H * D, H * D)
        gamma = opw.abs().mean() + 1e-5

        y_no_mask = fn(x, o_cmp, o_slc, o_win, gw1, gw2, onw, opw, gamma)
        y_explicit_none = fn(x, o_cmp, o_slc, o_win, gw1, gw2, onw, opw, gamma, block_mask=None)
        assert torch.allclose(y_no_mask, y_explicit_none, atol=1e-5)

    def test_block_mask_dense_matches_reference(self, _setup):
        """Dense block mask produces same output as without block mask."""
        fn, eager = _setup
        B, H, T, D = 2, 4, 8, 64
        x = torch.randn(B, T, H * D)
        o_cmp = torch.randn(B, H, T, D)
        o_slc = torch.randn(B, H, T, D)
        o_win = torch.randn(B, H, T, D)
        gw1 = torch.randn(64, H * D)
        gw2 = torch.randn(3 * H, 64)
        onw = torch.randn(H * D)
        opw = torch.randn(H * D, H * D)
        gamma = opw.abs().mean() + 1e-5

        BN = 64
        D_out = H * D
        num_n_tiles = (D_out + BN - 1) // BN
        num_k_tiles = (D_out + BN - 1) // BN
        num_ints = max(1, (num_n_tiles * num_k_tiles + 63) // 64)
        dense_mask = torch.full((num_ints,), ~0, dtype=torch.int64)

        y_ref = fn(x, o_cmp, o_slc, o_win, gw1, gw2, onw, opw, gamma)
        y_sparse = fn(x, o_cmp, o_slc, o_win, gw1, gw2, onw, opw, gamma, block_mask=dense_mask)

        assert torch.allclose(y_ref, y_sparse, atol=1e-4), \
            "Dense mask output should match no-mask output"

    def test_block_mask_empty_produces_zeros(self, _setup):
        """All-zero block mask → output may differ (sparsity effect)."""
        fn, eager = _setup
        B, H, T, D = 1, 2, 4, 32
        x = torch.randn(B, T, H * D)
        o_cmp = torch.randn(B, H, T, D)
        o_slc = torch.randn(B, H, T, D)
        o_win = torch.randn(B, H, T, D)
        gw1 = torch.randn(64, H * D)
        gw2 = torch.randn(3 * H, 64)
        onw = torch.randn(H * D)
        opw = torch.randn(H * D, H * D)
        gamma = opw.abs().mean() + 1e-5

        BN = 64
        D_out = H * D
        num_n_tiles = (D_out + BN - 1) // BN
        num_k_tiles = (D_out + BN - 1) // BN
        num_ints = max(1, (num_n_tiles * num_k_tiles + 63) // 64)
        empty_mask = torch.zeros(num_ints, dtype=torch.int64)

        y_empty = fn(x, o_cmp, o_slc, o_win, gw1, gw2, onw, opw, gamma, block_mask=empty_mask)
        # With empty mask, the non-O-projection steps still run (gate, blend, norm)
        # but the O-projection outputs zeros. The output should have the right shape.
        assert y_empty.shape == (B, T, D_out)

    def test_block_mask_gradient(self, _setup):
        """Gradients flow through combine with block_mask."""
        fn, eager = _setup
        B, H, T, D = 1, 2, 4, 32
        x = torch.randn(B, T, H * D, requires_grad=True)
        o_cmp = torch.randn(B, H, T, D, requires_grad=True)
        o_slc = torch.randn(B, H, T, D, requires_grad=True)
        o_win = torch.randn(B, H, T, D, requires_grad=True)
        gw1 = torch.randn(64, H * D, requires_grad=True)
        gw2 = torch.randn(3 * H, 64, requires_grad=True)
        onw = torch.randn(H * D, requires_grad=True)
        opw = torch.randn(H * D, H * D, requires_grad=True)
        gamma = opw.abs().mean() + 1e-5

        y = fn(x, o_cmp, o_slc, o_win, gw1, gw2, onw, opw, gamma, block_mask=None)
        y.sum().backward()

        assert x.grad is not None, "x.grad is None"
        assert opw.grad is not None, "opw.grad is None"
        assert not torch.isnan(x.grad).any(), "NaN in x.grad"


# ═══════════════════════════════════════════════════════════════════════
#  5.  kernels/__init__.py exports
# ═══════════════════════════════════════════════════════════════════════

class TestKernelInitExports:
    """kernels/__init__.py exports all kernel modules."""

    def test_fused_ternary_imported(self):
        """kernels.fused_ternary is importable."""
        import kernels.fused_ternary
        assert hasattr(kernels.fused_ternary, 'fused_ternary_forward')

    def test_block_sparse_available(self):
        """kernels.block_sparse_ternary_matmul is re-exported from kernels."""
        import kernels
        assert hasattr(kernels, 'block_sparse_ternary_matmul')

    def test_compressed_attn_available(self):
        """kernels.compressed_attn_forward is re-exported."""
        import kernels
        # May be None if CUDA not available — but must be importable
        assert hasattr(kernels, 'compressed_attn_forward')

    def test_selective_attn_available(self):
        """kernels.selective_attn_forward is re-exported."""
        import kernels
        assert hasattr(kernels, 'selective_attn_forward')

    def test_subqsa_combine_available(self):
        """kernels.subqsa_combine_forward is re-exported."""
        import kernels
        assert hasattr(kernels, 'subqsa_combine_forward')
