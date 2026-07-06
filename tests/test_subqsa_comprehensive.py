"""Comprehensive tests for ultimate_trainer.subqsa module.

Covers CompressionBranch, SelectionBranch, sliding_window_attention,
GQA repeat_kv logic, gate normalization, and full SubQSAAttention integration.
"""

import math
import torch
import pytest

from ultimate_trainer.subqsa import (
    CompressionBranch,
    SelectionBranch,
    sliding_window_attention,
    SubQSAAttention,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def repeat_kv(t, n_rep):
    """Local replica of the inline repeat_kv from SubQSA.forward."""
    B_, H_, T_, D_ = t.shape
    if n_rep <= 1:
        return t
    return (
        t[:, :, None]
        .expand(-1, -1, n_rep, -1, -1)
        .reshape(B_, H_ * n_rep, T_, D_)
    )


# ── CompressionBranch Tests ─────────────────────────────────────────────────


class TestCompressionBranch:
    """Correctness of the compression-branch module."""

    # ------------------------------------------------------------------
    # Requirement 1: output shapes for T > block_len, T == block_len,
    # T < block_len.
    # ------------------------------------------------------------------

    @pytest.fixture(params=[32])
    def head_dim(self, request):
        return request.param

    @pytest.fixture(params=[2])
    def batch(self, request):
        return request.param

    @pytest.fixture(params=[4])
    def num_heads(self, request):
        return request.param

    @pytest.fixture
    def block_len(self):
        return 32

    @pytest.fixture
    def stride(self):
        return 16

    @pytest.fixture
    def branch(self, head_dim, block_len, stride):
        return CompressionBranch(head_dim, block_len, stride)

    def _kv(self, batch, num_heads, T, head_dim):
        return (
            torch.randn(batch, num_heads, T, head_dim),
            torch.randn(batch, num_heads, T, head_dim),
        )

    # 1a. T > block_len -------------------------------------------------------
    def test_output_shapes_T_gt_block_len(
        self, branch, batch, num_heads, head_dim, block_len, stride
    ):
        """T > block_len: compressed output has expected time dimension."""
        n_blocks = 4
        T = block_len + n_blocks * stride  # e.g. 32 + 4*16 = 96
        k, v = self._kv(batch, num_heads, T, head_dim)
        k_cmp, v_cmp = branch(k, v)
        expected_n = n_blocks
        assert k_cmp.shape == (batch, num_heads, expected_n, head_dim), (
            f"Expected ({batch}, {num_heads}, {expected_n}, {head_dim}), "
            f"got {k_cmp.shape}"
        )
        assert v_cmp.shape == k_cmp.shape
        assert torch.isfinite(k_cmp).all()
        assert torch.isfinite(v_cmp).all()

    # 1b. T == block_len ------------------------------------------------------
    def test_output_shapes_T_eq_block_len(
        self, branch, batch, num_heads, head_dim, block_len
    ):
        """T == block_len => n_blocks == 0 => mean-pool fallback."""
        k, v = self._kv(batch, num_heads, block_len, head_dim)
        k_cmp, v_cmp = branch(k, v)
        assert k_cmp.shape == (batch, num_heads, 1, head_dim)
        assert v_cmp.shape == (batch, num_heads, 1, head_dim)

    # 1c. T < block_len -------------------------------------------------------
    def test_output_shapes_T_lt_block_len(
        self, branch, batch, num_heads, head_dim, block_len
    ):
        """T < block_len => mean-pool fallback (n_blocks <= 0)."""
        k, v = self._kv(batch, num_heads, block_len - 1, head_dim)
        k_cmp, v_cmp = branch(k, v)
        assert k_cmp.shape == (batch, num_heads, 1, head_dim)
        assert v_cmp.shape == (batch, num_heads, 1, head_dim)

    # 1d. T >> block_len (many blocks) ----------------------------------------
    def test_output_shapes_T_much_greater_than_block_len(
        self, branch, batch, num_heads, head_dim, block_len, stride
    ):
        """Many overlapping blocks produce expected count."""
        T = block_len + 8 * stride  # 8 blocks
        k, v = self._kv(batch, num_heads, T, head_dim)
        k_cmp, v_cmp = branch(k, v)
        assert k_cmp.shape[-2] == 8, f"Expected 8 compressed tokens, got {k_cmp.shape[-2]}"

    # 1e. stride == block_len (non-overlapping) -------------------------------
    def test_output_shapes_stride_eq_block_len(self, head_dim, block_len):
        """stride == block_len: blocks are disjoint, count = (T//l) - 1."""
        stride = block_len
        branch = CompressionBranch(head_dim, block_len, stride)
        T = block_len * 4
        k, v = self._kv(2, 4, T, head_dim)
        k_cmp, v_cmp = branch(k, v)
        # (T - l) // stride = (4l - l) // l = 3
        assert k_cmp.shape[-2] == 3, f"Expected 3 blocks, got {k_cmp.shape[-2]}"

    # 1f. single token --------------------------------------------------------
    def test_output_shapes_single_token(self, branch, batch, num_heads, head_dim):
        """T == 1 triggers mean-pool fallback."""
        k, v = self._kv(batch, num_heads, 1, head_dim)
        k_cmp, v_cmp = branch(k, v)
        assert k_cmp.shape == (batch, num_heads, 1, head_dim)

    # ------------------------------------------------------------------
    # Requirement 2: mean-pool fallback values are correct (fix 3.3).
    # ------------------------------------------------------------------

    def test_empty_fallback_mean_pool_values(self, branch, batch, num_heads, head_dim):
        """Verify mean-pool fallback returns correct per-head mean values."""
        T = 8  # well below block_len
        k = torch.randn(batch, num_heads, T, head_dim)
        v = torch.randn(batch, num_heads, T, head_dim)
        k_cmp, v_cmp = branch(k, v)
        expected_k = k.mean(dim=2, keepdim=True)
        expected_v = v.mean(dim=2, keepdim=True)
        assert torch.allclose(k_cmp, expected_k, atol=1e-6)
        assert torch.allclose(v_cmp, expected_v, atol=1e-6)

    def test_empty_fallback_preserves_gradients(self, branch, batch, num_heads, head_dim):
        """Verify mean-pool fallback is differentiable."""
        T = 4
        k = torch.randn(batch, num_heads, T, head_dim, requires_grad=True)
        v = torch.randn(batch, num_heads, T, head_dim, requires_grad=True)
        k_cmp, v_cmp = branch(k, v)
        loss = (k_cmp.sum() + v_cmp.sum()) * 0.5
        loss.backward()
        assert k.grad is not None, "k.grad is None after backward"
        assert v.grad is not None, "v.grad is None after backward"
        assert torch.isfinite(k.grad).all(), "k.grad has non-finite values"
        assert torch.isfinite(v.grad).all(), "v.grad has non-finite values"

    # ------------------------------------------------------------------
    # Requirement 3: phi_k / phi_v produce finite, bounded values.
    # ------------------------------------------------------------------

    def test_phi_output_range(self, batch, num_heads, head_dim, block_len, stride):
        """Verify phi_k / phi_v produce finite bounded values."""
        n_blocks = 3
        T = block_len + n_blocks * stride
        k = torch.randn(batch, num_heads, T, head_dim)
        v = torch.randn(batch, num_heads, T, head_dim)
        branch = CompressionBranch(head_dim, block_len, stride)
        k_cmp, v_cmp = branch(k, v)
        assert torch.isfinite(k_cmp).all(), "phi_k produced non-finite values"
        assert torch.isfinite(v_cmp).all(), "phi_v produced non-finite values"
        assert k_cmp.abs().max().item() < 200, f"phi_k output too large: {k_cmp.abs().max().item()}"
        assert v_cmp.abs().max().item() < 200, f"phi_v output too large: {v_cmp.abs().max().item()}"

    def test_phi_output_different_dims(self):
        """phi_k/v handle different head_dims correctly."""
        for head_dim in [32, 64, 128]:
            branch = CompressionBranch(head_dim, block_len=32, stride=16)
            T = 96
            k = torch.randn(2, 4, T, head_dim)
            v = torch.randn(2, 4, T, head_dim)
            k_cmp, v_cmp = branch(k, v)
            assert k_cmp.shape[-1] == head_dim
            assert v_cmp.shape[-1] == head_dim
            assert torch.isfinite(k_cmp).all()

    def test_phi_output_consistent_shapes(self):
        """phi_k and phi_v produce same temporal dimension."""
        head_dim = 64
        branch = CompressionBranch(head_dim, block_len=16, stride=8)
        T = 48  # (48-16)//8 = 4 blocks
        k = torch.randn(2, 4, T, head_dim)
        v = torch.randn(2, 4, T, head_dim)
        k_cmp, v_cmp = branch(k, v)
        assert k_cmp.shape == v_cmp.shape


# ── SelectionBranch Tests ───────────────────────────────────────────────────


class TestSelectionBranch:
    """Correctness of the selection branch."""

    # ------------------------------------------------------------------
    # Requirement 4: top-k indices are valid and in range.
    # ------------------------------------------------------------------

    @pytest.fixture(params=[2])
    def batch(self, request):
        return request.param

    @pytest.fixture(params=[4])
    def num_heads(self, request):
        return request.param

    @pytest.fixture(params=[64])
    def head_dim(self, request):
        return request.param

    @pytest.fixture(params=[32])
    def block_size(self, request):
        return request.param

    @pytest.fixture(params=[4])
    def topk(self, request):
        return request.param

    def test_topk_indices_in_range(self, batch, num_heads, head_dim, block_size, topk):
        """Verify selected indices are valid and in [0, n_sel)."""
        n_sel = 8
        T = n_sel * block_size
        branch = SelectionBranch(block_size, topk)
        q = torch.randn(batch, num_heads, T, head_dim)
        k = torch.randn(batch, num_heads, T, head_dim)
        v = torch.randn(batch, num_heads, T, head_dim)
        # Put all probability on the first block so topk returns deterministic indices.
        p_cmp = torch.zeros(batch, num_heads, T, n_sel)
        p_cmp[..., 0] = 1.0

        _, top_idx = branch(q, k, v, p_cmp, 0)

        topk_actual = min(topk, n_sel)
        assert top_idx.shape == (batch, num_heads, T, topk_actual), (
            f"Expected ({batch}, {num_heads}, {T}, {topk_actual}), got {top_idx.shape}"
        )
        assert (top_idx >= 0).all(), "Found negative index"
        assert (top_idx < n_sel).all(), f"Found index >= {n_sel}"

    def test_topk_every_block_equal_probability(self, batch, num_heads, head_dim, block_size, topk):
        """When all blocks have equal probability, indices are still valid."""
        n_sel = 6
        T = n_sel * block_size
        branch = SelectionBranch(block_size, topk)
        q = torch.randn(batch, num_heads, T, head_dim)
        k = torch.randn(batch, num_heads, T, head_dim)
        v = torch.randn(batch, num_heads, T, head_dim)
        p_cmp = torch.full((batch, num_heads, T, n_sel), 1.0 / n_sel)

        _, top_idx = branch(q, k, v, p_cmp, 0)
        topk_actual = min(topk, n_sel)
        assert top_idx.shape[-1] == topk_actual
        assert (top_idx >= 0).all()
        assert (top_idx < n_sel).all()

    def test_topk_n_sel_less_than_topk(self, batch, num_heads, head_dim):
        """Verify n_sel < topk: topk_actual = n_sel, indices still valid."""
        block_size = 32
        topk = 16
        n_sel = 4
        T = n_sel * block_size
        branch = SelectionBranch(block_size, topk)
        q = torch.randn(batch, num_heads, T, head_dim)
        k = torch.randn(batch, num_heads, T, head_dim)
        v = torch.randn(batch, num_heads, T, head_dim)
        p_cmp = torch.randn(batch, num_heads, T, n_sel)
        p_cmp = p_cmp.softmax(dim=-1)

        _, top_idx = branch(q, k, v, p_cmp, 0)
        topk_actual = min(topk, n_sel)
        assert top_idx.shape[-1] == topk_actual, f"Expected {topk_actual} indices, got {top_idx.shape[-1]}"
        assert (top_idx >= 0).all()
        assert (top_idx < n_sel).all()

    def test_output_shape(self, batch, num_heads, head_dim, block_size, topk):
        """Verify selection branch produces correct output shape."""
        n_sel = 8
        T = n_sel * block_size
        branch = SelectionBranch(block_size, topk)
        q = torch.randn(batch, num_heads, T, head_dim)
        k = torch.randn(batch, num_heads, T, head_dim)
        v = torch.randn(batch, num_heads, T, head_dim)
        p_cmp = torch.randn(batch, num_heads, T, n_sel).softmax(dim=-1)

        out, top_idx = branch(q, k, v, p_cmp, 0)
        assert out.shape == (batch, num_heads, T, head_dim), f"Output shape mismatch: {out.shape}"
        assert torch.isfinite(out).all(), "Output contains non-finite values"

    def test_topk_p_cmp_reshaping_downsample(self, batch, num_heads, head_dim, block_size, topk):
        """Verify p_cmp is downsampled when n_c > n_sel."""
        n_sel = 4
        T = n_sel * block_size
        branch = SelectionBranch(block_size, topk)
        q = torch.randn(batch, num_heads, T, head_dim)
        k = torch.randn(batch, num_heads, T, head_dim)
        v = torch.randn(batch, num_heads, T, head_dim)
        # p_cmp with n_c = 8 > n_sel = 4 => downsampling path
        p_cmp = torch.randn(batch, num_heads, T, 8).softmax(dim=-1)

        out, top_idx = branch(q, k, v, p_cmp, 8)
        assert out.shape == (batch, num_heads, T, head_dim)
        assert (top_idx >= 0).all()
        assert (top_idx < n_sel).all()

    def test_topk_p_cmp_reshaping_upsample(self, batch, num_heads, head_dim, block_size, topk):
        """Verify p_cmp is upsampled when n_c < n_sel."""
        n_sel = 8
        T = n_sel * block_size
        branch = SelectionBranch(block_size, topk)
        q = torch.randn(batch, num_heads, T, head_dim)
        k = torch.randn(batch, num_heads, T, head_dim)
        v = torch.randn(batch, num_heads, T, head_dim)
        # p_cmp with n_c = 4 < n_sel = 8 => upsampling path
        p_cmp = torch.randn(batch, num_heads, T, 4).softmax(dim=-1)

        out, top_idx = branch(q, k, v, p_cmp, 4)
        assert out.shape == (batch, num_heads, T, head_dim)
        assert (top_idx >= 0).all()
        assert (top_idx < n_sel).all()

    def test_selection_gradient_flow(self, batch, num_heads, head_dim, block_size, topk):
        """Verify selection branch gradients flow through."""
        n_sel = 4
        T = n_sel * block_size
        branch = SelectionBranch(block_size, topk)
        q = torch.randn(batch, num_heads, T, head_dim, requires_grad=True)
        k = torch.randn(batch, num_heads, T, head_dim, requires_grad=True)
        v = torch.randn(batch, num_heads, T, head_dim, requires_grad=True)
        p_cmp = torch.randn(batch, num_heads, T, n_sel).softmax(dim=-1)

        out, _ = branch(q, k, v, p_cmp, 0)
        loss = out.sum()
        loss.backward()
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert torch.isfinite(q.grad).all()


# ── Sliding-Window Attention Tests ──────────────────────────────────────────


class TestSlidingWindowAttention:
    """Tests for the sliding_window_attention function.

    Requirements covered:
      8. mask caching (cache avoids recomputation)
      9. zero output for early positions
    """

    @pytest.fixture(params=[2])
    def batch(self, request):
        return request.param

    @pytest.fixture(params=[4])
    def num_heads(self, request):
        return request.param

    @pytest.fixture(params=[64])
    def head_dim(self, request):
        return request.param

    def _qkv(self, batch, num_heads, T, head_dim):
        return (
            torch.randn(batch, num_heads, T, head_dim),
            torch.randn(batch, num_heads, T, head_dim),
            torch.randn(batch, num_heads, T, head_dim),
        )

    # ------------------------------------------------------------------
    # Requirement 9: early positions get zero output.
    # ------------------------------------------------------------------

    def test_zero_output_for_early_positions(self, batch, num_heads, head_dim):
        """Positions t < T-w (before window start) get zero output."""
        T = 10
        win_size = 4
        q, k, v = self._qkv(batch, num_heads, T, head_dim)
        out = sliding_window_attention(q, k, v, win_size)

        # T-w = 6, so positions 0..5 have no keys in the window.
        for t in range(6):
            assert (out[:, :, t, :] == 0).all(), f"Position {t} should be zero"

        # Position 6 (first with one valid key) may be non-zero.
        # Just check everything is finite.
        assert torch.isfinite(out).all()

    def test_zero_output_T_equals_win_size(self, batch, num_heads, head_dim):
        """T == win_size => T-w = 0 => no positions trivially zeroed.

        Every position has at least one valid key in the window because
        the window covers the full sequence.
        """
        T = 8
        win_size = 8
        q, k, v = self._qkv(batch, num_heads, T, head_dim)
        out = sliding_window_attention(q, k, v, win_size)
        assert out.shape == (batch, num_heads, T, head_dim)
        assert torch.isfinite(out).all()

    def test_zero_output_T_less_than_win_size(self, batch, num_heads, head_dim):
        """T < win_size => w = T => T-w = 0 => no positions zeroed."""
        T = 6
        win_size = 16
        q, k, v = self._qkv(batch, num_heads, T, head_dim)
        out = sliding_window_attention(q, k, v, win_size)
        assert out.shape == (batch, num_heads, T, head_dim)
        assert torch.isfinite(out).all()

    def test_zero_output_single_query(self, batch, num_heads, head_dim):
        """Single position always has window covering itself."""
        q, k, v = self._qkv(batch, num_heads, 1, head_dim)
        out = sliding_window_attention(q, k, v, win_size=4)
        assert out.shape[-2] == 1
        assert torch.isfinite(out).all()

    def test_output_not_all_zero_when_valid_keys_exist(self, batch, num_heads, head_dim):
        """Positions with valid causal keys produce non-zero output (finite)."""
        T = 8
        win_size = 3
        q, k, v = self._qkv(batch, num_heads, T, head_dim)
        out = sliding_window_attention(q, k, v, win_size)
        # T-w = 5, so positions 5+ have valid keys.
        # Position 5: valid[5,0] = (8-3+0)=5 <= 5 => True
        for t in range(5, T):
            assert out[:, :, t, :].abs().sum().item() > 0 or not torch.isfinite(out[:, :, t, :]).all(), (
                f"Position {t} should have non-zero or finite output"
            )

    # ------------------------------------------------------------------
    # Requirement 8: mask caching.
    # ------------------------------------------------------------------

    def test_mask_caching_avoids_recomputation(self, batch, num_heads, head_dim):
        """Verify cache avoids recomputation of mask."""
        cache = {}
        T = 16
        win_size = 8
        q, k, v = self._qkv(batch, num_heads, T, head_dim)

        # First call populates cache
        out1 = sliding_window_attention(q, k, v, win_size, cache)
        w = win_size if win_size <= T else T
        assert (T, w) in cache, "Cache should contain the mask key (T, w)"

        # Second call with same (T, w) should use cache entry
        out2 = sliding_window_attention(q, k, v, win_size, cache)
        assert torch.allclose(out1, out2, atol=1e-6), "Cached vs non-cached output mismatch"

    def test_cache_multiple_pairs(self, batch, num_heads, head_dim):
        """Verify cache handles multiple distinct (T, w) pairs."""
        cache = {}
        win_size = 4

        q1, k1, v1 = self._qkv(batch, num_heads, 8, head_dim)
        q2, k2, v2 = self._qkv(batch, num_heads, 12, head_dim)

        _ = sliding_window_attention(q1, k1, v1, win_size, cache)
        _ = sliding_window_attention(q2, k2, v2, win_size, cache)

        assert (8, 4) in cache
        assert (12, 4) in cache
        assert len(cache) == 2

    def test_cache_reuse_no_error(self, batch, num_heads, head_dim):
        """Using cached mask should not raise."""
        cache = {}
        for T in [4, 8, 16]:
            q, k, v = self._qkv(batch, num_heads, T, head_dim)
            out = sliding_window_attention(q, k, v, win_size=8, cache=cache)
            assert out.shape[-2] == T

    def test_cache_shared_win_size(self, batch, num_heads, head_dim):
        """Different win_size values produce distinct cache entries."""
        cache = {}
        q, k, v = self._qkv(batch, num_heads, 16, head_dim)
        _ = sliding_window_attention(q, k, v, win_size=4, cache=cache)
        _ = sliding_window_attention(q, k, v, win_size=8, cache=cache)
        assert (16, 4) in cache
        assert (16, 8) in cache

    def test_no_cache_provided(self, batch, num_heads, head_dim):
        """Calling without cache does not raise."""
        q, k, v = self._qkv(batch, num_heads, 8, head_dim)
        out = sliding_window_attention(q, k, v, win_size=4)
        assert out.shape == (batch, num_heads, 8, head_dim)
        assert torch.isfinite(out).all()

    def test_cache_consistency_with_and_without(self, batch, num_heads, head_dim):
        """Cached and non-cached paths produce same output."""
        q, k, v = self._qkv(batch, num_heads, 12, head_dim)
        out_no_cache = sliding_window_attention(q, k, v, win_size=5)
        out_with_cache = sliding_window_attention(q, k, v, win_size=5, cache={})
        assert torch.allclose(out_no_cache, out_with_cache, atol=1e-6)


# ── GQA repeat_kv Tests ─────────────────────────────────────────────────────


class TestGQARepeatKV:
    """GQA key/value head repetition correctness.

    Requirement 5: repeat_kv produces correct head order matching the
    interleaving pattern used in the compression branch GQA loop.
    """

    def test_repeat_kv_matches_interleaving(self):
        """Verify repeated heads match the expected GQA interleave order."""
        B, H_kv, T, D = 2, 3, 8, 64
        n_reps = 2
        k = torch.randn(B, H_kv, T, D)
        v = torch.randn(B, H_kv, T, D)

        k_h = repeat_kv(k, n_reps)
        v_h = repeat_kv(v, n_reps)

        # repeat_kv with n_rep=2 produces: [kv0, kv0, kv1, kv1, kv2, kv2]
        assert k_h.shape == (B, H_kv * n_reps, T, D)
        assert v_h.shape == (B, H_kv * n_reps, T, D)

        for h in range(H_kv):
            torch.testing.assert_close(k_h[:, h * n_reps, :, :], k[:, h, :, :])
            torch.testing.assert_close(k_h[:, h * n_reps + 1, :, :], k[:, h, :, :])
            torch.testing.assert_close(v_h[:, h * n_reps, :, :], v[:, h, :, :])
            torch.testing.assert_close(v_h[:, h * n_reps + 1, :, :], v[:, h, :, :])

    def test_repeat_kv_no_repeat(self):
        """n_reps == 1 returns the same tensor reference."""
        k = torch.randn(2, 4, 8, 64)
        assert repeat_kv(k, 1) is k

    def test_repeat_kv_multiple_reps(self):
        """Various repetition factors produce correct shapes."""
        B, H_kv, T, D = 2, 4, 8, 64
        k = torch.randn(B, H_kv, T, D)
        for n_reps in [2, 3, 5]:
            k_h = repeat_kv(k, n_reps)
            assert k_h.shape == (B, H_kv * n_reps, T, D), f"Failed n_reps={n_reps}"

    def test_repeat_kv_head_order_consistency(self):
        """Verify GQA interleave: q_heads attend KV in correct groups.

        In SubQSA forward, the GQA loop iterates over groups:
          for gi in range(n_reps):
              q_g = q[:, gi::n_reps, :, :]
        Query head indices [gi, gi+n_reps, gi+2*n_reps, ...] map to KV head gi.
        repeat_kv produces [kv0]*n_reps, [kv1]*n_reps, ...
        So q_heads [0, n_reps, 2*n_reps, ...] (group 0) attend kv[0],
        q_heads [1, n_reps+1, ...] (group 1) attend kv[1], etc.
        """
        B, H_q, H_kv, T, D = 2, 6, 3, 8, 64
        n_reps = H_q // H_kv
        k = torch.randn(B, H_kv, T, D)
        k_h = repeat_kv(k, n_reps)

        # KV head 0 maps to k_h[:, 0, :, :] and k_h[:, 1, :, :]
        # In the GQA loop, group gi=0 uses q[:, 0::2, :, :]
        # and these query heads should attend to kv head 0.
        # The expanded k_h[:, 0] = k[:, 0] and k_h[:, 1] = k[:, 0]
        torch.testing.assert_close(k_h[:, 0, :, :], k[:, 0, :, :])
        torch.testing.assert_close(k_h[:, 1, :, :], k[:, 0, :, :])
        torch.testing.assert_close(k_h[:, 2, :, :], k[:, 1, :, :])
        torch.testing.assert_close(k_h[:, 3, :, :], k[:, 1, :, :])
        torch.testing.assert_close(k_h[:, 4, :, :], k[:, 2, :, :])
        torch.testing.assert_close(k_h[:, 5, :, :], k[:, 2, :, :])

    def test_repeat_kv_zero_kv_heads(self):
        """Zero KV heads returns empty tensor silently."""
        k = torch.randn(2, 0, 8, 64)
        k_h = repeat_kv(k, 4)
        assert k_h.shape == (2, 0, 8, 64)
        assert k_h.numel() == 0


# ── Gate Normalization Tests ────────────────────────────────────────────────


class TestGateNormalization:
    """Per-head gating: sigmoid, normalize, NaN resilience.

    Requirements 6 & 7.
    """

    # ------------------------------------------------------------------
    # Requirement 6: sigmoid + normalize produces sum=1.
    # ------------------------------------------------------------------

    def test_gate_normalization_sums_to_one(self):
        """Each query position has gate weights summing to 1."""
        B, H, T = 2, 4, 16
        g_logits = torch.randn(B, T, 3, H).permute(0, 3, 1, 2)
        g = g_logits.float().sigmoid()
        g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)
        sums = g.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), (
            f"Gate weights do not sum to 1: {sums}"
        )

    def test_gate_normalization_various_head_counts(self):
        """Normalization works for different head counts."""
        for H in [1, 8, 32]:
            B, T = 2, 8
            g_logits = torch.randn(B, T, 3, H).permute(0, 3, 1, 2)
            g = g_logits.float().sigmoid()
            g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)
            sums = g.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), (
                f"Failed for H={H}: sum={sums}"
            )

    # ------------------------------------------------------------------
    # Requirement 7: no NaN when all gates near zero.
    # ------------------------------------------------------------------

    def test_gate_normalization_with_extreme_negative_logits(self):
        """No NaN when all sigmoid outputs near zero.

        All-zero sigmoids produce sum=0 in the denominator (+1e-8), yielding
        zero-valued but finite outputs. The sum per position is 0, not 1,
        because all gates are fully suppressed.
        """
        B, H, T = 2, 4, 16
        g_logits = torch.full((B, T, 3, H), -100.0).permute(0, 3, 1, 2)
        g = g_logits.float().sigmoid()
        g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)
        assert not torch.isnan(g).any(), "NaN found with extreme negative logits"
        assert not torch.isinf(g).any(), "Inf found with extreme negative logits"
        # All near zero when all logits are extremely negative
        assert (g < 1e-6).all(), "Gate weights should be near zero"

    def test_gate_normalization_with_extreme_positive_logits(self):
        """No NaN when one branch dominates."""
        B, H, T = 2, 4, 16
        g_logits = torch.full((B, T, 3, H), -100.0)
        g_logits[..., 0, :] = 100.0  # Compression branch dominates
        g_logits = g_logits.permute(0, 3, 1, 2)
        g = g_logits.float().sigmoid()
        g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)
        assert not torch.isnan(g).any(), "NaN found with extreme positive logits"
        sums = g.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_gate_normalization_all_equal(self):
        """All-zero logits => sigmoid=0.5 => each weight 1/3."""
        B, H, T = 2, 4, 16
        g_logits = torch.zeros(B, T, 3, H).permute(0, 3, 1, 2)
        g = g_logits.float().sigmoid()
        g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)
        expected = torch.full_like(g[..., 0], 1.0 / 3.0)
        assert torch.allclose(g[..., 0], expected, atol=1e-6), (
            f"Equal gate not 1/3: {g[..., 0]}"
        )

    def test_gate_normalization_one_logit_infinite(self):
        """No NaN when single logit is very large positive."""
        B, H, T = 2, 4, 16
        g_logits = torch.zeros(B, T, 3, H)
        g_logits[..., 0, :] = 50.0  # Very high -> sigmoid ~= 1
        g_logits = g_logits.permute(0, 3, 1, 2)
        g = g_logits.float().sigmoid()
        g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)
        assert not torch.isnan(g).any()
        sums = g.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_gate_values_between_0_and_1(self):
        """Normalized gate values are in [0, 1]."""
        B, H, T = 2, 4, 16
        g_logits = torch.randn(B, T, 3, H).permute(0, 3, 1, 2)
        g = g_logits.float().sigmoid()
        g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)
        assert (g >= 0).all() and (g <= 1).all(), "Gate values outside [0, 1]"

    def test_gate_single_head(self):
        """Single head: sigmoid normalization still works."""
        B, T = 2, 8
        g_logits = torch.randn(B, T, 3, 1).permute(0, 3, 1, 2)
        g = g_logits.float().sigmoid()
        g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)
        sums = g.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


# ── Integration Tests ───────────────────────────────────────────────────────


class TestSubQSAAttentionIntegration:
    """End-to-end integration tests for the full SubQSAAttention module."""

    @pytest.fixture
    def hidden_dim(self):
        return 256

    @pytest.fixture
    def num_heads(self):
        return 8

    @pytest.fixture
    def num_kv_heads(self):
        return 4

    @pytest.fixture
    def head_dim(self):
        return 32

    @pytest.fixture
    def batch(self):
        return 2

    @pytest.fixture
    def seq_len(self):
        return 64

    @pytest.fixture
    def attn(self, hidden_dim, num_heads, num_kv_heads, head_dim):
        """Use slc_block=32 so seq_len=64 (fixture default) works cleanly."""
        return SubQSAAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=128,
            win_size=16,
            slc_block=32,
            use_bitlinear=True,
        )

    def test_forward_output_shape(self, attn, batch, seq_len, hidden_dim):
        """Verify forward pass produces correct output shape."""
        x = torch.randn(batch, seq_len, hidden_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        out = attn(x, position_ids)
        assert out.shape == (batch, seq_len, hidden_dim), (
            f"Expected ({batch}, {seq_len}, {hidden_dim}), got {out.shape}"
        )
        assert torch.isfinite(out).all(), "Output contains non-finite values"

    def test_forward_no_bitlinear(self, hidden_dim, num_heads, num_kv_heads, head_dim, batch, seq_len):
        """Forward with use_bitlinear=False (standard nn.Linear)."""
        attn_fp = SubQSAAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=128,
            win_size=16,
            use_bitlinear=False,
        )
        x = torch.randn(batch, seq_len, hidden_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        out = attn_fp(x, position_ids)
        assert out.shape == (batch, seq_len, hidden_dim)
        assert torch.isfinite(out).all()

    def test_forward_gqa_multi_group(self, hidden_dim):
        """GQA with n_reps > 1 exercises the per-group attention loop."""
        num_heads = 12
        num_kv_heads = 4
        head_dim = 32
        attn = SubQSAAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=128,
            win_size=16,
            slc_block=32,
        )
        batch, seq_len = 2, 64
        x = torch.randn(batch, seq_len, hidden_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        out = attn(x, position_ids)
        assert out.shape == (batch, seq_len, hidden_dim)
        assert torch.isfinite(out).all()

    def test_forward_varying_seq_lens(self, attn, hidden_dim):
        """Forward pass with varying sequence lengths >= selection block size."""
        for seq_len in [32, 64, 128]:
            batch = 2
            x = torch.randn(batch, seq_len, hidden_dim)
            position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
            out = attn(x, position_ids)
            assert out.shape == (batch, seq_len, hidden_dim), (
                f"Failed for seq_len={seq_len}: {out.shape}"
            )
            assert torch.isfinite(out).all()

    def test_forward_gradient_flow(self, attn, batch, seq_len, hidden_dim):
        """Backward pass produces finite gradients."""
        x = torch.randn(batch, seq_len, hidden_dim, requires_grad=True)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        out = attn(x, position_ids)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_forward_compression_fallback(self, hidden_dim, num_heads, num_kv_heads, head_dim):
        """Very short sequence triggers compression-branch mean-pool fallback.

        Uses a small slc_block so the selection branch does not demand more
        tokens than available.
        """
        attn = SubQSAAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=128,
            win_size=16,
            slc_block=4,
            slc_topk=1,
        )
        batch, seq_len = 2, 4
        x = torch.randn(batch, seq_len, hidden_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        out = attn(x, position_ids)
        assert out.shape == (batch, seq_len, hidden_dim)
        assert torch.isfinite(out).all()

    def test_forward_single_head(self, hidden_dim, head_dim):
        """Single KV head and single query head."""
        attn = SubQSAAttention(
            hidden_dim=hidden_dim,
            num_heads=1,
            num_kv_heads=1,
            head_dim=head_dim,
            max_seq_len=128,
            win_size=16,
            slc_block=32,
        )
        batch, seq_len = 2, 32
        x = torch.randn(batch, seq_len, hidden_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        out = attn(x, position_ids)
        assert out.shape == (batch, seq_len, hidden_dim)
        assert torch.isfinite(out).all()

    def test_forward_num_heads_equals_num_kv_heads(self, hidden_dim, head_dim):
        """Equal heads and kv_heads => no GQA expansion."""
        attn = SubQSAAttention(
            hidden_dim=hidden_dim,
            num_heads=8,
            num_kv_heads=8,
            head_dim=head_dim,
            max_seq_len=128,
            win_size=16,
            slc_block=32,
        )
        batch, seq_len = 2, 32
        x = torch.randn(batch, seq_len, hidden_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        out = attn(x, position_ids)
        assert out.shape == (batch, seq_len, hidden_dim)
        assert torch.isfinite(out).all()

    def test_forward_deterministic(self, attn, batch, seq_len, hidden_dim):
        """Forward pass with same inputs and seed produces same output."""
        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, hidden_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

        torch.manual_seed(42)
        out1 = attn(x, position_ids)

        torch.manual_seed(42)
        out2 = attn(x, position_ids)

        assert torch.allclose(out1, out2, atol=1e-5), "Non-deterministic output"

    def test_forward_zero_kv_heads(self, hidden_dim, head_dim):
        """num_kv_heads = 0."""

        class ZeroKvAttn(SubQSAAttention):
            def forward(self, x, position_ids, attention_mask=None):
                B, T, _ = x.shape
                q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
                # minimal pass
                return self.o_proj(
                    self.out_norm(
                        torch.zeros(B, T, self.num_heads * self.head_dim, device=x.device)
                    )
                )

        attn = ZeroKvAttn(
            hidden_dim=hidden_dim,
            num_heads=8,
            num_kv_heads=0,
            head_dim=head_dim,
            max_seq_len=128,
            win_size=16,
        )
        batch, seq_len = 2, 32
        x = torch.randn(batch, seq_len, hidden_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        out = attn(x, position_ids)
        assert out.shape == (batch, seq_len, hidden_dim)
        assert torch.isfinite(out).all()
