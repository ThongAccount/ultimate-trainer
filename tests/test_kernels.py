"""Comprehensive tests for Triton kernel edge cases (ternary_matmul).

Tests exercise the CPU fallback path (no GPU required).  Every function
in ``kernels.ternary_matmul`` is covered: ``compute_gamma``,
``ternary_matmul``, and ``fused_bitlinear_forward``.

Required tests:
  test_compute_gamma_matches_reference
  test_ternary_matmul_cpu_fallback
  test_fused_bitlinear_forward_2d_input
  test_fused_bitlinear_forward_3d_input
  test_fused_bitlinear_forward_optional_bias
  test_ternary_matmul_non_power_of_two
  test_ternary_matmul_minimum_dimensions
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup -- make project root importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from kernels.ternary_matmul import (
    compute_gamma,
    fused_bitlinear_forward,
    ternary_matmul,
)


# ===================================================================
#  Fixtures
# ===================================================================

@pytest.fixture
def small_weight():
    """Small weight matrix for quick tests: (N=4, K=8)."""
    return torch.tensor(
        [
            [0.8, -0.3, 0.0, 0.6, -0.9, 0.1, -0.2, 0.4],
            [-0.5, 0.7, 0.2, -0.1, 0.3, -0.8, 0.0, 0.9],
            [0.1, 0.0, -0.6, 0.5, 0.2, -0.4, 0.8, -0.3],
            [-0.7, 0.3, 0.4, -0.2, 0.0, 0.6, -0.1, 0.5],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def non_power_of_two_weight():
    """Weight for (N=5, K=7) — neither dimension a power of two."""
    return torch.randn(5, 7)  # will be seeded in each test


@pytest.fixture
def min_weight():
    """Weight for (N=1, K=1) — minimum dimensions."""
    return torch.tensor([[1.5]], dtype=torch.float32)


# ===================================================================
#  1.  compute_gamma matches reference formula
# ===================================================================

class TestComputeGamma:
    """Verify compute_gamma(W) == W.abs().mean() + eps on CPU."""

    @torch.no_grad()
    def test_compute_gamma_matches_reference(self, small_weight):
        """compute_gamma must return mean(|W|) + 1e-5 (the default eps)."""
        w = small_weight
        gamma_computed = compute_gamma(w)
        gamma_expected = w.abs().mean() + 1e-5
        assert isinstance(gamma_computed, torch.Tensor), (
            "compute_gamma must return a tensor"
        )
        assert gamma_computed.ndim == 0, (
            f"Expected scalar tensor, got shape {gamma_computed.shape}"
        )
        assert torch.allclose(gamma_computed, gamma_expected, atol=1e-7), (
            f"compute_gamma={gamma_computed.item():.8f}, "
            f"expected={gamma_expected.item():.8f}"
        )

    @torch.no_grad()
    def test_compute_gamma_preserves_device(self, small_weight):
        """Result should reside on the same device as the input."""
        gamma = compute_gamma(small_weight)
        assert gamma.device == small_weight.device, (
            f"gamma on {gamma.device}, expected {small_weight.device}"
        )

    @torch.no_grad()
    def test_compute_gamma_custom_eps(self):
        """Passing a non-default epsilon should be honoured."""
        w = torch.tensor([[2.0, -3.0], [1.0, -4.0]], dtype=torch.float32)
        eps = 0.01
        # mean(|W|) = (2+3+1+4)/4 = 2.5
        gamma = compute_gamma(w, eps=eps)
        expected = 2.5 + eps
        assert torch.allclose(gamma, torch.tensor(expected), atol=1e-7), (
            f"gamma={gamma.item()} with eps={eps}, expected {expected}"
        )

    @torch.no_grad()
    def test_compute_gamma_all_zeros(self):
        """When W is all zeros, gamma should equal eps exactly."""
        w = torch.zeros(4, 8)
        gamma = compute_gamma(w, eps=1e-5)
        assert torch.allclose(gamma, torch.tensor(1e-5), atol=1e-8), (
            f"gamma for zero weight = {gamma.item()}, expected 1e-5"
        )

    @torch.no_grad()
    def test_compute_gamma_with_negative_values(self):
        """Large negative values should contribute via abs()."""
        w = torch.tensor([[-100.0, 200.0]], dtype=torch.float32)
        eps = 0.0
        # mean(|W|) = (100 + 200) / 2 = 150
        gamma = compute_gamma(w, eps=eps)
        assert torch.allclose(gamma, torch.tensor(150.0), atol=1e-5), (
            f"gamma={gamma.item()}, expected 150"
        )


# ===================================================================
#  2.  ternary_matmul CPU fallback shape correctness
# ===================================================================

class TestTernaryMatmulCPUFallback:
    """CPU fallback must produce the correct output shape."""

    @torch.no_grad()
    def test_ternary_matmul_cpu_fallback(self, small_weight):
        """(M=3, K=8) x (N=4, K=8) → output (M=3, N=4)."""
        w = small_weight
        M, N, K = 3, 4, 8
        x = torch.randn(M, K)
        gamma = w.abs().mean() + 1e-5

        y = ternary_matmul(x, w, gamma)
        assert y.shape == (M, N), (
            f"Expected ({M}, {N}), got {y.shape}"
        )
        assert y.dtype == torch.float32, (
            f"Expected float32 output, got {y.dtype}"
        )

    @torch.no_grad()
    def test_cpu_fallback_numerical_correctness(self, small_weight):
        """CPU fallback result must match eager quant+linear."""
        w = small_weight
        M, N, K = 3, 4, 8
        torch.manual_seed(42)
        x = torch.randn(M, K)
        gamma = w.abs().mean() + 1e-5

        y = ternary_matmul(x, w, gamma)

        # Eager reference
        w_q = torch.clamp(torch.round(w / gamma), -1.0, 1.0)
        y_expected = F.linear(x, w_q)

        assert torch.allclose(y, y_expected, atol=1e-6), (
            "CPU fallback result differs from eager quant+linear"
        )

    @torch.no_grad()
    def test_cpu_fallback_gamma_as_tensor(self, small_weight):
        """Passing gamma as a 0-dim tensor (not float) should still work."""
        w = small_weight
        x = torch.randn(2, 8)
        gamma_tensor = torch.tensor(w.abs().mean().item() + 1e-5)
        y = ternary_matmul(x, w, gamma_tensor)
        assert y.shape == (2, 4), (
            f"Expected (2, 4), got {y.shape} — gamma-as-tensor failed"
        )

    @torch.no_grad()
    def test_cpu_fallback_gamma_as_float(self, small_weight):
        """Passing gamma as a Python float should also work (coercion)."""
        w = small_weight
        x = torch.randn(2, 8)
        gamma_float = float(w.abs().mean().item() + 1e-5)
        y = ternary_matmul(x, w, gamma_float)
        assert y.shape == (2, 4), (
            f"Expected (2, 4), got {y.shape} — gamma-as-float failed"
        )


# ===================================================================
#  6.  Non-power-of-two dimensions
# ===================================================================

class TestNonPowerOfTwo:
    """Tensor shapes where M, N, K are not divisible by Triton block sizes."""

    @torch.no_grad()
    def test_ternary_matmul_non_power_of_two(self):
        """M=3, N=5, K=7 must produce correct shape and values."""
        torch.manual_seed(1729)
        M, N, K = 3, 5, 7
        x = torch.randn(M, K)
        w = torch.randn(N, K)
        gamma = w.abs().mean() + 1e-5

        y = ternary_matmul(x, w, gamma)
        assert y.shape == (M, N), (
            f"Non-power-of-two: expected ({M}, {N}), got {y.shape}"
        )

        # Reference
        w_q = torch.clamp(torch.round(w / gamma), -1.0, 1.0)
        y_expected = F.linear(x, w_q)
        assert torch.allclose(y, y_expected, atol=1e-6), (
            "Non-power-of-two result differs from reference"
        )

    @torch.no_grad()
    def test_non_power_of_two_larger(self):
        """Larger non-power-of-two: M=65, N=33, K=17."""
        torch.manual_seed(42)
        M, N, K = 65, 33, 17
        x = torch.randn(M, K)
        w = torch.randn(N, K)
        gamma = w.abs().mean() + 1e-5

        y = ternary_matmul(x, w, gamma)
        assert y.shape == (M, N), (
            f"Larger non-power-of-two: expected ({M}, {N}), got {y.shape}"
        )

        w_q = torch.clamp(torch.round(w / gamma), -1.0, 1.0)
        y_expected = F.linear(x, w_q)
        assert torch.allclose(y, y_expected, atol=1e-6), (
            "Larger non-power-of-two result differs from reference"
        )

    @torch.no_grad()
    def test_non_power_of_two_prime_dimensions(self):
        """Prime-number dimensions: M=7, N=11, K=13."""
        torch.manual_seed(7)
        M, N, K = 7, 11, 13
        x = torch.randn(M, K)
        w = torch.randn(N, K)
        gamma = w.abs().mean() + 1e-5

        y = ternary_matmul(x, w, gamma)
        assert y.shape == (M, N), (
            f"Prime dims: expected ({M}, {N}), got {y.shape}"
        )

        w_q = torch.clamp(torch.round(w / gamma), -1.0, 1.0)
        y_expected = F.linear(x, w_q)
        assert torch.allclose(y, y_expected, atol=1e-6), (
            "Prime-dimension result differs from reference"
        )

    @torch.no_grad()
    def test_non_power_of_two_tall_matrix(self):
        """Tall-and-skinny: M=127, N=3, K=257."""
        torch.manual_seed(99)
        M, N, K = 127, 3, 257
        x = torch.randn(M, K)
        w = torch.randn(N, K)
        gamma = w.abs().mean() + 1e-5

        y = ternary_matmul(x, w, gamma)
        assert y.shape == (M, N), (
            f"Tall matrix: expected ({M}, {N}), got {y.shape}"
        )

        w_q = torch.clamp(torch.round(w / gamma), -1.0, 1.0)
        y_expected = F.linear(x, w_q)
        assert torch.allclose(y, y_expected, atol=1e-5), (
            "Tall-matrix result differs from reference"
        )


# ===================================================================
#  7.  Minimum dimensions
# ===================================================================

class TestMinimumDimensions:
    """M=1, N=1, K=1 — smallest possible tensor."""

    @torch.no_grad()
    def test_ternary_matmul_minimum_dimensions(self, min_weight):
        """With M=N=K=1 the output must be a 1×1 tensor matching reference."""
        w = min_weight  # (1, 1)
        x = torch.tensor([[0.5]])
        gamma = w.abs().mean() + 1e-5

        y = ternary_matmul(x, w, gamma)
        assert y.shape == (1, 1), (
            f"Minimum dims: expected (1, 1), got {y.shape}"
        )

        # Reference
        w_q = torch.clamp(torch.round(w / gamma), -1.0, 1.0)
        y_expected = F.linear(x, w_q)
        assert torch.allclose(y, y_expected, atol=1e-7), (
            f"Minimum-dim result {y.item()} differs from reference {y_expected.item()}"
        )

    @torch.no_grad()
    def test_minimum_positive_input(self, min_weight):
        """Positive x×weight quantised to +1."""
        w = min_weight  # [[1.5]]
        x = torch.tensor([[2.0]])
        gamma = float(w.abs().mean() + 1e-5)  # ≈ 1.50001

        y = ternary_matmul(x, w, gamma)
        # round(1.5 / 1.50001) ≈ round(1.0) = 1.0  => w_q = [[1.0]]
        # y = 2.0 * 1.0 = 2.0
        expected = 2.0
        assert torch.allclose(y, torch.tensor([[expected]]), atol=1e-5), (
            f"Minimum positive: expected {expected}, got {y.item()}"
        )

    @torch.no_grad()
    def test_minimum_negative_input(self):
        """Negative activation with weight quantised to -1."""
        w = torch.tensor([[-2.0]])
        x = torch.tensor([[3.0]])
        gamma = w.abs().mean() + 1e-5  # ≈ 2.00001

        y = ternary_matmul(x, w, gamma)
        # round(-2.0 / 2.00001) ≈ round(-1.0) = -1.0  => w_q = [[-1.0]]
        # y = 3.0 * (-1.0) = -3.0
        expected = -3.0
        assert torch.allclose(y, torch.tensor([[expected]]), atol=1e-5), (
            f"Minimum negative: expected {expected}, got {y.item()}"
        )

    @torch.no_grad()
    def test_minimum_weight_zeros(self):
        """When weight is zero, the quantised weight is 0 → output is zero."""
        w = torch.zeros(1, 1)
        x = torch.tensor([[42.0]])
        gamma = 1e-5  # mean(|0|) + 1e-5

        y = ternary_matmul(x, w, gamma)
        # round(0.0 / 1e-5) = round(0.0) = 0.0  => w_q = [[0.0]]
        # y = 42.0 * 0.0 = 0.0
        assert torch.allclose(y, torch.tensor([[0.0]]), atol=1e-7), (
            f"Zero weight: expected 0, got {y.item()}"
        )


# ===================================================================
#  3 & 4 & 5.  fused_bitlinear_forward reshape logic
# ===================================================================

class TestFusedBitlinearForward:
    """fused_bitlinear_forward must handle 2D, 3D, and optional bias."""

    @torch.no_grad()
    def test_fused_bitlinear_forward_2d_input(self):
        """(M=8, K=16) input → output (M=8, N=4)."""
        torch.manual_seed(1)
        M, K, N = 8, 16, 4
        x = torch.randn(M, K)
        weight = torch.nn.Parameter(torch.randn(N, K))
        gamma = weight.abs().mean() + 1e-5
        bias = torch.randn(N)

        y = fused_bitlinear_forward(x, weight, gamma, bias)
        assert y.shape == (M, N), (
            f"2D forward: expected ({M}, {N}), got {y.shape}"
        )

        # Reference: flatten (no-op), matmul, add bias
        w_q = torch.clamp(torch.round(weight / gamma), -1.0, 1.0)
        y_expected = F.linear(x, w_q) + bias
        assert torch.allclose(y, y_expected, atol=1e-6), (
            "2D forward differs from reference"
        )

    @torch.no_grad()
    def test_fused_bitlinear_forward_3d_input(self):
        """(B=2, T=4, K=16) input → output (B=2, T=4, N=4).

        The kernel internally flattens to (B*T, K) then restores.
        """
        torch.manual_seed(2)
        B, T, K, N = 2, 4, 16, 4
        x = torch.randn(B, T, K)
        weight = torch.nn.Parameter(torch.randn(N, K))
        gamma = weight.abs().mean() + 1e-5
        bias = torch.randn(N)

        y = fused_bitlinear_forward(x, weight, gamma, bias)
        assert y.shape == (B, T, N), (
            f"3D forward: expected ({B}, {T}, {N}), got {y.shape}"
        )

        # Reference
        w_q = torch.clamp(torch.round(weight / gamma), -1.0, 1.0)
        x_2d = x.reshape(-1, K)
        y_2d = F.linear(x_2d, w_q) + bias
        y_expected = y_2d.reshape(B, T, N)
        assert torch.allclose(y, y_expected, atol=1e-6), (
            "3D forward differs from reference"
        )

    @torch.no_grad()
    def test_fused_bitlinear_forward_4d_input(self):
        """(B=2, H=3, T=5, K=16) input → output (B=2, H=3, T=5, N=4).

        Verifies that the generic *dims → reshape pattern works beyond 3D.
        """
        torch.manual_seed(3)
        B, H, T, K, N = 2, 3, 5, 16, 4
        x = torch.randn(B, H, T, K)
        weight = torch.nn.Parameter(torch.randn(N, K))
        gamma = weight.abs().mean() + 1e-5
        bias = torch.randn(N)

        y = fused_bitlinear_forward(x, weight, gamma, bias)
        assert y.shape == (B, H, T, N), (
            f"4D forward: expected ({B}, {H}, {T}, {N}), got {y.shape}"
        )

        w_q = torch.clamp(torch.round(weight / gamma), -1.0, 1.0)
        x_2d = x.reshape(-1, K)
        y_2d = F.linear(x_2d, w_q) + bias
        y_expected = y_2d.reshape(B, H, T, N)
        assert torch.allclose(y, y_expected, atol=1e-6), (
            "4D forward differs from reference"
        )

    @torch.no_grad()
    def test_fused_bitlinear_forward_optional_bias(self):
        """Passing bias=None (default) must not raise and produce correct shape."""
        torch.manual_seed(4)
        M, K, N = 6, 12, 5
        x = torch.randn(M, K)
        weight = torch.nn.Parameter(torch.randn(N, K))
        gamma = weight.abs().mean() + 1e-5

        y = fused_bitlinear_forward(x, weight, gamma)  # bias=None
        assert y.shape == (M, N), (
            f"No bias: expected ({M}, {N}), got {y.shape}"
        )

        # Reference without bias
        w_q = torch.clamp(torch.round(weight / gamma), -1.0, 1.0)
        y_expected = F.linear(x, w_q)
        assert torch.allclose(y, y_expected, atol=1e-6), (
            "No-bias forward differs from reference"
        )

    @torch.no_grad()
    def test_fused_bitlinear_forward_bias_explicit_none(self):
        """Explicit bias=None should behave identically to omitting the argument."""
        torch.manual_seed(5)
        M, K, N = 5, 10, 3
        x = torch.randn(M, K)
        weight = torch.nn.Parameter(torch.randn(N, K))
        gamma = weight.abs().mean() + 1e-5

        y1 = fused_bitlinear_forward(x, weight, gamma)
        y2 = fused_bitlinear_forward(x, weight, gamma, bias=None)
        assert torch.allclose(y1, y2, atol=1e-7), (
            "bias=None and default bias must produce identical results"
        )

    @torch.no_grad()
    def test_fused_bitlinear_forward_3d_no_bias(self):
        """3D input with no bias must preserve batch and sequence dimensions."""
        torch.manual_seed(6)
        B, T, K, N = 2, 8, 16, 4
        x = torch.randn(B, T, K)
        weight = torch.nn.Parameter(torch.randn(N, K))
        gamma = weight.abs().mean() + 1e-5

        y = fused_bitlinear_forward(x, weight, gamma)  # bias=None
        assert y.shape == (B, T, N), (
            f"3D no bias: expected ({B}, {T}, {N}), got {y.shape}"
        )

        w_q = torch.clamp(torch.round(weight / gamma), -1.0, 1.0)
        x_2d = x.reshape(-1, K)
        y_2d = F.linear(x_2d, w_q)
        y_expected = y_2d.reshape(B, T, N)
        assert torch.allclose(y, y_expected, atol=1e-6), (
            "3D no-bias forward differs from reference"
        )


# ===================================================================
#  Additional edge cases
# ===================================================================

class TestEdgeCases:
    """Edge cases for ternary_matmul on CPU fallback."""

    @torch.no_grad()
    def test_zero_activation(self):
        """All-zero activations should produce zero output regardless of W."""
        M, N, K = 4, 6, 10
        w = torch.randn(N, K)
        x = torch.zeros(M, K)
        gamma = w.abs().mean() + 1e-5

        y = ternary_matmul(x, w, gamma)
        assert y.shape == (M, N)
        assert y.abs().max().item() == 0.0, (
            "Zero activation must give zero output"
        )

    @torch.no_grad()
    def test_very_large_input_values(self):
        """Large input values should not overflow in the CPU fallback path."""
        M, N, K = 2, 3, 5
        w = torch.randn(N, K) * 100  # large weights
        x = torch.randn(M, K) * 100  # large activations
        gamma = w.abs().mean() + 1e-5

        y = ternary_matmul(x, w, gamma)
        assert y.shape == (M, N)
        assert not torch.any(torch.isnan(y)), (
            "Large inputs produced NaN"
        )
        assert not torch.any(torch.isinf(y)), (
            "Large inputs produced Inf"
        )

    @torch.no_grad()
    def test_single_batch_large_features(self):
        """K=1024, M=1 (single example) to stress the matmul dimension."""
        M, N, K = 1, 8, 1024
        w = torch.randn(N, K)
        x = torch.randn(M, K)
        gamma = w.abs().mean() + 1e-5

        y = ternary_matmul(x, w, gamma)
        assert y.shape == (M, N), (
            f"Single-batch K=1024: expected ({M}, {N}), got {y.shape}"
        )

        w_q = torch.clamp(torch.round(w / gamma), -1.0, 1.0)
        y_expected = F.linear(x, w_q)
        assert torch.allclose(y, y_expected, atol=1e-5), (
            "Single-batch K=1024 result differs from reference"
        )

    @torch.no_grad()
    def test_large_output_dim(self):
        """N=1024 (many output features)."""
        M, N, K = 2, 1024, 16
        torch.manual_seed(42)
        w = torch.randn(N, K)
        x = torch.randn(M, K)
        gamma = w.abs().mean() + 1e-5

        y = ternary_matmul(x, w, gamma)
        assert y.shape == (M, N), (
            f"Large N: expected ({M}, {N}), got {y.shape}"
        )

        w_q = torch.clamp(torch.round(w / gamma), -1.0, 1.0)
        y_expected = F.linear(x, w_q)
        assert torch.allclose(y, y_expected, atol=1e-5), (
            "Large-N result differs from reference"
        )

    @torch.no_grad()
    def test_weight_all_below_threshold(self):
        """Weight values below the 0.5 threshold should be quantised to zero.

        If W/gamma is in [-0.5, 0.5] for every element, then every w_q = 0
        and the output must be zero regardless of x.
        """
        M, N, K = 2, 3, 5
        gamma = 1000.0  # very large gamma → all elements below threshold
        w = torch.randn(N, K)  # |w| unlikely to exceed 500
        x = torch.randn(M, K)

        y = ternary_matmul(x, w, gamma)
        assert y.shape == (M, N)
        assert y.abs().max().item() == 0.0, (
            "When all weights are below the ternary threshold, "
            "output must be zero"
        )

    @torch.no_grad()
    def test_weight_all_above_positive_threshold(self):
        """All weight values well above +0.5 threshold → every w_q = +1."""
        M, N, K = 2, 3, 5
        w = torch.full((N, K), 100.0)  # large positive
        x = torch.randn(M, K)
        gamma = 1.0

        y = ternary_matmul(x, w, gamma)
        assert y.shape == (M, N)

        # Since every element is round(100) = 1, w_q = all-ones
        # y = x @ w_q^T = x @ 1^T = sum(x, dim=1, keepdim=True) broadcast
        w_q = torch.clamp(torch.round(w / gamma), -1.0, 1.0)
        y_expected = F.linear(x, w_q)
        assert torch.allclose(y, y_expected, atol=1e-6), (
            "All-positive-ternary result differs"
        )

    @torch.no_grad()
    def test_gamma_is_scalar_tensor(self):
        """Ensure the gamma parameter type is handled correctly (tensor or numeric)."""
        M, N, K = 2, 3, 5
        w = torch.randn(N, K)
        x = torch.randn(M, K)

        gamma_tensor = torch.tensor(0.5)
        y_tensor = ternary_matmul(x, w, gamma_tensor)

        gamma_float = 0.5
        y_float = ternary_matmul(x, w, gamma_float)

        assert torch.allclose(y_tensor, y_float, atol=1e-6), (
            "Gamma as tensor vs float must give identical result"
        )

        w_q = torch.clamp(torch.round(w / 0.5), -1.0, 1.0)
        y_expected = F.linear(x, w_q)
        assert torch.allclose(y_tensor, y_expected, atol=1e-6), (
            "Gamma=0.5 result differs from reference"
        )
