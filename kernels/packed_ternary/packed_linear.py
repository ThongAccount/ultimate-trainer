"""PackedTernaryLinear — trainable nn.Module with fused autograd.

Integrates forward (auto-dispatch TC/scalar), backward (dX), and
counter-based weight update (dW→sign→counter→flip) into a single
torch.autograd.Function.

Usage:
    layer = PackedTernaryLinear(4096, 4096, threshold=64)
    x = torch.randn(2, 4096, dtype=torch.float16, device='cuda')
    y = layer(x)
    loss = y.mean()
    loss.backward()   # dX computed, W updated in-place via counter
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pack_tensor, compute_stride_words
from .pack_forward import (
    has_forward_kernel,
    packed_ternary_forward,
    has_forward_kernel_v2,
    packed_ternary_forward_v2,
    has_tc,
    packed_ternary_forward_tc,
    has_packed,
    packed_ternary_forward_packed,
)
from .pack_update import (
    backward_dx, update, backward_update, backward_update_fused, init_counter,
)

# Custom ops — makes kernels traceable by torch.compile
try:
    from .custom_ops import forward_tc as co_forward_tc
    from .custom_ops import backward_dx_tc as co_backward_dx_tc
    from .custom_ops import update_tc_v2 as co_update_tc_v2
    from .custom_ops import backward_update_fused as co_backward_update_fused
    _HAS_CUSTOM_OPS = True
except Exception:
    _HAS_CUSTOM_OPS = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Auto-dispatch: pick the best forward kernel for the given dimensions
# ═══════════════════════════════════════════════════════════════════════════════

def _forward_auto(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Auto-select forward kernel based on tensor dimensions."""
    B, K = X.shape       # M=B, K=in_features
    N = W.shape[0]       # N=out_features
    # TC WMMA m16n16k16 needs M, N, K all >= 16 AND multiples of 16
    from .pack_update import _tc_ok
    if _tc_ok(B) and _tc_ok(N) and _tc_ok(K) and has_tc():
        # Prefer custom op (traceable by torch.compile)
        if _HAS_CUSTOM_OPS:
            return co_forward_tc(W, X, K)
        return packed_ternary_forward_tc(W, X)
    # v2 needs N ≥ 4 for multi-output sharing
    if N >= 4 and has_forward_kernel_v2():
        return packed_ternary_forward_v2(W, X)
    # v1 fallback
    return packed_ternary_forward(W, X)


# ═══════════════════════════════════════════════════════════════════════════════
#  Initialization helpers
# ═══════════════════════════════════════════════════════════════════════════════

def xavier_init(out_features: int, in_features: int, gamma: float | None = None) -> torch.Tensor:
    """Xavier-uniform initialised weights, packed to ternary.

    gamma controls the quantization threshold.  With default gamma=None,
    gamma is set to std = sqrt(2/(in+out)) so that ~50% of weights flip
    to ±1 at init.  If gamma is too large (e.g., 1.0), almost all weights
    round to 0 and the network produces flat output.
    """
    std = math.sqrt(2.0 / (in_features + out_features))
    if gamma is None:
        gamma = std  # match quantization threshold to weight scale
    W_fp32 = torch.randn(out_features, in_features) * std
    return pack_tensor(W_fp32, gamma=gamma)


# ═══════════════════════════════════════════════════════════════════════════════
#  Autograd Function
# ═══════════════════════════════════════════════════════════════════════════════

# Pre-allocated dX buffer for backward (avoids allocation every step)
_dX_buf = None
_dX_shape = None
_dX_device = None


class PackedTernaryLinearFn(torch.autograd.Function):
    """Fused forward + backward + counter update for packed ternary weights.

    Forward:  Y = W @ X^T    (ternary × FP16)
    Backward: dX = W^T @ dY   (gradient w.r.t. input, for upstream)
    Update:   dW → sign → int16 counter → bit-flip   (no dW tensor stored)
    """

    @staticmethod
    def forward(
        ctx,
        X: torch.Tensor,
        W_packed: torch.Tensor,
        counter: torch.Tensor,
        in_features: int,
        threshold: int = 8,
    ) -> torch.Tensor:
        # Ensure autograd graph hooks this Function even if X has no grad.
        # Without this, PyTorch prunes the graph and backward() is never
        # called, so the update() kernel never executes.
        # clone() is required — detach() alone shares storage and causes
        # autograd tracking issues that triple Python overhead.
        if torch.is_grad_enabled() and not X.requires_grad:
            X = X.detach().clone().requires_grad_(True)
        # Store X as a plain ctx attribute, NOT via save_for_backward.
        # save_for_backward has PyTorch version-tracking that can corrupt
        # the saved tensor's data between forward and backward.
        ctx.X_saved = X
        ctx.W_packed = W_packed
        ctx.counter = counter
        ctx.in_features = in_features
        ctx.threshold = threshold
        return _forward_auto(W_packed, X)

    @staticmethod
    def backward(ctx, dY: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        global _dX_buf, _dX_shape, _dX_device
        X = ctx.X_saved
        W_packed = ctx.W_packed
        counter = ctx.counter
        threshold = ctx.threshold
        B = dY.size(0)

        # Fused backward + update: single kernel when dimensions allow
        if counter is not None:
            # Separate backward_dx + update is faster than fused kernel
            # due to atomicAdd contention on dX when many CTAs compete.
            # Use fused only when B, N, K are small.
            if _HAS_CUSTOM_OPS and B >= 16 and dY.size(1) >= 16 and ctx.in_features >= 16 and B <= 64 and dY.size(1) <= 128 and ctx.in_features <= 128:
                from .pack_update import _tc_ok
                dX = co_backward_update_fused(
                    W_packed, counter, dY, X, ctx.in_features, int(threshold))
            else:
                # Separate launches: faster for large dimensions
                if _HAS_CUSTOM_OPS and B >= 16 and dY.size(1) >= 16 and ctx.in_features >= 16:
                    dX = co_backward_dx_tc(W_packed, dY, ctx.in_features)
                    co_update_tc_v2(W_packed, counter, X.contiguous(), dY.contiguous(), int(threshold))
                else:
                    dX = backward_update(W_packed, counter, dY, X, ctx.in_features, threshold)
        else:
            dX = backward_dx(W_packed, dY, ctx.in_features)

        return dX, None, None, None, None


# ═══════════════════════════════════════════════════════════════════════════════
#  nn.Module
# ═══════════════════════════════════════════════════════════════════════════════

class PackedTernaryLinear(nn.Module):
    """Linear layer with packed ternary weights and counter-based optimizer.

    Characteristics:
        - Weights are always pack=True and ternary {-1,0,+1}
        - No FP32/BF16 master weights
        - Training uses sign→counter→flip (no AdamW)
        - Forward auto-dispatches TC (batch≥16) or scalar (small batch)
        - Backward computes dX for upstream, consumes dW in-place

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        threshold: Counter flip threshold (default 64).
        bias: Whether to include a FP16 bias term.
        init_scale: Weight initialisation scale (gamma parameter).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        threshold: int = 8,
        bias: bool = True,
        init_scale: float | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        stride = compute_stride_words(in_features)

        # Packed ternary weights (always ternary, no FP master)
        self.register_buffer(
            "W_packed",
            xavier_init(out_features, in_features, gamma=init_scale),
        )
        # int16 counter for the discrete optimizer
        self.register_buffer(
            "counter",
            torch.zeros(out_features, in_features, dtype=torch.int16),
        )

        # Optional FP16 bias (standard — not quantised)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply packed ternary linear transform.

        Args:
            X: Input tensor (batch, in_features) in FP16 or BF16.

        Returns:
            Y: Output tensor (batch, out_features) in FP16.
        """
        # Ensure FP16 (our kernels require it)
        if X.dtype != torch.float16:
            X = X.to(torch.float16)

        # Ensure autograd graph hooks PackedTernaryLinearFn even if root X has no grad
        if torch.is_grad_enabled() and not X.requires_grad:
            X = X.detach().requires_grad_(True)

        Y = PackedTernaryLinearFn.apply(
            X, self.W_packed, self.counter,
            self.in_features, self.threshold,
        )

        if self.bias is not None:
            Y = Y + self.bias.unsqueeze(0)

        return Y

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"threshold={self.threshold}, "
            f"bias={self.bias is not None}"
        )

    # ── Serialization ──────────────────────────────────────────────────────

    def state_dict(self, *args, **kwargs):
        """Override to ensure packed weights + counter are included."""
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """Handle dimension mismatches gracefully."""
        return super().load_state_dict(state_dict, strict=strict)

    # ── Reset ──────────────────────────────────────────────────────────────

    def reset_counter(self):
        """Zero out all counters (e.g. start of a new training phase)."""
        self.counter.zero_()

    def reset_weights(self, init_scale: float = 1.0):
        """Reinitialise weights with Xavier init."""
        self.W_packed = xavier_init(
            self.out_features, self.in_features, gamma=init_scale
        ).to(self.W_packed.device)


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience factory
# ═══════════════════════════════════════════════════════════════════════════════

def from_pretrained_linear(
    linear: nn.Linear,
    threshold: int = 64,
) -> PackedTernaryLinear:
    """Convert an FP16 nn.Linear to a PackedTernaryLinear with ternarised weights.

    The original FP32 weights are quantised to ternary {-1,0,+1} × γ
    where γ = mean(|W|).  Bias and shape are preserved.
    """
    W_fp32 = linear.weight.data.float()
    gamma = W_fp32.abs().mean().item()
    out_f, in_f = W_fp32.shape

    layer = PackedTernaryLinear(in_f, out_f, threshold=threshold,
                                bias=linear.bias is not None, init_scale=gamma)

    # The xavier_init already set random weights.  Overwrite with ternarised
    # version of the pretrained weights.
    layer.W_packed = pack_tensor(W_fp32, gamma=gamma).to(layer.W_packed.device)

    if linear.bias is not None:
        layer.bias.data = linear.bias.data.to(torch.float16)

    return layer
