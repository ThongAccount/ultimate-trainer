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
)
from .pack_update import backward_dx, update, init_counter


# ═══════════════════════════════════════════════════════════════════════════════
#  Auto-dispatch: pick the best forward kernel for the given dimensions
# ═══════════════════════════════════════════════════════════════════════════════

def _forward_auto(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Auto-select forward kernel based on tensor dimensions."""
    B, _ = X.shape
    N = W.shape[0]
    # TC needs batch ≥ 16 and features ≥ 16 for good speedup
    if B >= 16 and N >= 16 and has_tc():
        return packed_ternary_forward_tc(W, X)
    # v2 needs N ≥ 4 for multi-output sharing
    if N >= 4 and has_forward_kernel_v2():
        return packed_ternary_forward_v2(W, X)
    # v1 fallback
    return packed_ternary_forward(W, X)


# ═══════════════════════════════════════════════════════════════════════════════
#  Initialization helpers
# ═══════════════════════════════════════════════════════════════════════════════

def xavier_init(out_features: int, in_features: int, gamma: float = 1.0) -> torch.Tensor:
    """Xavier-uniform initialised weights, packed to ternary."""
    std = math.sqrt(2.0 / (in_features + out_features))
    W_fp32 = torch.randn(out_features, in_features) * std
    return pack_tensor(W_fp32, gamma=gamma)


# ═══════════════════════════════════════════════════════════════════════════════
#  Autograd Function
# ═══════════════════════════════════════════════════════════════════════════════

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
        threshold: int = 64,
    ) -> torch.Tensor:
        # Ensure autograd graph hooks this Function even if X has no grad.
        # Without this, PyTorch prunes the graph and backward() is never
        # called, so the update() kernel never executes.
        if torch.is_grad_enabled() and not X.requires_grad:
            X = X.detach().requires_grad_(True)
        ctx.save_for_backward(X)
        ctx.W_packed = W_packed
        ctx.counter = counter
        ctx.in_features = in_features
        ctx.threshold = threshold
        return _forward_auto(W_packed, X)

    @staticmethod
    def backward(ctx, dY: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        (X,) = ctx.saved_tensors
        W_packed = ctx.W_packed
        counter = ctx.counter

        # Gradient w.r.t. input (needed upstream)
        dX = backward_dx(W_packed, dY, ctx.in_features)

        # Fused weight update: dW consumed, never stored
        # Only update if we're not frozen (counter exists)
        if counter is not None:
            update(W_packed, counter, X, dY, ctx.threshold)

        # Return gradients for each forward arg (None for non-tensor args)
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
        threshold: int = 64,
        bias: bool = True,
        init_scale: float = 1.0,
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
