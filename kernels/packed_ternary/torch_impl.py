"""Pure-PyTorch ternary linear: forward / backward-dx / counter update.

Reference implementation in plain PyTorch tensor ops (no CUDA C++, no Triton).
Matches the CUDA kernels' packed 2-bit ternary layout:
  - W is packed int32 [N, stride_words], 16 ternary codes per word, little-endian
    bit order: word = Σ code_i << (2i),  code 0→0, 1→+1, 2→−1.
  - counter is int16 [N, K].
  - update (CUDA-matching, gradient DESCENT):
      dW[r,c] = Σ_b dY[b,r]·X[b,c]   (fp32, never materialised)
      counter[r,c] -= sign(dW)            # positive dW pushes weight DOWN
      counter >  +threshold → ternary += 1  (−1→0→+1), counter reset
      counter <  −threshold → ternary −= 1  (+1→0→−1), counter reset
      flip re-packs the 2-bit code back into the word in place.

Sign convention note (semantic gap inferred):
  The task prompt described the update as ``counter += sign(dW)`` with
  ``> threshold → ternary += 1`` — that is gradient ASCENT.  The CUDA kernels
  (gemm_update.cu, gemm_fused_backward_update.cu) and the Python wrappers
  (pack_update.py) both agree on ``counter -= sign(dW)`` — gradient DESCENT —
  which is also the direction required for training to converge (the ASCENT
  form makes loss increase).  Per the task rule "where CUDA and wrappers
  disagree, wrapper + prompt win", there is no CUDA↔wrapper disagreement here,
  so the CUDA/wrapper descent semantics are the contract.  The ascent form is
  kept available behind ``descent=False`` for completeness.

Correctness contract: outputs match the CUDA stack (ref_linear ground truth).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import compute_stride_words, pack_tensor, unpack_tensor  # noqa: F401

_LUT = torch.tensor([0, 1, -1, 0], dtype=torch.int8)


def unpack_ternary(packed: torch.Tensor, rows: int, cols: int,
                   device: torch.device | None = None) -> torch.Tensor:
    """Decode packed int32 [rows, stride_words] → int8 [rows, cols] in {-1,0,1}.

    Vectorized: no Python loop over rows/cols (16 fixed-width bit lanes).
    Decodes into a padded [rows, stride*16] buffer so every lane's slice is
    exactly stride wide (ragged K where cols % 16 != 0 is sliced off at the
    end), then returns the [rows, cols] view.
    """
    stride = packed.shape[1]
    w = packed.to(torch.int64) & 0xFFFFFFFF  # treat as unsigned, keep 64-bit
    out = torch.zeros(rows, stride * 16, dtype=torch.int8, device=packed.device)
    for i in range(16):
        code = ((w >> (2 * i)) & 0b11).to(torch.long)  # index dtype for LUT
        out[:, i::16] = _LUT.to(device=packed.device)[code]
    return out[:, :cols].contiguous()


def ternary_forward(packed: torch.Tensor, X: torch.Tensor, K: int,
                    gamma: float = 1.0) -> torch.Tensor:
    """Y = X @ W^T, W decoded from packed ternary. Returns FP16 like CUDA."""
    rows, cols = packed.shape[0], K
    tern = unpack_ternary(packed, rows, cols).float() * gamma
    return F.linear(X.float(), tern).half()


def ternary_backward_dx(packed: torch.Tensor, dY: torch.Tensor, K: int,
                        gamma: float = 1.0) -> torch.Tensor:
    """dX = dY @ W (upstream gradient w.r.t. X)."""
    rows, cols = packed.shape[0], K
    tern = unpack_ternary(packed, rows, cols).float() * gamma
    return (dY.float() @ tern).half()


def ternary_update(packed: torch.Tensor, counter: torch.Tensor,
                   X: torch.Tensor, dY: torch.Tensor, threshold: int,
                   K: int, gamma: float = 1.0,
                   descent: bool = True) -> torch.Tensor:
    """In-place counter → flip update. Returns updated packed (same object).

    dW[r,c] = Σ_b dY[b,r]·X[b,c] computed in FP32 (never materialized by the
    CUDA kernel, but the math is identical); then per weight:
      counter[r,c] -= sign(dW[r,c])                       (descent, = CUDA)
      counter >  +threshold → ternary += 1  (−1→0→+1), counter := 0
      counter <  −threshold → ternary −= 1  (+1→0→−1), counter := 0

    With ``descent=False`` the sign is flipped (counter += sign(dW)) — the
    ascent form from the original task prompt, kept for reference only.

    Re-packing the whole packed word each step is intentional (correctness
    target, not performance).
    """
    rows, cols = packed.shape[0], K
    dW = (X.float().t() @ dY.float()).t()          # [N, K]; dW[r,c] = Σ_b dY[b,r]·X[b,c]
    sign = torch.sign(dW).to(counter.dtype)        # ±1 (0 where dW == 0)
    if descent:
        counter.add_(-sign)
    else:
        counter.add_(sign)

    pos = counter > threshold                       # flip: −1→0, 0→+1, 1→+1 (clamped)
    neg = counter < -threshold                      # flip: +1→0, 0→−1, −1→−1 (clamped)

    # decode current ternary values as int8
    tern = unpack_ternary(packed, rows, cols)          # [N, K] int8
    new = tern.clone()
    new[pos] = torch.clamp(tern[pos].to(torch.int16) + 1, -1, 1).to(torch.int8)
    new[neg] = torch.clamp(tern[neg].to(torch.int16) - 1, -1, 1).to(torch.int8)

    # re-pack: new → codes → packed words (pad to stride*16 lanes for ragged K)
    codes = torch.where(new == 0, torch.tensor(0, device=new.device),
             torch.where(new == 1, torch.tensor(1, device=new.device),
                         torch.tensor(2, device=new.device))).to(torch.int64)
    stride = packed.shape[1]
    codes_pad = torch.zeros(rows, stride * 16, dtype=torch.int64, device=new.device)
    codes_pad[:, :cols] = codes
    packed_new = torch.zeros(rows, stride, dtype=torch.int64, device=new.device)
    for i in range(16):
        packed_new |= codes_pad[:, i::16] << (2 * i)   # int64 accum, no overflow
    packed.copy_(packed_new.to(torch.int32))

    # reset counters where flipped; leaves all others unchanged
    counter[pos | neg] = 0
    return packed


def ternary_backward_update(packed: torch.Tensor, counter: torch.Tensor,
                            X: torch.Tensor, dY: torch.Tensor, threshold: int,
                            K: int, gamma: float = 1.0,
                            descent: bool = True) -> torch.Tensor:
    """Fused backward-dX + counter update. Returns dX, mutates W/counter.

    Single call equivalent to the fused CUDA entry point
    (gemm_fused_backward_update.cu); internally two torch ops.
    """
    dX = ternary_backward_dx(packed, dY, K, gamma)
    ternary_update(packed, counter, X, dY, threshold, K, gamma=gamma,
                   descent=descent)
    return dX


class PackedTernaryLinearTorchFn(torch.autograd.Function):
    """Fused forward + backward + counter update, pure-PyTorch.

    Mirror of ``PackedTernaryLinearFn`` (packed_linear.py): forward computes
    Y = X @ W^T; backward computes dX = dY @ W for the upstream graph AND runs
    the counter → bit-flip update in-place (no dW stored, no optimizer needed).

    The ``.detach().clone().requires_grad_(True)`` trick keeps the autograd
    graph hooked onto this Function even when the input has no grad, so that
    ``backward()`` — and therefore the weight update — always executes.
    """

    @staticmethod
    def forward(ctx, X: torch.Tensor, W_packed: torch.Tensor,
                counter: torch.Tensor, in_features: int, threshold: int = 8,
                gamma: float = 1.0, descent: bool = True) -> torch.Tensor:
        if torch.is_grad_enabled() and not X.requires_grad:
            X = X.detach().clone().requires_grad_(True)
        # Plain ctx attributes, NOT save_for_backward (version-tracking in
        # save_for_backward can corrupt the saved tensor between fwd/bwd).
        ctx.X_saved = X
        ctx.W_packed = W_packed
        ctx.counter = counter
        ctx.in_features = in_features
        ctx.threshold = threshold
        ctx.gamma = gamma
        ctx.descent = descent
        return ternary_forward(W_packed, X, in_features, gamma)

    @staticmethod
    def backward(ctx, dY: torch.Tensor):
        # no_grad: the matmuls/updates here are pure bookkeeping for the
        # counter optimizer — building a grad graph inside backward() would
        # only be wasted memory (and risks in-place-modification warnings).
        with torch.no_grad():
            X = ctx.X_saved
            dX = ternary_backward_dx(ctx.W_packed, dY, ctx.in_features, ctx.gamma)
            if ctx.counter is not None:
                ternary_update(ctx.W_packed, ctx.counter, X, dY, ctx.threshold,
                               ctx.in_features, gamma=ctx.gamma, descent=ctx.descent)
        return dX, None, None, None, None, None, None


def _xavier_init(out_features: int, in_features: int,
                 gamma: float | None = None) -> torch.Tensor:
    """Xavier-uniform init packed to ternary (gamma = std when None)."""
    std = math.sqrt(2.0 / (in_features + out_features))
    if gamma is None:
        gamma = std
    W_fp32 = torch.randn(out_features, in_features) * std
    return pack_tensor(W_fp32, gamma=gamma)


class PackedTernaryLinearTorch(nn.Module):
    """nn.Module for the pure-PyTorch ternary stack (no CUDA, no optimizer).

    Trains purely through the in-place counter update run inside
    ``PackedTernaryLinearTorchFn.backward()`` — same contract as
    ``PackedTernaryLinear`` but every op is a plain torch op.
    """

    def __init__(self, in_features: int, out_features: int,
                 threshold: int = 8, bias: bool = True,
                 init_scale: float | None = None, gamma: float = 1.0,
                 descent: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.gamma = gamma
        self.descent = descent
        self.register_buffer(
            "W_packed", _xavier_init(out_features, in_features, gamma=init_scale))
        self.register_buffer(
            "counter", torch.zeros(out_features, in_features, dtype=torch.int16))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.dtype != torch.float16:
            X = X.to(torch.float16)
        # Hook PackedTernaryLinearTorchFn into the autograd graph even when
        # the root X has no grad. Without this, apply() sees only buffers
        # (W_packed/counter) + a non-grad X, so the Function is pruned and
        # its backward (counter update) never runs — while bias alone still
        # makes loss.backward() succeed, masking the bug.
        if torch.is_grad_enabled() and not X.requires_grad:
            X = X.detach().requires_grad_(True)
        Y = PackedTernaryLinearTorchFn.apply(
            X, self.W_packed, self.counter, self.in_features,
            self.threshold, self.gamma, self.descent,
        )
        if self.bias is not None:
            Y = Y + self.bias.unsqueeze(0)
        return Y

    def reset_counter(self) -> None:
        self.counter.zero_()

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"threshold={self.threshold}, gamma={self.gamma}, "
                f"descent={self.descent}")


def packed_ternary_linear_torch(packed: torch.Tensor, X: torch.Tensor,
                                counter: torch.Tensor | None,
                                threshold: int = 8, K: int | None = None,
                                gamma: float = 1.0) -> torch.Tensor:
    """Autograd-compatible forward; update runs inside backward()."""
    K = K or X.shape[1]
    return PackedTernaryLinearTorchFn.apply(
        X, packed, counter, K, threshold, gamma, True)
