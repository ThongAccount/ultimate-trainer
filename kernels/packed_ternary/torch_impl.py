"""Pure-PyTorch ternary linear: forward / backward-dx / counter update.

Reference implementation in plain PyTorch tensor ops (no CUDA C++, no Triton).
Matches the CUDA kernels' packed 2-bit ternary layout:
  - W is packed int32 [N, stride_words], 16 ternary codes per word, little-endian
    bit order: word = Σ code_i << (2i),  code 0→0, 1→+1, 2→−1.
  - counter is int16 [N, K].
  - update: dW[r,c] = Σ_b dY[b,r]·X[b,c];  counter += sign(dW);
    flip −1→0→+1 when counter > +threshold, reverse when < −threshold, reset.

Correctness contract: outputs match the CUDA stack (ref_linear ground truth).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import compute_stride_words, unpack_tensor  # noqa: F401  (for tests)

_LUT = torch.tensor([0, 1, -1, 0], dtype=torch.int8)


def unpack_ternary(packed: torch.Tensor, rows: int, cols: int,
                   device: torch.device | None = None) -> torch.Tensor:
    """Decode packed int32 [rows, stride_words] → int8 [rows, cols] in {-1,0,1}.

    Vectorized: no Python loop over rows/cols (16 fixed-width shifts).
    """
    stride = packed.shape[1]
    w = packed.to(torch.int64) & 0xFFFFFFFF  # treat as unsigned, keep 64-bit
    out = torch.empty(rows, cols, dtype=torch.int8, device=packed.device)
    for i in range(16):
        code = ((w >> (2 * i)) & 0b11).to(torch.int8)
        out[:, i::16] = _LUT.to(device=packed.device)[code]
    return out


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
                   K: int) -> torch.Tensor:
    """In-place counter → flip update. Returns updated packed (same object).

    dW = X^T dY in FP32 (never materialized in the CUDA kernel, but the math
    is identical); counter += sign(dW); flip when |counter| > threshold.
    """
    rows, cols = packed.shape[0], K
    dW = X.float().t() @ dY.float()          # [K, N] → transpose to [N, K]
    dW = dW.t()
    sign = torch.sign(dW).to(counter.dtype)  # ±1
    counter.add_(sign)

    pos = counter > threshold                 # +1 flip: -1→0, 0→+1, 1→1
    neg = counter < -threshold                # −1 flip: 1→0, 0→−1, −1→−1

    # decode current ternary values as int8
    tern = unpack_ternary(packed, rows, cols)          # [N, K] int8
    new = tern.clone()
    new[pos] = torch.clamp(tern[pos].to(torch.int16) + 1, -1, 1).to(torch.int8)
    new[neg] = torch.clamp(tern[neg].to(torch.int16) - 1, -1, 1).to(torch.int8)

    # re-pack: new → codes → packed words
    codes = torch.where(new == 0, torch.tensor(0, device=new.device),
             torch.where(new == 1, torch.tensor(1, device=new.device),
                         torch.tensor(2, device=new.device))).to(torch.int64)
    packed_new = torch.zeros(rows, compute_stride_words(cols),
                             dtype=torch.int32, device=packed.device)
    for i in range(16):
        packed_new[:, :] |= (codes[:, i::16] << (2 * i)).to(torch.int32)
    packed.copy_(packed_new)

    # reset counters where flipped
    counter[pos | neg] = 0
    return packed


def packed_ternary_linear_torch(packed: torch.Tensor, X: torch.Tensor,
                                counter: torch.Tensor | None,
                                threshold: int = 8, K: int | None = None,
                                gamma: float = 1.0) -> torch.Tensor:
    """One full forward (autograd-compatible, update hook outside)."""
    K = K or X.shape[1]
    return ternary_forward(packed, X, K, gamma)
