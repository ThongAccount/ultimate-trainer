"""PT_BACKEND=cublas — decode-once + cuBLAS GEMMs for fwd and dX.

Phase 0 measured: dispatch fwd = 1.60 TFLOPS, cuBLAS (pre-decoded fp16) =
33.15 TFLOPS — a 20.7x gap at B=16384, with decode costing only 0.105 ms.
This module routes fwd/dX through:

    W16 = decode_into(W_packed, N, K)     # persistent scratch, GPU decode
    fwd: Y  = X  @ W16^T                  # [B,K] x [K,N]
    dX : dX = dY @ W16                    # [B,N] x [N,K]

Update keeps the v2_32 counter kernel (semantics-critical; the counterword
variant is parked pending its parity diagnosis).

Env: PT_BACKEND=cublas|custom (default custom). PT_VERBOSE_DISPATCH=1 logs
the chosen backend once per shape.
"""
import os

_BACKEND = os.environ.get("PT_BACKEND", "custom").lower()
_LOGGED = set()


def use_cublas() -> bool:
    return _BACKEND == "cublas"


def _log_once(key: str, msg: str) -> None:
    if os.environ.get("PT_VERBOSE_DISPATCH") == "1" and key not in _LOGGED:
        _LOGGED.add(key)
        print(f"[dispatch] {msg}", flush=True)


def forward(W_packed, X, K: int):
    """fwd: Y = X @ W^T. W_packed is [N, stride_words] int32; W16 is [N, K]."""
    from .scratch import decode_into

    N = W_packed.size(0)
    W16 = decode_into(W_packed, N, K)
    _log_once(("fwd", N, K), f"fwd cublas {N}x{K} (decode+matmul)")
    return torch_matmul_xt(X, W16)


def backward_dx(W_packed, dY, K: int):
    """dX = dY @ W. W16 is [N, K]; dY is [B, N]."""
    from .scratch import decode_into

    N = W_packed.size(0)
    W16 = decode_into(W_packed, N, K)
    _log_once(("dx", N, K), f"dX cublas {N}x{K} (decode+matmul)")
    return torch_matmul(dY, W16)


def torch_matmul_xt(X, W16):
    """X @ W16^T — cuBLAS handles the transpose view natively."""
    import torch

    return torch.matmul(X, W16.t())


def torch_matmul(dY, W16):
    import torch

    return torch.matmul(dY, W16)
