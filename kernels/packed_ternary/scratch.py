"""Decode-once scratch buffers (Phase 2 plumbing).

The cuBLAS backend needs W decoded to fp16 before every matmul. Allocating
inside the forward would (a) burn allocator time per call and (b) break CUDA
graph capture (torch.empty is capturable, but a fresh size per layer makes
the graph shape-specialized per layer — and capture requires a stable
pointer for static tensors).

Design: one persistent fp16 decode buffer per (N, K) shape, sized on first
use, reused afterwards. Sized for the LARGEST layer per distinct shape — the
model has exactly three distinct (N,K) shapes, so this is three buffers.

The buffer contents are refreshed by decode_into() (GPU decode kernel) before
each matmul use — decoding stays per-forward (W mutates every update step),
but the ALLOCATION happens once.
"""
import threading

_lock = threading.Lock()
_scratch: dict = {}


def get_buffer(N: int, K: int, device):
    """One persistent fp16 [N, K] decode buffer per (N, K) shape."""
    key = (N, K)
    buf = _scratch.get(key)
    if buf is None:
        with _lock:
            buf = _scratch.get(key)
            if buf is None:
                import torch

                buf = torch.empty(
                    (N, K), dtype=torch.float16, device=device
                )
                _scratch[key] = buf
    return buf


def decode_into(W_packed, N: int, K: int):
    """Decode W into the persistent buffer for (N, K) and return it.

    Buffer allocated once per shape; the decode kernel re-runs every call
    (W mutates on update steps) through the current stream, so capture is
    stable: same input pointer + same output pointer every replay.
    """
    from .pack_decode import load_decode

    buf = get_buffer(N, K, W_packed.device)
    load_decode().decode_into(W_packed, buf, N, K)
    return buf


def reset_scratch():
    """Drop all scratch buffers (test isolation / memory release)."""
    with _lock:
        _scratch.clear()


def scratch_shapes():
    return sorted(_scratch.keys())
