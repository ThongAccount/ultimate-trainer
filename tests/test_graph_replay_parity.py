"""Phase 1 verification: CUDA-graph replay == eager (max abs diff ~0).

Root cause being guarded: every load_inline wrapper used to pass nullptr
(legacy default stream) — graph capture on the default stream then silently
drops the kernels, replaying a stale/no-op graph. This test would FAIL on the
pre-fix code and PASS with the current-stream plumbing.

Runs fwd, dX and update under torch.cuda.graph capture with static inputs,
mutates the static input, replays, and compares against a fresh eager run.

Acceptance: max abs diff == 0 (bit-exact — kernels are deterministic for
fixed inputs and shapes) for fwd and dX. Update mutates in place, so it is
compared W+counter bit-exact too.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _has_cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _pack(W_fp16):
    from kernels.packed_ternary import pack_tensor

    return pack_tensor(W_fp16)


def _decode(W_packed, N, K):
    from kernels.packed_ternary import unpack_tensor

    return unpack_tensor(W_packed, N, K)


@pytest.mark.skipif(not _has_cuda(), reason="no CUDA")
def test_fwd_graph_replay_parity():
    import torch

    torch.manual_seed(0)
    B, K, N = 256, 64, 128
    W = _pack(torch.randn(N, K, device="cuda") * 0.05)
    X_static = torch.randn(B, K, device="cuda", dtype=torch.float16)

    # warmup on a side stream (required before capture)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            from kernels.packed_ternary.custom_ops import forward_tc

            _ = forward_tc(W, X_static, K)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        from kernels.packed_ternary.custom_ops import forward_tc

        Y_static = forward_tc(W, X_static, K)

    # eager reference on NEW inputs
    X_new = torch.randn(B, K, device="cuda", dtype=torch.float16)
    X_static.copy_(X_new)
    from kernels.packed_ternary.custom_ops import forward_tc

    Y_eager = forward_tc(W, X_static, K)
    g.replay()
    torch.cuda.synchronize()

    diff = (Y_static.float() - Y_eager.float()).abs().max().item()
    assert diff == 0.0, f"fwd graph replay != eager: max abs diff {diff}"


@pytest.mark.skipif(not _has_cuda(), reason="no CUDA")
def test_dx_graph_replay_parity():
    import torch

    torch.manual_seed(0)
    B, K, N = 256, 64, 128
    W = _pack(torch.randn(N, K, device="cuda") * 0.05)
    dY_static = torch.randn(B, N, device="cuda", dtype=torch.float16)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            from kernels.packed_ternary.pack_update import backward_dx

            _ = backward_dx(W, dY_static, K)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        from kernels.packed_ternary.pack_update import backward_dx

        dX_static = backward_dx(W, dY_static, K)

    dY_new = torch.randn(B, N, device="cuda", dtype=torch.float16)
    dY_static.copy_(dY_new)
    from kernels.packed_ternary.pack_update import backward_dx

    dX_eager = backward_dx(W, dY_static, K)
    g.replay()
    torch.cuda.synchronize()

    diff = (dX_static.float() - dX_eager.float()).abs().max().item()
    assert diff == 0.0, f"dX graph replay != eager: max abs diff {diff}"


@pytest.mark.skipif(not _has_cuda(), reason="no CUDA")
def test_update_graph_replay_parity():
    import torch

    torch.manual_seed(0)
    B, K, N = 256, 64, 128
    W = _pack(torch.randn(N, K, device="cuda") * 0.05)
    X = torch.randn(B, K, device="cuda", dtype=torch.float16)
    counter0 = torch.zeros(N, K, dtype=torch.int16, device="cuda")

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        from kernels.packed_ternary.pack_update import update

        for _ in range(3):
            W0 = W.clone()
            C0 = counter0.clone()
            update(W0, C0, X, torch.randn(B, N, device="cuda", dtype=torch.float16), 32)
    torch.cuda.current_stream().wait_stream(s)

    # update mutates in place: static W/counter inputs + a dY we mutate
    dY_static = torch.randn(B, N, device="cuda", dtype=torch.float16)
    W_static = _pack(torch.zeros(N, K, device="cuda"))
    C_static = torch.zeros(N, K, dtype=torch.int16, device="cuda")
    W_static_ref = W_static.clone()
    C_static_ref = C_static.clone()

    g = torch.cuda.CUDAGraph()
    from kernels.packed_ternary.pack_update import update

    with torch.cuda.graph(g):
        update(W_static, C_static, X, dY_static, 32)

    # eager on the same fresh inputs
    from kernels.packed_ternary.pack_update import update as upd2

    dY_new = torch.randn(B, N, device="cuda", dtype=torch.float16)
    dY_static.copy_(dY_new)          # mutate the static input in place
    W_static.copy_(W_static_ref)
    C_static.copy_(C_static_ref)
    upd2(W_static, C_static, X, dY_static, 32)
    W_eager, C_eager = W_static.clone(), C_static.clone()

    # reset and replay
    dY_static.copy_(dY_new)
    W_static.copy_(W_static_ref)
    C_static.copy_(C_static_ref)
    g.replay()
    torch.cuda.synchronize()

    w_same = torch.equal(W_static, W_eager)
    c_same = torch.equal(C_static, C_eager)
    assert w_same and c_same, (
        f"update graph replay != eager: W {'ok' if w_same else 'DIFF'} "
        f"C {'ok' if c_same else 'DIFF'}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
