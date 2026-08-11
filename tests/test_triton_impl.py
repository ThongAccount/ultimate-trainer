"""Correctness: Triton JIT impl vs CUDA ground truth (ref_linear / unpack)."""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernels.packed_ternary import pack_tensor, unpack_tensor
from kernels.packed_ternary.pack_forward import ref_linear
from kernels.packed_ternary.torch_impl import unpack_ternary, ternary_update as ref_update
from kernels.packed_ternary import triton_impl

torch.manual_seed(0)


def test_fwd():
    for (B, N, K) in [(64, 64, 64), (32, 128, 128), (128, 96, 96), (17, 33, 65)]:
        Wf = torch.randn(N, K) * 0.1
        Wp = pack_tensor(Wf, gamma=1.0).cuda()
        X = torch.randn(B, K).half().cuda()
        y_mine = triton_impl.ternary_forward(Wp, X, K)
        y_ref = ref_linear(Wp.cpu(), X.cpu()).cuda()
        err = (y_mine.float() - y_ref.float()).abs().max().item()
        assert err < 1e-2, f"fwd err {err} @ {B}x{N}x{K}"
        print(f"  fwd {B}x{N}x{K} err={err:.2e} OK")


def test_bwd():
    for (B, N, K) in [(64, 64, 64), (32, 128, 128)]:
        Wf = torch.randn(N, K) * 0.1
        Wp = pack_tensor(Wf, gamma=1.0).cuda()
        dY = torch.randn(B, N).half().cuda()
        tern = unpack_tensor(Wp.cpu(), N, K).half().cuda()
        dX_ref = (dY.float() @ tern.float()).half()
        dX_mine = triton_impl.ternary_backward_dx(Wp, dY, K)
        err = (dX_mine.float() - dX_ref.float()).abs().max().item()
        assert err < 1e-2, f"bwd err {err} @ {B}x{N}x{K}"
        print(f"  bwd {B}x{N}x{K} err={err:.2e} OK")


def test_update():
    """Triton update must equal the pure-torch reference update."""
    N, K, B, th = 64, 64, 32, 8
    Wf = torch.randn(N, K) * 0.1
    Wp_a = pack_tensor(Wf, gamma=1.0).cuda()
    Wp_b = Wp_a.clone()
    counter_a = torch.zeros(N, K, dtype=torch.int16).cuda()
    counter_b = torch.zeros(N, K, dtype=torch.int16).cuda()
    X = torch.randn(B, K).half().cuda()
    dY = torch.randn(B, N).half().cuda()

    triton_impl.ternary_update(Wp_a, counter_a, X, dY, th, K)
    ref_update(Wp_b, counter_b, X, dY, th, K)

    assert torch.equal(Wp_a.cpu(), Wp_b.cpu()), "packed W mismatch"
    assert torch.equal(counter_a.cpu(), counter_b.cpu()), "counter mismatch"
    print("  update matches torch reference OK")


if __name__ == "__main__":
    test_fwd()
    test_bwd()
    test_update()
    print("ALL TRITON-IMPL TESTS PASSED")