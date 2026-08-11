"""Correctness: pure-PyTorch impl vs CUDA ground truth (ref_linear / unpack)."""
import sys, os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernels.packed_ternary import pack_tensor, unpack_tensor
from kernels.packed_ternary.pack_forward import ref_linear
from kernels.packed_ternary.torch_impl import (
    ternary_forward, ternary_backward_dx, ternary_update, unpack_ternary,
)

torch.manual_seed(0)


def test_unpack_matches_reference():
    for (N, K) in [(64, 64), (32, 128), (128, 96), (17, 33)]:
        Wf = (torch.randn(N, K) * 0.1).float()
        Wp = pack_tensor(Wf, gamma=1.0)
        mine = unpack_ternary(Wp, N, K).float()
        ref = unpack_tensor(Wp, N, K)
        assert torch.equal(mine, ref), f"unpack mismatch {N}x{K}"
        print(f"  unpack {N}x{K} OK")


def test_forward_matches_ref():
    for (B, N, K) in [(32, 64, 64), (16, 128, 128), (8, 96, 96), (7, 33, 17)]:
        Wf = torch.randn(N, K) * 0.1
        Wp = pack_tensor(Wf, gamma=1.0)
        X = torch.randn(B, K).half()
        y_mine = ternary_forward(Wp, X, K)
        y_ref = ref_linear(Wp, X)
        err = (y_mine.float() - y_ref.float()).abs().max().item()
        assert err < 1e-2, f"fwd err {err} @ {B}x{N}x{K}"
        print(f"  fwd {B}x{N}x{K} err={err:.2e} OK")


def test_backward_dx_matches_ref():
    for (B, N, K) in [(32, 64, 64), (16, 128, 128)]:
        Wf = torch.randn(N, K) * 0.1
        Wp = pack_tensor(Wf, gamma=1.0)
        dY = torch.randn(B, N).half()
        # reference: autograd through dequantized F.linear
        tern = unpack_tensor(Wp, N, K).half()
        dX_ref = (dY.float() @ tern.float()).half()
        dX_mine = ternary_backward_dx(Wp, dY, K)
        err = (dX_mine.float() - dX_ref.float()).abs().max().item()
        assert err < 1e-2, f"bwd err {err} @ {B}x{N}x{K}"
        print(f"  bwd {B}x{N}x{K} err={err:.2e} OK")


def test_update_semantics():
    """counter += sign(dW); flip when |c|>th; reset."""
    N, K, B, th = 64, 64, 32, 8
    Wf = torch.randn(N, K) * 0.1
    Wp = pack_tensor(Wf, gamma=1.0)
    counter = torch.zeros(N, K, dtype=torch.int16)
    X = torch.randn(B, K).half()
    dY = torch.randn(B, N).half()

    w0 = unpack_ternary(Wp, N, K).clone()
    Wp = ternary_update(Wp.clone(), counter, X, dY, th, K)
    w1 = unpack_ternary(Wp, N, K)

    dW = (X.float().t() @ dY.float()).t()
    sign = torch.sign(dW).to(torch.int16)
    expected_cnt = sign  # was zeros
    assert torch.equal(counter, expected_cnt), "counter should be sign(dW) after step 1"

    # flip positions must be exactly |counter| > th
    flip_pos = (counter > th)
    flip_neg = (counter < -th)
    expected_w = w0.clone()
    expected_w[flip_pos] = torch.clamp(w0[flip_pos].to(torch.int16) + 1, -1, 1)
    expected_w[flip_neg] = torch.clamp(w0[flip_neg].to(torch.int16) - 1, -1, 1)
    assert torch.equal(w1, expected_w), "flip logic mismatch"
    # reset check
    assert torch.equal(counter[flip_pos | flip_neg],
                       torch.zeros_like(counter[flip_pos | flip_neg])), "reset missing"
    print("  update semantics OK")


if __name__ == "__main__":
    test_unpack_matches_reference()
    test_forward_matches_ref()
    test_backward_dx_matches_ref()
    test_update_semantics()
    print("ALL TORCH-IMPL TESTS PASSED")