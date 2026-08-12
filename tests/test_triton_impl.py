"""Correctness: Triton JIT impl vs ground truth (ref_linear / autograd dX) and
the pure-torch oracle (torch_impl.ternary_update) for the counter update.

Run on a GPU host (Triton kernels are CUDA-only):
    python tests/test_triton_impl.py
"""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernels.packed_ternary import pack_tensor, unpack_tensor
from kernels.packed_ternary.pack_forward import ref_linear
from kernels.packed_ternary.torch_impl import unpack_ternary, ternary_update as ref_update
from kernels.packed_ternary import triton_impl

torch.manual_seed(0)

# (B, N, K) — includes a ragged K (65) that is not a multiple of 16.
SHAPES = [(64, 64, 64), (32, 128, 128), (128, 96, 96), (17, 33, 65)]


def _make(N, K, B):
    Wf = torch.randn(N, K) * 0.1
    Wp = pack_tensor(Wf, gamma=1.0).cuda()
    X = torch.randn(B, K).half().cuda()
    dY = torch.randn(B, N).half().cuda()
    return Wp, X, dY


def test_fwd():
    for (B, N, K) in SHAPES:
        Wp, X, _ = _make(N, K, B)
        y_mine = triton_impl.ternary_forward(Wp, X, K)
        y_ref = ref_linear(Wp.cpu(), X.cpu()).cuda()
        err = (y_mine.float() - y_ref.float()).abs().max().item()
        assert err < 1e-2, f"fwd err {err} @ {B}x{N}x{K}"
        print(f"  fwd {B}x{N}x{K} err={err:.2e} OK")


def test_bwd():
    for (B, N, K) in SHAPES:
        Wp, X, dY = _make(N, K, B)
        # reference: autograd-through-dequantized W  (dX = dY @ W)
        tern = unpack_tensor(Wp.cpu(), N, K).half().cuda()
        dX_ref = (dY.float() @ tern.float()).half()
        dX_mine = triton_impl.ternary_backward_dx(Wp, dY, K)
        err = (dX_mine.float() - dX_ref.float()).abs().max().item()
        assert err < 1e-2, f"bwd err {err} @ {B}x{N}x{K}"
        print(f"  bwd {B}x{N}x{K} err={err:.2e} OK")


def test_update_matches_torch():
    """Bit-exact vs torch_impl.ternary_update after one step (gradient DESCENT).

    Includes ragged K=65 so the partial final word (1 valid col + 15 pad lanes)
    is exercised against the torch reference's padded re-pack.
    """
    # (N, K, B, threshold)
    for (N, K, B, th) in [(64, 64, 32, 8), (128, 96, 64, 16), (32, 128, 16, 4),
                          (33, 65, 16, 8)]:
        Wf = torch.randn(N, K) * 0.1
        Wp_a = pack_tensor(Wf, gamma=1.0).cuda()
        Wp_b = Wp_a.clone()
        # Start from a non-zero counter so both the add-sign path and the
        # threshold edges (flip vs no-flip positions) are exercised.
        counter_a = torch.randint(-8, 9, (N, K), dtype=torch.int16).cuda()
        counter_b = counter_a.clone()
        X = torch.randn(B, K).half().cuda()
        dY = torch.randn(B, N).half().cuda()

        triton_impl.ternary_update(Wp_a, counter_a, X, dY, th, K)
        ref_update(Wp_b, counter_b, X, dY, th, K)

        assert torch.equal(Wp_a.cpu(), Wp_b.cpu()), f"packed W mismatch @ {N}x{K}"
        assert torch.equal(counter_a.cpu(), counter_b.cpu()), f"counter mismatch @ {N}x{K}"
        print(f"  update {N}x{K} B={B} th={th} matches torch reference OK")


def test_fused_equals_separate():
    """ternary_backward_update == backward_dx + update (dX from pre-update W)."""
    for (B, N, K) in [(64, 64, 64), (32, 128, 128)]:
        Wf = torch.randn(N, K) * 0.1
        Wp_fus = pack_tensor(Wf, gamma=1.0).cuda()
        Wp_sep = Wp_fus.clone()
        cnt_fus = torch.zeros(N, K, dtype=torch.int16).cuda()
        cnt_sep = torch.zeros(N, K, dtype=torch.int16).cuda()
        X = torch.randn(B, K).half().cuda()
        dY = torch.randn(B, N).half().cuda()
        th = 8

        dX_fus = triton_impl.ternary_backward_update(Wp_fus, cnt_fus, X, dY, th, K)
        dX_sep = triton_impl.ternary_backward_dx(Wp_sep, dY, K)
        triton_impl.ternary_update(Wp_sep, cnt_sep, X, dY, th, K)

        assert torch.equal(dX_fus.cpu(), dX_sep.cpu()), "dX mismatch"
        assert torch.equal(Wp_fus.cpu(), Wp_sep.cpu()), "W mismatch"
        assert torch.equal(cnt_fus.cpu(), cnt_sep.cpu()), "counter mismatch"
        print(f"  fused {B}x{N}x{K} OK")


if __name__ == "__main__":
    test_fwd()
    test_bwd()
    test_update_matches_torch()
    test_fused_equals_separate()
    print("ALL TRITON-IMPL TESTS PASSED")
