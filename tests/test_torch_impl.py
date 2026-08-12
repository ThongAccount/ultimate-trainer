"""Correctness: pure-PyTorch impl vs CUDA ground truth (ref_linear / unpack).

Covers the spec shapes — aligned (64,64,64), (32,128,128), (128,96,96) and
ragged (17,33,65) — on CPU and (when available) CUDA:
  - unpack vs ``unpack_tensor`` decode reference
  - forward vs ``ref_linear``
  - backward dX via autograd-through-dequantized-weights
  - hand-rolled mask-math update reference + re-pack (ground-truth packer)
  - update resets counters where flipped, leaves others unchanged
  - the autograd.Function trains without an optimizer
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernels.packed_ternary import pack_tensor, unpack_tensor
from kernels.packed_ternary.pack_forward import ref_linear
from kernels.packed_ternary.torch_impl import (
    PackedTernaryLinearTorch,
    PackedTernaryLinearTorchFn,
    ternary_backward_dx,
    ternary_backward_update,
    ternary_forward,
    ternary_update,
    unpack_ternary,
)

torch.manual_seed(0)

# (B, N, K): aligned ×3 + ragged K (not a multiple of 16)
SHAPES = [(64, 64, 64), (32, 128, 128), (128, 96, 96), (17, 33, 65)]
# (N, K) pairs for the unpack/update tests (same dims, no batch)
NK = [(64, 64), (128, 128), (96, 96), (33, 65)]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _packed(N, K, device):
    """Random ternary-packed W on `device`."""
    Wf = torch.randn(N, K) * 0.1
    return pack_tensor(Wf, gamma=1.0).to(device)


def _repack(tern):
    """Re-pack decoded int8 [N,K] via the ground-truth packer (gamma=1.0)."""
    return pack_tensor(tern.float(), gamma=1.0)


# ── decode ─────────────────────────────────────────────────────────────

def test_unpack_matches_reference():
    for device in DEVICES:
        for N, K in NK:
            Wp = _packed(N, K, device)
            mine = unpack_ternary(Wp, N, K).float()
            ref = unpack_tensor(Wp.cpu(), N, K)
            assert torch.equal(mine.cpu(), ref), f"unpack mismatch {N}x{K} @ {device}"
            print(f"  unpack {N}x{K} @ {device} OK")


# ── forward ────────────────────────────────────────────────────────────

def test_forward_matches_ref():
    for device in DEVICES:
        for B, N, K in SHAPES:
            Wp = _packed(N, K, device)
            X = torch.randn(B, K).half().to(device)
            y_mine = ternary_forward(Wp, X, K)
            y_ref = ref_linear(Wp.cpu(), X.cpu())
            err = (y_mine.float().cpu() - y_ref.float()).abs().max().item()
            assert err < 1e-2, f"fwd err {err} @ {B}x{N}x{K} {device}"
            print(f"  fwd {B}x{N}x{K} @ {device} err={err:.2e} OK")


# ── backward dX via autograd-through-dequantized-weights ──────────────

def test_backward_dx_autograd():
    for device in DEVICES:
        for B, N, K in SHAPES:
            Wp = _packed(N, K, device)
            dY = torch.randn(B, N).half().to(device)
            tern = unpack_tensor(Wp.cpu(), N, K).half()          # [N, K] in {-1,0,1}
            X = torch.randn(B, K).half().to(device)
            X.requires_grad_(True)
            loss = (F.linear(X, tern.to(device)) * dY).sum()
            loss.backward()
            dX_mine = ternary_backward_dx(Wp, dY, K)
            err = (dX_mine.float() - X.grad.float()).abs().max().item()
            assert err < 1e-2, f"bwd err {err} @ {B}x{N}x{K} {device}"
            print(f"  bwd {B}x{N}x{K} @ {device} err={err:.2e} OK")


# ── update: hand-rolled mask-math reference + re-pack ─────────────────

def test_update_semantics():
    """Deterministic multi-step: flips happen (and reset), others unchanged.

    Same (X, dY) repeated 3× → counter for each weight accumulates −3·sign(dW).
    With threshold=2 the strict `>`/`<` means flips fire only on step 3.  dY
    col 3 and X col 7 are zeroed so those weights have dW ≡ 0 (never flip),
    guaranteeing both the flipped and untouched paths are exercised.
    """
    for device in DEVICES:
        N, K, B, th, steps = 32, 64, 16, 2, 3
        Wp = _packed(N, K, device)
        counter = torch.zeros(N, K, dtype=torch.int16, device=device)

        dY = torch.randn(B, N).half().to(device)
        dY[:, 3] = 0                                # row 3 of W: dW ≡ 0
        X = torch.randn(B, K).half().to(device)
        X[:, 7] = 0                                 # col 7 of W: dW ≡ 0
        data = [(X, dY)] * steps

        w0 = unpack_ternary(Wp, N, K)
        Wp_impl = Wp.clone()
        c_impl = torch.zeros_like(counter)
        for x, dy in data:
            Wp_impl = ternary_update(Wp_impl, c_impl, x, dy, th, K)
        w1 = unpack_ternary(Wp_impl, N, K)

        # hand-rolled reference: mask math on decoded weights + re-pack
        tern = w0.clone()
        c_ref = torch.zeros_like(counter)
        flipped = torch.zeros(N, K, dtype=torch.bool, device=device)
        for x, dy in data:
            dW = (x.float().t() @ dy.float()).t()
            sgn = torch.sign(dW).to(torch.int16)
            c_ref.add_(-sgn)                        # descent, = CUDA
            pos = c_ref > th
            neg = c_ref < -th
            tern[pos] = torch.clamp(tern[pos].to(torch.int16) + 1, -1, 1).to(torch.int8)
            tern[neg] = torch.clamp(tern[neg].to(torch.int16) - 1, -1, 1).to(torch.int8)
            c_ref[pos | neg] = 0
            flipped |= (pos | neg)

        assert flipped.sum().item() > 0, f"test did not produce a flip @ {device}"
        assert torch.equal(c_impl, c_ref), f"counter != hand-rolled ref @ {device}"
        assert torch.equal(w1, tern), f"flip != hand-rolled ref @ {device}"
        assert torch.equal(Wp_impl, _repack(tern).to(device)), \
            f"re-pack != ground-truth packer @ {device}"

        # reset where flipped; preserved + weight unchanged elsewhere
        assert torch.equal(c_impl[flipped], torch.zeros_like(c_impl[flipped])), \
            "counter not reset where flipped"
        assert torch.equal(c_impl[~flipped], c_ref[~flipped]), \
            "counter changed where not flipped"
        assert torch.equal(w1[~flipped], w0[~flipped]), \
            "weights changed where not flipped"
        print(f"  update {N}x{K} @ {device} OK "
              f"(flips={flipped.sum().item()}, unchanged={((~flipped).sum().item())})")


def test_update_ragged():
    """Ragged K (65) must not crash unpack/repack and must match reference."""
    for device in DEVICES:
        N, K, B, th = 33, 65, 17, 4
        Wp = _packed(N, K, device)
        counter = torch.zeros(N, K, dtype=torch.int16, device=device)
        X = torch.randn(B, K).half().to(device)
        dY = torch.randn(B, N).half().to(device)

        w0 = unpack_ternary(Wp, N, K)
        Wp = ternary_update(Wp.clone(), counter, X, dY, th, K)
        w1 = unpack_ternary(Wp, N, K)

        dW = (X.float().t() @ dY.float()).t()
        sgn = torch.sign(dW).to(torch.int16)
        c_ref = counter.clone()                     # = -sign(dW) after 1 step
        pos = c_ref > th
        neg = c_ref < -th
        tern = w0.clone()
        tern[pos] = torch.clamp(w0[pos].to(torch.int16) + 1, -1, 1).to(torch.int8)
        tern[neg] = torch.clamp(w0[neg].to(torch.int16) - 1, -1, 1).to(torch.int8)
        assert torch.equal(w1, tern), f"flip mismatch (ragged) @ {device}"
        assert torch.equal(Wp, _repack(tern).to(device)), \
            f"re-pack mismatch (ragged) @ {device}"
        print(f"  update ragged {N}x{K} @ {device} OK "
              f"(flips={pos.sum().item() + neg.sum().item()})")


def test_fused_backward_update():
    """Fused entry point ≡ backward_dx + update, one call."""
    for device in DEVICES:
        N, K, B, th = 64, 64, 32, 8
        Wp = _packed(N, K, device)
        X = torch.randn(B, K).half().to(device)
        dY = torch.randn(B, N).half().to(device)

        Wp2 = Wp.clone()
        counter2 = torch.zeros(N, K, dtype=torch.int16, device=device)
        dX_fused = ternary_backward_update(Wp2, counter2, X, dY, th, K)
        assert torch.equal(dX_fused, ternary_backward_dx(Wp, dY, K)), \
            f"fused dX mismatch @ {device}"

        Wp3 = Wp.clone()
        counter3 = torch.zeros(N, K, dtype=torch.int16, device=device)
        ternary_update(Wp3, counter3, X, dY, th, K)
        assert torch.equal(Wp2, Wp3), f"fused vs separate W mismatch @ {device}"
        assert torch.equal(counter2, counter3), \
            f"fused vs separate counter mismatch @ {device}"
        print(f"  fused {N}x{K} @ {device} OK")


# ── autograd.Function: trains without an optimizer ────────────────────

def test_autograd_fn_trains_without_optimizer():
    for device in DEVICES:
        torch.manual_seed(1)
        N, K, B, th = 32, 64, 16, 2
        layer = PackedTernaryLinearTorch(K, N, threshold=th).to(device)
        W_initial = layer.W_packed.clone()
        losses = []
        for _ in range(12):
            x = torch.randn(B, K).half().to(device)
            y = torch.randn(B, N).half().to(device)
            y_pred = layer(x)
            loss = F.mse_loss(y_pred, y)
            loss.backward()                     # update runs inside backward()
            losses.append(loss.item())

        assert all(torch.isfinite(torch.tensor(losses))), f"non-finite loss @ {device}"
        w_changed = not torch.equal(layer.W_packed, W_initial)
        c_nonzero = (layer.counter != 0).sum().item()
        assert w_changed, f"weights never flipped in 12 steps @ {device}"
        assert c_nonzero > 0, f"counter never updated @ {device}"
        print(f"  autograd train @ {device} OK (w_changed={w_changed}, "
              f"counter_nonzero={c_nonzero}, loss_final={losses[-1]:.4f})")


if __name__ == "__main__":
    test_unpack_matches_reference()
    test_forward_matches_ref()
    test_backward_dx_autograd()
    test_update_semantics()
    test_update_ragged()
    test_fused_backward_update()
    test_autograd_fn_trains_without_optimizer()
    print("ALL TORCH-IMPL TESTS PASSED")
