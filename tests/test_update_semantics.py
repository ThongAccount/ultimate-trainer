"""Reference-transition gate for the production TC v2_32 update kernel.

Sweep #5 (WeightFlipGate) found the existing coverage vacuous in the strong
sense: it only checks old-vs-new parity or `flips > 0`, so wrong-column /
wrong-sign bugs shared by both arms pass.  The production update kernel
(gemm_update_tc_v2_32.cu) had ZERO direct semantic tests.

These tests close that hole: seed a known packed-W and counter state at a
known (n, k), run update steps with crafted X/dY of known-sign dW, and assert
the BIT-EXACT packed-W transition AND counter state against a Python-computed
reference that mirrors the kernel semantics exactly:

  dW = dY^T @ X            (fp16 inputs, fp32 accumulate — exact for ±1/0 data)
  cnt += (dW > 0) ? -1 : (dW < 0) ? +1 : 0
  flip (increment W) when cnt >  threshold;  flip (decrement W) when cnt < -threshold
  counter resets to 0 on flip;  increment/decrement saturate at +1 / -1.

Dims B=K=N=32 pass _tc_ok (>=16, %16==0) so the kernel under test is the same
32x32 TC tile class production runs.  The kernel is driven DIRECTLY via
pack_update._up_tc_v2_32_fn (the public update() prefers v3 at 16-63 dims —
that dispatch trap is exactly what the sweep flagged as #2/#19).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from kernels.packed_ternary import pack_update as pu

# Dims chosen so _tc_ok passes for all GEMM dims (B, N_out, N_in) but dims are
# below 64, so this is the 32x32 tile kernel (v2_32) — the production class.
B, K, N, THRESH = 32, 32, 32, 16

WEIGHTS_PER_WORD = 16
_CODE = {0: 0, 1: 1, -1: 2}  # 2-bit codes: kCode0/kCodeP1/kCodeM1


def _has_cuda():
    return torch.cuda.is_available()


def _require_tc_v2_32():
    """Load the v2_32 kernel; hard-fail on a CUDA machine if it won't load."""
    for d in (B, K, N):
        assert pu._tc_ok(d), f"test dims must dispatch to TC (dim {d} fails _tc_ok)"
    pu._load_up_tc_v2_32()
    if pu._up_tc_v2_32_fn is None or not pu._HAS_UP_TC_V2_32:
        pytest.fail("TC v2_32 update kernel failed to load on a CUDA machine")


# ── Bit-exact pack/unpack helpers (mirror packed_ternary.cuh) ────────────

def _pack_bits(W_int8):
    """[N, K] int8 ternary -> [N, stride] int32 packed (bit-exact codes)."""
    N_, K_ = W_int8.shape
    stride = (K_ + WEIGHTS_PER_WORD - 1) // WEIGHTS_PER_WORD
    packed = torch.zeros(N_, stride, dtype=torch.int32)
    for r in range(N_):
        for c in range(K_):
            packed[r, c // WEIGHTS_PER_WORD] |= _CODE[int(W_int8[r, c])] << (2 * (c % WEIGHTS_PER_WORD))
    return packed


def _unpack_bits(packed, rows, cols):
    """[N, stride] int32 packed -> [N, K] int8 (for readable assertions)."""
    out = torch.zeros(rows, cols, dtype=torch.int8)
    lut = {0: 0, 1: 1, 2: -1, 3: 0}
    for r in range(rows):
        for c in range(cols):
            code = (int(packed[r, c // WEIGHTS_PER_WORD]) >> (2 * (c % WEIGHTS_PER_WORD))) & 3
            out[r, c] = lut[code]
    return out


# ── Reference step (mirrors gemm_update_tc_v2_32.cu exactly) ─────────────

def _ref_step(W_int8, cnt, X, dY, threshold):
    """One update step. Returns (new W_int8, new counter) bit-exact."""
    dW = dY.T.float() @ X.float()  # [N, K]
    cnt = cnt.clone()
    cnt = cnt + torch.where(dW > 0, -1, torch.where(dW < 0, 1, 0))
    W = W_int8.clone()
    inc = cnt > threshold   # increment_weight: -1→0, 0→+1, +1→+1 (saturated)
    dec = cnt < -threshold  # decrement_weight: +1→0, 0→-1, -1→-1 (saturated)
    W[inc] = torch.where(W[inc] == -1, torch.zeros_like(W[inc]),
                         torch.where(W[inc] == 0, torch.ones_like(W[inc]), W[inc]))
    W[dec] = torch.where(W[dec] == 1, torch.zeros_like(W[dec]),
                         torch.where(W[dec] == 0, -torch.ones_like(W[dec]), W[dec]))
    flipped = inc | dec
    cnt[flipped] = 0
    return W, cnt


def _make_inputs(grad_sign):
    """X all +1, dY all grad_sign -> dW[n,k] = B*grad_sign (sign known, exact)."""
    X = torch.ones(B, K, dtype=torch.float16, device="cuda")
    dY = torch.full((B, N), float(grad_sign), dtype=torch.float16, device="cuda")
    return X, dY


def _run_kernel(packed_W, counter, X, dY, threshold):
    pu._up_tc_v2_32_fn(packed_W, counter, X, dY, int(threshold))


def _assert_state(packed_W, counter, W_exp_int8, cnt_exp, label):
    W_exp_packed = _pack_bits(W_exp_int8).to(torch.int32)
    assert torch.equal(packed_W.cpu(), W_exp_packed), (
        f"{label}: packed W mismatch — got 0x{int(packed_W[0,0]):08x}, "
        f"expected 0x{int(W_exp_packed[0,0]):08x}"
    )
    assert torch.equal(counter.cpu(), cnt_exp.to(torch.int16)), (
        f"{label}: counter mismatch — got {counter[0,0].item()}, "
        f"expected {int(cnt_exp[0,0])}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Reference-transition gate: (grad±) × (start W ∈ {-1, 0, +1})
#  Seed counter at ±(THRESH-1); step 1 lands exactly at ±THRESH (no flip —
#  kernel is strict >); step 2 crosses and flips, resetting the counter.
# ═══════════════════════════════════════════════════════════════════════════════

def _transition_case(grad_sign, w_start):
    if not _has_cuda():
        return
    _require_tc_v2_32()
    X, dY = _make_inputs(grad_sign)

    W_int8 = torch.full((N, K), w_start, dtype=torch.int8)
    # grad<0 → counter increments toward +threshold; grad>0 → toward -threshold
    seed = (THRESH - 1) if grad_sign < 0 else -(THRESH - 1)
    cnt = torch.full((N, K), seed, dtype=torch.int32)

    packed_W = _pack_bits(W_int8).cuda()
    counter = cnt.to(torch.int16).cuda()

    # Step 1: counter lands exactly AT ±threshold — strict > means NO flip yet.
    _run_kernel(packed_W, counter, X, dY, THRESH)
    W_int8, cnt = _ref_step(W_int8, cnt, X.cpu(), dY.cpu(), THRESH)
    _assert_state(packed_W, counter, W_int8, cnt, f"grad{grad_sign:+d}/W{w_start:+d} step1")
    assert int(counter[0, 0]) in (THRESH, -THRESH), "step1 must park exactly at threshold"
    assert _unpack_bits(packed_W.cpu(), N, K).eq(w_start).all(), "no flip allowed at threshold"

    # Step 2: crosses threshold → bit-exact flip, counter reset to 0.
    _run_kernel(packed_W, counter, X, dY, THRESH)
    W_int8, cnt = _ref_step(W_int8, cnt, X.cpu(), dY.cpu(), THRESH)
    _assert_state(packed_W, counter, W_int8, cnt, f"grad{grad_sign:+d}/W{w_start:+d} step2")
    assert int(counter[0, 0]) == 0, "counter must reset to 0 after flip"
    # Expected terminal weight: saturating ±1 walk in the descent direction.
    exp_w = w_start - grad_sign  # grad<0 walks +1; grad>0 walks -1; saturates
    exp_w = max(-1, min(1, exp_w))
    assert int(_unpack_bits(packed_W.cpu(), N, K)[0, 0]) == exp_w


def test_transition_grad_neg_w_m1():
    """grad<0, W=-1 → flip to 0, counter +15 → +16(no flip) → +17(flip) → 0."""
    _transition_case(-1, -1)


def test_transition_grad_neg_w_0():
    """grad<0, W=0 → flip to +1."""
    _transition_case(-1, 0)


def test_transition_grad_neg_w_p1():
    """grad<0, W=+1 → saturated: W bits unchanged, counter still resets."""
    _transition_case(-1, 1)


def test_transition_grad_pos_w_p1():
    """grad>0, W=+1 → flip to 0."""
    _transition_case(1, 1)


def test_transition_grad_pos_w_0():
    """grad>0, W=0 → flip to -1."""
    _transition_case(1, 0)


def test_transition_grad_pos_w_m1():
    """grad>0, W=-1 → saturated: W bits unchanged, counter still resets."""
    _transition_case(1, -1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Repeated-update counter carry: counter persists across non-flip steps and
#  resets only on the flip step.
# ═══════════════════════════════════════════════════════════════════════════════

def test_counter_carry_across_steps():
    if not _has_cuda():
        return
    _require_tc_v2_32()
    X, dY = _make_inputs(-1)  # counter walks +1 per step

    W_int8 = torch.zeros(N, K, dtype=torch.int8)
    cnt = torch.full((N, K), THRESH - 4, dtype=torch.int32)
    packed_W = _pack_bits(W_int8).cuda()
    counter = cnt.to(torch.int16).cuda()

    carry = [THRESH - 4, THRESH - 3, THRESH - 2, THRESH - 1, THRESH, 0]
    for step, expected_cnt in enumerate(carry):
        _run_kernel(packed_W, counter, X, dY, THRESH)
        W_int8, cnt = _ref_step(W_int8, cnt, X.cpu(), dY.cpu(), THRESH)
        _assert_state(packed_W, counter, W_int8, cnt, f"carry step {step}")
        assert int(counter[0, 0]) == expected_cnt, (
            f"step {step}: counter carry broken — got {int(counter[0,0])}, "
            f"expected {expected_cnt}"
        )
    # 4 non-flip steps must leave W untouched; the 5th flips 0 → +1.
    assert int(_unpack_bits(packed_W.cpu(), N, K)[0, 0]) == 1


# ═══════════════════════════════════════════════════════════════════════════════
#  Strict-boundary: counter parked exactly AT ±threshold does NOT flip
#  (kernel uses strict > / <, not >=).
# ═══════════════════════════════════════════════════════════════════════════════

def test_counter_at_threshold_does_not_flip():
    if not _has_cuda():
        return
    _require_tc_v2_32()
    # Zero gradients: counter parked at ±threshold must stay put, W untouched.
    X = torch.zeros(B, K, dtype=torch.float16, device="cuda")
    dY = torch.zeros(B, N, dtype=torch.float16, device="cuda")

    W_int8 = torch.zeros(N, K, dtype=torch.int8)
    packed_W = _pack_bits(W_int8).cuda()
    cnt = torch.full((N, K), THRESH, dtype=torch.int32)
    counter_pos = cnt.to(torch.int16).cuda()
    counter_neg = (-cnt).to(torch.int16).cuda()

    for tag, counter in (("+T", counter_pos), ("-T", counter_neg)):
        _run_kernel(packed_W, counter, X, dY, THRESH)
        seed = THRESH if tag == "+T" else -THRESH
        assert int(counter[0, 0]) == seed, (
            f"counter at {tag} moved under zero grad — got {int(counter[0,0])}"
        )
        assert _unpack_bits(packed_W.cpu(), N, K).eq(0).all(), (
            f"W flipped while counter parked at {tag} — strict > violated"
        )

    # A same-direction step that lands exactly ON ±threshold must not flip.
    X1, dY1 = _make_inputs(-1)  # counter walks +
    packed_W = _pack_bits(torch.zeros(N, K, dtype=torch.int8)).cuda()
    counter = torch.full((N, K), THRESH - 1, dtype=torch.int16, device="cuda")
    _run_kernel(packed_W, counter, X1, dY1, THRESH)
    assert int(counter[0, 0]) == THRESH
    assert _unpack_bits(packed_W.cpu(), N, K).eq(0).all(), (
        "flip at counter == threshold: kernel must use strict >"
    )


if __name__ == "__main__":
    tests = [
        test_transition_grad_neg_w_m1,
        test_transition_grad_neg_w_0,
        test_transition_grad_neg_w_p1,
        test_transition_grad_pos_w_p1,
        test_transition_grad_pos_w_0,
        test_transition_grad_pos_w_m1,
        test_counter_carry_across_steps,
        test_counter_at_threshold_does_not_flip,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ✅ {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  ❌ {fn.__name__}: {e}")
            import traceback; traceback.print_exc()
    print(f"\n{'FAILED' if failed else 'ALL PASSED'} ({failed} failures)")
    sys.exit(1 if failed else 0)
