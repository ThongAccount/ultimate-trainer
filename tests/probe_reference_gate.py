"""Reference-transition gate for the packed-ternary update kernel (sweep #5).

The syncwarp probes' old-vs-new parity check is vacuous in the strong sense
(docs/speedpass/2026-10-10-20-agent-crack-sweep.md, investigation #5): it
only proves OLD == NEW, so any bug shared by BOTH arms — wrong column,
inverted counter sign convention, double flip, missing reset — passes,
because both arms produce the same wrong answer.

This module pins the kernel to ABSOLUTE semantics: a deterministic fixture
whose expected post-update state is computed in Python from the production
rules read off kernels/packed_ternary/gemm_update_tc_v2_32.cu (line numbers
at commit 6218dde + C13 counter clamp; see kernel for the authoritative
source):

    counter update runs on int32 PAIRS (k, k+1), k even (:162-196);
    a pair is SKIPPED entirely iff BOTH grads are zero (:182-183);
    per element:  cnt += (dW > 0) ? -1 : (dW < 0) ? +1 : 0   (:198-200)
                  — the counter moves AGAINST the gradient
    flip when     cnt >  threshold  (STRICT, :212)  -> W = increment(W)
    or when       cnt < -threshold  (STRICT, :215)  -> W = decrement(W)
    on flip:      cnt = 0 (reset, :214/:217)

    increment: -1 -> 0, 0 -> +1, +1 -> +1  (saturates at +1)
    decrement: +1 -> 0, 0 -> -1, -1 -> -1  (saturates at -1)

    (packed_ternary.cuh atomic CAS variants; saturating, code 11 = INVALID
    never produced by a correct kernel and never expected here)

Fixture design: B=16 (exactly one kK=16 WMMA tile -> single deterministic
pass), in=out=64 (full 32x32 super-tiles, no partial tiles, no odd tail).
Every active probe position uses its own batch row and a distinct (n, k), so
each crafted dW[n,k] is exactly +/-1.0 (fp16-exact products, fp32-exact
accumulation) and every other dW entry is exactly 0.0. Expected W / counter
arrays are asserted bit-exact.

Bug classes this gate catches that OLD==NEW parity cannot (all demonstrated
by simulation in the C13 fix-batch session):
  - inverted counter sign convention: a negative grad pushing the counter
    past +T must flip via INCREMENT (and vice versa on the -T side)
  - off-by-one threshold (>= instead of >): a counter landing exactly at
    +T or -T must NOT flip
  - wrong-column / wrong-row addressing: crafted grads live at known
    (n, k); an addressing bug flips a different packed 2-bit slot
  - double flip / missing reset: the counter must be exactly 0 after a
    flip, and W must move exactly one ternary step
  - W transition direction and saturation (-1 -> 0 -> +1, clamped at +/-1)
  - pair-skip rule: a pair with BOTH grads zero must leave its counters
    completely untouched (skip path), while a zero-grad element whose
    pair-mate is ACTIVE must still be flip-checked (and flip if its
    counter is past threshold)
  - plain counter arithmetic (+/-1 moves that cross no boundary)

Note: in_features=64 is even, so only the aligned int32 pair path of the
counter loop runs here; the odd-in_features scalar tail is a separate,
known-latent path (sweep #1) and is intentionally not exercised.

The C13 int16 counter clamp (saturate to ±32767 before the threshold check,
sweep #4) is a no-op at this fixture's counter range (|cnt| <= T+3), so the
gate is agnostic to its presence in either arm.

Usage (probe or pytest):

    from tests.probe_reference_gate import run_reference_gate
    run_reference_gate(kernel.upd, "cuda", threshold=8)

raises AssertionError with the first mismatches on failure. `upd` must
match the load_inline wrapper used by the probes:

    upd(X, dY, W_packed_int32, counter_int16, in_features, threshold)

selfcheck() runs the pure-Python expectation against hand-derived values
on CPU (no CUDA needed) and is a real unit check of this module's math.

Usage: python tests/probe_reference_gate.py   (CPU selfcheck only)
"""

try:
    import torch
except ImportError:  # pragma: no cover - allows pytest collection w/o torch
    torch = None

_CODE = {0: 0, 1: 1, -1: 2}  # packed-ternary 2-bit codes (packed_ternary.cuh)


def _positions(threshold):
    """Probe positions: (n, k, batch_row, grad_sign, w_init, cnt_init, case).

    grad_sign: +1 -> dW = +1.0 (counter delta -1); -1 -> dW = -1.0 (+1);
               0  -> no dY/X contribution (dW = 0.0).
    batch_row is unused (and unconstrained) when grad_sign == 0.
    """
    T = threshold
    return [
        # -- flips via increment (positive grad, cnt crosses +T) ----------
        (0, 0, 0, +1, -1, T + 2, "flip inc: W -1->0, cnt reset"),
        (16, 16, 1, +1, 0, T + 2, "flip inc: W 0->+1, cnt reset"),
        (33, 40, 2, +1, +1, T + 2, "flip inc saturating at +1 (reset proves flip)"),
        # -- flips via decrement (negative grad, cnt crosses -T) -----------
        (48, 32, 3, -1, +1, -(T + 2), "flip dec: W +1->0, cnt reset"),
        (9, 25, 4, -1, 0, -(T + 2), "flip dec: W 0->-1, cnt reset"),
        (21, 48, 5, -1, -1, -(T + 2), "flip dec saturating at -1 (reset proves flip)"),
        # -- boundary: lands EXACTLY at +/-T -> must NOT flip (strict >/<) --
        (37, 9, 6, +1, 0, T + 1, "boundary: +T+1 - 1 = +T, no flip (strict >)"),
        (52, 21, 7, -1, 0, -(T + 1), "boundary: -T-1 + 1 = -T, no flip (strict <)"),
        # -- sign-convention catches: grad pushes counter across the
        #    OPPOSITE-side boundary -> flip direction reveals the convention
        (2, 41, 8, -1, 0, T + 1, "neg grad across +T -> INCREMENT flip, W 0->+1"),
        (7, 55, 11, +1, 0, -(T + 1), "pos grad across -T -> DECREMENT flip, W 0->-1"),
        # -- plain arithmetic, no boundary crossed --------------------------
        (29, 57, 9, +1, 0, T - 1, "arithmetic: +T-1 - 1 = T-2"),
        (45, 5, 10, -1, 0, T - 1, "arithmetic: +T-1 + 1 = +T, no flip"),
        (20, 30, 12, +1, 0, 0, "arithmetic: fresh counter 0 -> -1"),
        # -- pair-skip rule: BOTH grads zero -> pair skipped entirely --------
        (12, 12, 0, 0, -1, T + 3, "both-zero pair skipped: counter and W untouched"),
        # -- zero-grad element whose pair-mate (16,16) is ACTIVE: the pair is
        #    NOT skipped, so this element is still flip-checked. Its grad adds
        #    0 but its past-threshold counter must flip+reset (W 0->+1).
        #    Pins the pair granularity of the skip optimization (:182-183).
        (16, 17, 0, 0, 0, T + 3, "zero-grad mate of active pair: still flip-checked"),
    ]


def _increment(w):
    return {-1: 0, 0: 1, 1: 1}[w]  # saturates at +1


def _decrement(w):
    return {1: 0, 0: -1, -1: -1}[w]  # saturates at -1


def _expected_step(w, c, grad, threshold):
    """One element's production update, pure Python (the reference)."""
    c += -1 if grad > 0 else (1 if grad < 0 else 0)
    if c > threshold:
        w, c = _increment(w), 0
    elif c < -threshold:
        w, c = _decrement(w), 0
    return w, c


def build_fixture(threshold, device="cuda"):
    """Deterministic (X, dY, W0, C0) plus the Python-computed expectation."""
    in_f, out_f, B = 64, 64, 16
    pos = _positions(threshold)

    # Only ACTIVE positions (grad != 0) inject into dY/X, so only those need
    # distinct (n, k, batch_row); a zero-grad position may share its row with
    # an active one (e.g. the same-int32-pair mate probe) — it injects
    # nothing and its W/counter slots are seeded directly.
    act = [p for p in pos if p[3] != 0]
    ns = [p[0] for p in act]
    ks = [p[1] for p in act]
    bs = [p[2] for p in act]
    assert len(set(ns)) == len(ns), "active probe rows (n) must be distinct"
    assert len(set(ks)) == len(ks), "active probe cols (k) must be distinct"
    assert len(set(bs)) == len(bs) and max(bs) < B, "batch rows must be distinct"

    X = torch.zeros(B, in_f, device=device, dtype=torch.float16)
    dY = torch.zeros(B, out_f, device=device, dtype=torch.float16)
    stride = (in_f + 15) // 16
    W0 = torch.zeros(out_f, stride, device=device, dtype=torch.int32)
    C0 = torch.zeros(out_f * in_f, device=device, dtype=torch.int16)

    for n, k, b, g, _w, _c, _case in pos:
        if g != 0:
            dY[b, n] = 1.0
            X[b, k] = float(g)  # dW[n,k] = 1.0 * g = g, exact in fp16/fp32

    for n, k, _b, _g, w0, c0, _case in pos:
        W0[n, k // 16] |= _CODE[w0] << (2 * (k % 16))
        C0[n * in_f + k] = c0

    # ── Expected state: replay the kernel's PAIR loop in Python ──────
    # grad[n][k] is the crafted dW; w/c default to 0 everywhere unseeded.
    # A pair (k, k+1), k even, is processed iff at least one grad is nonzero
    # (kernel :182-183 skips only when BOTH are zero); unprocessed pairs
    # leave W and counters bit-identical.
    grad = {(p[0], p[1]): p[3] for p in pos}
    wstate = {(p[0], p[1]): p[4] for p in pos}
    cstate = {(p[0], p[1]): p[5] for p in pos}

    touched_pairs = sorted({(n, k - (k & 1)) for n, k, *_ in pos})
    exp_flips = 0
    for n, k in touched_pairs:
        g0 = grad.get((n, k), 0)
        g1 = grad.get((n, k + 1), 0)
        if g0 == 0 and g1 == 0:
            continue  # both-zero pair: skipped by the kernel
        for g, kk in ((g0, k), (g1, k + 1)):
            w = wstate.get((n, kk), 0)
            c = cstate.get((n, kk), 0)
            w1, c1 = _expected_step(w, c, g, threshold)
            wstate[(n, kk)] = w1
            cstate[(n, kk)] = c1
            if c1 == 0 and c != 0:
                exp_flips += 1

    W_exp = torch.zeros_like(W0)
    C_exp = torch.zeros_like(C0)
    for (n, k), w in wstate.items():
        W_exp[n, k // 16] |= _CODE[w] << (2 * (k % 16))
    for (n, k), c in cstate.items():
        C_exp[n * in_f + k] = c

    return {
        "X": X, "dY": dY, "W0": W0, "C0": C0,
        "W_exp": W_exp, "C_exp": C_exp,
        "in_features": in_f, "out_features": out_f, "batch": B,
        "positions": pos, "exp_flips": exp_flips,
    }


def run_reference_gate(upd, device="cuda", threshold=8):
    """Run ONE update through `upd` and assert bit-exact expected transitions.

    upd(X, dY, W_packed_int32, counter_int16, in_features, threshold)
    — matches the load_inline wrapper built by tests/probe_update_syncwarp2.py.

    Raises AssertionError listing the first mismatches on failure; returns a
    summary dict on pass. Safe to call from pytest (AssertionError) or a
    probe (caller converts to SystemExit).
    """
    fx = build_fixture(threshold, device)
    W, C = fx["W0"].clone(), fx["C0"].clone()
    upd(fx["X"], fx["dY"], W, C, fx["in_features"], threshold)
    if str(W.device).startswith("cuda"):
        torch.cuda.synchronize()

    errs = []
    if not torch.equal(W, fx["W_exp"]):
        for r, wi in (W != fx["W_exp"]).nonzero()[:4].tolist():
            errs.append(f"W[{r},{wi}] got=0x{int(W[r, wi]) & 0xFFFFFFFF:08x} "
                        f"want=0x{int(fx['W_exp'][r, wi]) & 0xFFFFFFFF:08x}")
    if not torch.equal(C, fx["C_exp"]):
        for i in (C != fx["C_exp"]).nonzero().flatten()[:4].tolist():
            errs.append(f"counter[{i // fx['in_features']},{i % fx['in_features']}] "
                        f"got={int(C[i])} want={int(fx['C_exp'][i])}")
    if errs:
        raise AssertionError(
            "reference-transition mismatch (semantics, not parity): "
            + "; ".join(errs))

    n_flips = int((W != fx["W0"]).sum().item())
    n_resets = int(((fx["C0"] != 0) & (C == 0)).sum().item())
    if n_resets != fx["exp_flips"]:
        raise AssertionError(
            f"flip/reset count wrong: got {n_resets}, expected {fx['exp_flips']}")
    return {
        "flips": n_flips,          # packed-word bit changes (saturated flips excluded)
        "resets": n_resets,        # counter resets == flips incl. saturated ones
        "positions": len(fx["positions"]),
    }


def _decode(W, n, k):
    return {0: 0, 1: 1, 2: -1}[int(W[n, k // 16]) >> (2 * (k % 16)) & 3]


def selfcheck():
    """CPU-only check of this module's expectation math against hand-derived
    values for threshold=8 (the probe's THRESHOLD). No CUDA required."""
    fx = build_fixture(8, device="cpu")
    T = 8
    # (n, k, w_exp, c_exp) hand-derived from the kernel rules:
    hand = [
        (0, 0, 0, 0),        # cnt 10, grad + -> 9 > 8: inc flip, W -1->0
        (16, 16, 1, 0),      # W 0->+1
        (33, 40, 1, 0),      # W +1 saturates; reset to 0 proves the flip
        (48, 32, 0, 0),      # cnt -10, grad - -> -9 < -8: dec flip, W +1->0
        (9, 25, -1, 0),      # W 0->-1
        (21, 48, -1, 0),     # W -1 saturates; reset proves the flip
        (37, 9, 0, 8),       # 9 - 1 = 8 = T exactly -> NO flip (strict >)
        (52, 21, 0, -8),     # -9 + 1 = -T exactly -> NO flip (strict <)
        (2, 41, 1, 0),       # neg grad: 9 + 1 = 10 > 8 -> INCREMENT, W 0->+1
        (7, 55, -1, 0),      # pos grad: -9 - 1 = -10 < -8 -> DECREMENT, W 0->-1
        (29, 57, 0, T - 2),  # 7 - 1 = 6
        (45, 5, 0, 8),       # 7 + 1 = 8 = T -> no flip
        (20, 30, 0, -1),     # fresh counter 0 - 1 = -1
        (12, 12, -1, T + 3),  # both-zero pair (12,13): skipped, untouched
        (16, 17, 1, 0),      # zero-grad mate of ACTIVE (16,16): pair processed,
                             # cnt 11 + 0 = 11 > 8 -> inc flip W 0->+1, reset
    ]
    for n, k, w_exp, c_exp in hand:
        w_got, c_got = _decode(fx["W_exp"], n, k), int(fx["C_exp"][n * 64 + k])
        assert w_got == w_exp, f"W[{n},{k}] got {w_got}, want {w_exp}"
        assert c_got == c_exp, f"C[{n},{k}] got {c_got}, want {c_exp}"
    # 8 crossing/sign flips + the pair-mate flip at (16,17) = 9 resets
    assert fx["exp_flips"] == 9, fx["exp_flips"]
    # W0 packing spot-check: pos (0,0) seeds W=-1 -> code 2 in bits 0-1
    assert _decode(fx["W0"], 0, 0) == -1
    # dW exactness precondition: every crafted entry is +/-1.0, all else 0
    dW = fx["dY"].float().t() @ fx["X"].float()  # (out, in)
    for n, k, b, g, _w, _c, _ in fx["positions"]:
        if g:
            assert int(dW[n, k]) == g and abs(float(dW[n, k])) == 1.0
    crafted = {(p[0], p[1]) for p in fx["positions"] if p[3]}
    nz = {(int(i) // 64, int(i) % 64) for i in dW.flatten().nonzero().flatten().tolist()}
    assert nz == crafted, "dW nonzero outside crafted positions"
    print(f"selfcheck OK: {len(hand)} positions, exp_flips={fx['exp_flips']}, "
          f"dW exact (±1.0 at {len(crafted)} crafted positions, 0 elsewhere)")


if __name__ == "__main__":
    selfcheck()
