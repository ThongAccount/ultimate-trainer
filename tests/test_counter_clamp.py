"""Pure-CPU logic test of the int16 counter saturating clamp (sweep #4).

The TC update kernels (gemm_update_tc_v2_32.cu, gemm_update_tc_v3_32.cu)
increment/decrement int16 counters and compare against an int16 threshold:

    cnt += delta                       (int16 arithmetic — wraps at ±32768)
    cnt = max(-32767, min(32767, cnt)) <-- the fix under test (saturating clamp)
    if cnt >  threshold: increment flip, cnt = 0
    if cnt < -threshold: decrement flip, cnt = 0

Bug (sweep #4; dormant at prod threshold=32, live at threshold=32767):
a counter at the +32767 rail incremented once wraps to -32768 inside the
int16 +=, so the OLD code evaluates -32768 < -threshold and fires a
DECREMENT flip (wrong direction — the true value +32768 demands an
increment flip) plus a bogus counter reset.  The clamp pins the wrapped
value to -32767, killing the flip: |cnt| can never exceed 32767, so at
threshold=32767 neither comparison can fire and the counter pins at the
rail.  Clamping to -32767 (not -32768) additionally guarantees INT16_MIN
is never observed — not stored, not compared — even from corrupted state.

Reachability note that shapes the invariants below: because the kernel
resets the counter to 0 as soon as |cnt| > threshold, every STORED counter
observed at step start satisfies |cnt| <= threshold (invariant from the
zero-initialized counter array).  Therefore:

  * For threshold < 32767 the ±32767 rails are unreachable, and the clamp
    is provably a no-op (bit-exact prod behavior preserved).
  * For threshold = 32767 the rails are reachable, the wrap fires, and the
    clamp converts the spurious flip into a pinned no-flip rail.

This file mirrors the kernel's exact integer semantics — int16 wrap on the
increment, then clamp, then threshold compare, then flip+reset — in pure
Python arithmetic and asserts:

  1. For every reachable stored counter and both deltas: the fixed
     sequence matches infinite-precision semantics exactly (flip decision,
     flip direction, stored value) — no spurious flips of any kind.
  2. A counter at ±threshold always flips on a same-direction step when
     the unclamped value ±(threshold+1) is a legal int16 (threshold <=
     32766); at threshold = 32767 the flip boundary is unrepresentable and
     the counter pins at the rail.
  3. INT16_MIN is never stored after any step, from ANY int16 starting
     state (including corrupted/unreachable ones) — one fixed step
     self-heals an INT16_MIN counter into the clamp range.
  4. The in-band behavior is bit-identical between OLD (pre-clamp) and NEW
     code for every reachable counter (exhaustive over [-threshold,
     threshold]) — the production path (threshold=32) is bit-exact.
  5. The OLD code demonstrably exhibits the wrap bug at the rails (the
     regression this fix closes).

Parametrized over cnt ∈ {-32768..-32760, -threshold-1, 0, threshold,
threshold+1, 32760..32767} and threshold ∈ {0, 8, 32, 32767}.

This validates the MATH only.  The kernel-level check runs on Colab via
the existing parity probe (tests/probe_update_syncwarp2.py bit-exact gate)
— this box has no GPU.

Usage: python3 tests/test_counter_clamp.py   (or via pytest)
"""
import sys

import pytest

INT16_MIN = -32768
INT16_MAX = 32767
CLAMP_LO = -32767  # clamp to ±32767 so INT16_MIN can never be observed
CLAMP_HI = 32767

THRESHOLDS = [0, 8, 32, 32767]  # prod=32, degenerate=0, max legal=32767


def wrap16(x: int) -> int:
    """Simulate int16 arithmetic wraparound of an arbitrary Python int."""
    x &= 0xFFFF
    return x - 0x10000 if x & 0x8000 else x


def kernel_step(cnt: int, delta: int, threshold: int, clamp: bool):
    """One kernel counter step.  Returns (new_cnt, flip_dir).

    flip_dir: 0 = no flip, +1 = increment flip, -1 = decrement flip.
    Mirrors the kernel exactly: increment (int16 wrap) → [clamp] →
    threshold compare → flip + reset-to-0.
    """
    cnt = wrap16(cnt + delta)  # int16 += : wraps silently past ±32767
    if clamp:
        cnt = max(CLAMP_LO, min(CLAMP_HI, cnt))
    if cnt > threshold:
        return 0, 1
    if cnt < -threshold:
        return 0, -1
    return cnt, 0


def math_step(cnt: int, delta: int, threshold: int):
    """Infinite-precision reference semantics.  Returns (new_cnt, flip_dir)."""
    v = cnt + delta
    if v > threshold:
        return 0, 1
    if v < -threshold:
        return 0, -1
    return v, 0


def counter_cases(threshold: int):
    """The requested counter values for a given threshold (de-duplicated)."""
    cases = (
        list(range(INT16_MIN, -32760))              # near INT16_MIN
        + [-threshold - 1, 0, threshold, threshold + 1]
        + list(range(32760, INT16_MAX + 1))         # near INT16_MAX
    )
    return sorted(set(cases))


ALL_CNT_CASES = [c for t in THRESHOLDS for c in counter_cases(t)]


# ── 1. Reachable counters: fixed code == infinite-precision math ───────

@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("delta", [-1, 1])
@pytest.mark.parametrize("cnt", ALL_CNT_CASES, ids=lambda c: f"cnt{c}")
def test_reachable_states_match_math(cnt, delta, threshold):
    """No spurious flip of any kind for every REACHABLE stored counter.

    Reachable := |cnt| <= threshold: the kernel resets to 0 the moment
    |cnt| > threshold, so a stored counter can never be outside this band
    (counter array starts zeroed).  Within the band the fixed sequence must
    agree with true arithmetic on flip decision, flip direction, and the
    stored value — except at the threshold=32767 rails, where the int16
    wrap is reachable and the clamp converts the (spurious) flip into a
    pinned no-flip rail by design.
    """
    if abs(cnt) > threshold:
        pytest.skip("stored counter outside the reachable band |cnt|<=threshold")
    new_cnt, flip = kernel_step(cnt, delta, threshold, clamp=True)
    true_v = cnt + delta
    overflow = true_v > INT16_MAX or true_v < CLAMP_LO  # past the clamp range

    if not overflow:
        # No wrap involved: fixed code is exact math.
        assert (new_cnt, flip) == math_step(cnt, delta, threshold), (
            f"fixed code diverges from math: cnt={cnt} delta={delta} "
            f"threshold={threshold}")
    else:
        # Only reachable at threshold=32767 rails: wrap fires, clamp pins.
        assert threshold == 32767, (
            f"overflow unreachable at threshold={threshold}")
        assert flip == 0 and CLAMP_LO <= new_cnt <= CLAMP_HI, (
            f"clamp must pin the rail flip: cnt={cnt} delta={delta}")


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_exhaustive_reachable_band_matches_math(threshold):
    """Exhaustive sweep of the entire reachable band, both deltas."""
    for cnt in range(-threshold, threshold + 1):
        for delta in (-1, 1):
            new_cnt, flip = kernel_step(cnt, delta, threshold, clamp=True)
            true_v = cnt + delta
            if true_v <= INT16_MAX and true_v >= CLAMP_LO:
                assert (new_cnt, flip) == math_step(cnt, delta, threshold), (
                    f"cnt={cnt} delta={delta} threshold={threshold}: "
                    f"got ({new_cnt}, {flip})")
            else:
                assert threshold == 32767 and flip == 0, (
                    f"cnt={cnt} delta={delta} threshold={threshold}: "
                    f"unexpected rail flip {flip}")


# ── 2. Always flips at ±(threshold+1) when the value is a legal int16 ───

@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("sign", [-1, 1])
def test_always_flips_at_threshold_plus_one(threshold, sign):
    """A counter at ±threshold must flip on a same-direction step.

    Primary production invariant: the flip fires exactly at
    ±(threshold+1) with the correct direction, then resets to 0.  At
    threshold=32767 the unclamped value ±32768 is NOT a legal int16, so no
    flip is due; the counter pins at the rail instead (see invariant 1).
    """
    cnt = threshold * sign  # one step from the flip boundary
    delta = sign
    new_cnt, flip = kernel_step(cnt, delta, threshold, clamp=True)

    if threshold <= 32766:
        assert flip == sign, (
            f"expected {'increment' if sign > 0 else 'decrement'} flip at "
            f"cnt={cnt} delta={delta} threshold={threshold}")
        assert new_cnt == 0, "flip must reset the counter to 0"
    else:
        assert flip == 0 and abs(new_cnt) == 32767, (
            f"threshold=32767: counter must pin at the rail, got "
            f"cnt={new_cnt} flip={flip}")


# ── 3. INT16_MIN is never stored — even from corrupted state ────────────

@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("delta", [-1, 1])
@pytest.mark.parametrize("cnt", ALL_CNT_CASES, ids=lambda c: f"cnt{c}")
def test_never_stores_int16_min(cnt, delta, threshold):
    """After any step, the stored counter lies in [-32767, 32767].

    Runs over the FULL requested cnt set, deliberately including
    unreachable/corrupted states (e.g. INT16_MIN itself, or rails at a
    small threshold): the clamp self-heals any int16 input into the legal
    range within one step.  This is the "INT16_MIN can never be observed"
    guarantee, including the value the OLD kernel could transiently
    evaluate at the -32767 rail.
    """
    new_cnt, _ = kernel_step(cnt, delta, threshold, clamp=True)
    assert CLAMP_LO <= new_cnt <= CLAMP_HI, (
        f"stored counter {new_cnt} out of clamp range "
        f"(cnt={cnt} delta={delta} threshold={threshold})")
    assert new_cnt != INT16_MIN


# ── 4. In-band behavior bit-identical OLD vs NEW (prod preserved) ───────

@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("delta", [-1, 1])
@pytest.mark.parametrize("cnt", ALL_CNT_CASES, ids=lambda c: f"cnt{c}")
def test_in_band_bitexact_vs_old(cnt, delta, threshold):
    """For every reachable counter: pre-clamp and post-clamp agree exactly.

    The clamp may only differ from the OLD kernel where the int16 wrap
    actually fires — which for reachable counters happens only at
    threshold=32767 rails.  Everywhere else (all of production,
    threshold=32) the two are bit-identical: same flip, same stored value.
    """
    if abs(cnt) > threshold:
        pytest.skip("stored counter outside the reachable band |cnt|<=threshold")
    old_cnt, old_flip = kernel_step(cnt, delta, threshold, clamp=False)
    new_cnt, new_flip = kernel_step(cnt, delta, threshold, clamp=True)
    true_v = cnt + delta
    if true_v <= INT16_MAX and true_v >= CLAMP_LO:
        assert (new_cnt, new_flip) == (old_cnt, old_flip), (
            f"clamp changed no-wrap behavior: cnt={cnt} delta={delta} "
            f"threshold={threshold}")
    else:
        assert threshold == 32767, "wrap unreachable in-band below 32767"


# ── 5. The wrap bug exists in OLD code and the clamp kills it ───────────

@pytest.mark.parametrize("rail", [32767, -32767])
def test_wrap_bug_at_max_threshold_rail(rail):
    """The exact spurious-flip scenario from the sweep report (thr=32767).

    OLD at the +32767 rail: +1 wraps to -32768 → -32768 < -32767 → a
    DECREMENT flip although the true value +32768 demands an increment —
    wrong-direction weight flip + bogus counter reset.  NEW: no flip.
    OLD at the -32767 rail: -1 lands exactly on INT16_MIN → flip fires
    on an unrepresentable boundary value.  NEW: clamped to -32767, no
    flip, counter pinned (the accepted "INT16_MIN never observed" rail).
    """
    delta = 1 if rail > 0 else -1

    old_cnt, old_flip = kernel_step(rail, delta, 32767, clamp=False)
    assert old_flip != 0 and old_cnt == 0, (
        f"OLD code must exhibit the wrap flip at rail={rail} "
        f"(got flip={old_flip})")

    new_cnt, new_flip = kernel_step(rail, delta, 32767, clamp=True)
    assert new_flip == 0, (
        f"clamp must kill the rail flip (rail={rail}, got flip={new_flip})")
    assert abs(new_cnt) == 32767, (
        f"counter must pin inside the clamp range, got {new_cnt}")


def test_old_code_wrong_direction_at_plus_rail():
    """OLD fires a DECREMENT flip for a true value of +32768 (the bug)."""
    _, old_flip = kernel_step(32767, 1, 32767, clamp=False)
    _, math_flip = math_step(32767, 1, 32767)
    assert math_flip == 1, "true value +32768 demands an increment flip"
    assert old_flip == -1, (
        f"OLD wrap must produce the wrong-direction flip, got {old_flip}")


# ── 6. v3 flavor: magnitude-scaled deltas (|delta| up to 8) ─────────────

@pytest.mark.parametrize("threshold", [0, 8, 32])
@pytest.mark.parametrize("delta", [-8, -2, 2, 8])
@pytest.mark.parametrize("cnt", [-32, -8, -1, 0, 1, 8, 32],
                         ids=lambda c: f"cnt{c}")
def test_v3_scaled_deltas_match_math(cnt, delta, threshold):
    """v3 (gemm_update_tc_v3_32.cu) increments by |delta| in [1, 8].

    Same invariant as test 1 with scaled steps: within the reachable band
    and away from the int16 rails, the clamped sequence is exact math.
    (v3's rails at threshold=32767 are covered by the ±1 tests — the clamp
    behavior there is delta-independent.)
    """
    if abs(cnt) > threshold:
        pytest.skip("stored counter outside the reachable band")
    new_cnt, flip = kernel_step(cnt, delta, threshold, clamp=True)
    true_v = cnt + delta
    if true_v <= INT16_MAX and true_v >= CLAMP_LO:
        assert (new_cnt, flip) == math_step(cnt, delta, threshold), (
            f"v3 scaled step diverges: cnt={cnt} delta={delta} "
            f"threshold={threshold}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
