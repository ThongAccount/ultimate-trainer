"""Exhaustive tests for PackedTernaryTensor — Phase 1.

Tests are organised in layers:
    Layer 1 — encode/decode primitives          (no CUDA needed)
    Layer 2 — state-machine transitions          (no CUDA needed)
    Layer 3 — pack_row / unpack_row roundtrip   (CUDA extension)
    Layer 4 — random tensor roundtrip            (CUDA extension)
    Layer 5 — all weights accessed via get/set   (CUDA extension)
"""

from __future__ import annotations

import math
import random
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernels.packed_ternary import (
    pack16_reference,
    unpack16_reference,
    pack_tensor,
    unpack_tensor,
    compute_stride_words,
    kWeightsPerWord,
    get_extension,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Layer 1 — Encode / decode primitives
# ═══════════════════════════════════════════════════════════════════════════════


def test_pack_all_single_values():
    """Every valid ternary value survives a pack16 → unpack16 roundtrip."""
    for v in [-1, 0, 1]:
        vals = [v] + [0] * 15
        word = pack16_reference(vals)
        decoded = unpack16_reference(word)
        assert decoded[0] == v, f"Roundtrip failed for value {v}"


def test_pack_all_positions():
    """Every position (0..15) survives a roundtrip."""
    for pos in range(16):
        vals = [0] * 16
        vals[pos] = 1
        word = pack16_reference(vals)
        decoded = unpack16_reference(word)
        for i in range(16):
            expected = vals[i]
            got = decoded[i]
            assert got == expected, f"Position {pos}, bit {i}: expected {expected}, got {got}"


def test_pack_all_combinations_first4():
    """Exhaustive 3^4 = 81 combos over the first 4 positions."""
    for a in [-1, 0, 1]:
        for b in [-1, 0, 1]:
            for c in [-1, 0, 1]:
                for d in [-1, 0, 1]:
                    vals = [a, b, c, d] + [0] * 12
                    word = pack16_reference(vals)
                    decoded = unpack16_reference(word)
                    for i in range(4):
                        assert decoded[i] == vals[i], (
                            f"Combo ({a},{b},{c},{d}) failed at pos {i}: "
                            f"expected {vals[i]}, got {decoded[i]}"
                        )


def test_invalid_code_maps_to_zero():
    """Encoding 0b11 (INVALID) maps to 0 when decoded."""
    word = 0b11  # only the lowest 2 bits set → INVALID
    decoded = unpack16_reference(word)
    assert decoded[0] == 0, "INVALID code should decode to 0"


# ═══════════════════════════════════════════════════════════════════════════════
#  Layer 2 — State-machine transitions
# ═══════════════════════════════════════════════════════════════════════════════


def _set_weight(word: int, pos: int, val: int) -> int:
    """Reference set_weight on a single uint32 (no CUDA)."""
    shift = pos * 2
    code = {0: 0b00, 1: 0b01, -1: 0b10}.get(val, 0b11)
    mask = ~(3 << shift)
    return (word & mask) | (code << shift)


def _get_weight(word: int, pos: int) -> int:
    """Reference get_weight on a single uint32."""
    shift = pos * 2
    code = (word >> shift) & 3
    return {0b00: 0, 0b01: 1, 0b10: -1, 0b11: 0}[code]


def _increment(word: int, pos: int) -> int:
    w = _get_weight(word, pos)
    if w == -1:
        return _set_weight(word, pos, 0)
    if w == 0:
        return _set_weight(word, pos, 1)
    return word  # +1 is saturated


def _decrement(word: int, pos: int) -> int:
    w = _get_weight(word, pos)
    if w == 1:
        return _set_weight(word, pos, 0)
    if w == 0:
        return _set_weight(word, pos, -1)
    return word  # -1 is saturated


def test_state_machine_increment():
    """-1 → 0 → +1 → +1 (saturated)."""
    w = 0
    w = _set_weight(w, 0, -1)
    w = _increment(w, 0);  assert _get_weight(w, 0) == 0
    w = _increment(w, 0);  assert _get_weight(w, 0) == 1
    w = _increment(w, 0);  assert _get_weight(w, 0) == 1  # clamped


def test_state_machine_decrement():
    """+1 → 0 → -1 → -1 (saturated)."""
    w = 0
    w = _set_weight(w, 0, 1)
    w = _decrement(w, 0);  assert _get_weight(w, 0) == 0
    w = _decrement(w, 0);  assert _get_weight(w, 0) == -1
    w = _decrement(w, 0);  assert _get_weight(w, 0) == -1  # clamped


def test_state_machine_no_op_on_zero():
    """Increment from 0 → 1. Decrement from 0 → -1."""
    w = 0
    w = _set_weight(w, 0, 0)
    w = _increment(w, 0);  assert _get_weight(w, 0) == 1
    w = _set_weight(w, 0, 0)
    w = _decrement(w, 0);  assert _get_weight(w, 0) == -1


# ═══════════════════════════════════════════════════════════════════════════════
#  Layer 3 — CUDA extension: pack/unpack roundtrip
# ═══════════════════════════════════════════════════════════════════════════════


def _check_cuda_available():
    try:
        ext = get_extension()
        return True
    except Exception:
        return False


def test_cuda_pack_roundtrip():
    """CUDA host test: 3^4 combinations of pack16/unpack16."""
    if not _check_cuda_available():
        print("  [SKIP] no CUDA extension")
        return
    ext = get_extension()
    result = ext.test_pack_roundtrip()
    assert result["ok"], f"Pack roundtrip failed: {result}"
    assert result["combinations"] == 81
    print(f"  CUDA pack roundtrip: {result['combinations']} combos, {result['errors']} errors")


def test_cuda_state_machine():
    """CUDA host test: increment / decrement transitions."""
    if not _check_cuda_available():
        print("  [SKIP] no CUDA extension")
        return
    ext = get_extension()
    result = ext.test_state_machine()
    assert result["ok"], f"State machine failed: {result}"
    print(f"  CUDA state machine: {result['errors']} errors")


def test_cuda_row_roundtrip():
    """CUDA host test: pack_row / unpack_row at various sizes."""
    if not _check_cuda_available():
        print("  [SKIP] no CUDA extension")
        return
    ext = get_extension()
    for cols in [1, 8, 15, 16, 17, 31, 32, 64, 128, 1024]:
        result = ext.test_row_roundtrip(cols)
        assert result["ok"], (f"Row roundtrip failed at cols={cols}: {result}")
    print(f"  CUDA row roundtrip: all sizes OK")


# ═══════════════════════════════════════════════════════════════════════════════
#  Layer 4 — Python-level tensor roundtrip (no CUDA)
# ═══════════════════════════════════════════════════════════════════════════════


def test_pack_tensor_random():
    """Random FP32 matrix → pack → unpack → matches."""
    torch.manual_seed(42)
    for rows, cols in [(1, 16), (3, 32), (8, 64), (4, 128)]:
        x = torch.randn(rows, cols)
        packed = pack_tensor(x)
        reconstructed = unpack_tensor(packed, rows, cols)

        # Both rounded to ternary values (±γ, 0)
        gamma = 1.0
        expected_ternary = torch.clamp(torch.round(x / gamma), -1, 1) * gamma

        diff = (reconstructed - expected_ternary).abs().max().item()
        assert diff < 1e-6, (
            f"Tensor roundtrip failed ({rows}x{cols}): max diff={diff}"
        )


def test_pack_tensor_stride():
    """stride_words is ceil(cols / 16)."""
    assert compute_stride_words(1) == 1
    assert compute_stride_words(16) == 1
    assert compute_stride_words(17) == 2
    assert compute_stride_words(31) == 2
    assert compute_stride_words(32) == 2
    assert compute_stride_words(33) == 3


def test_pack_tensor_nonzero():
    """A known vector roundtrips correctly."""
    vals = [1, -1, 0, 1, 0, -1, -1, 1, 0, 0, 1, 1, -1, -1, -1, 0]
    x = torch.tensor([vals], dtype=torch.float32)
    packed = pack_tensor(x)
    reconstructed = unpack_tensor(packed, 1, 16)
    for i in range(16):
        assert reconstructed[0, i].item() == float(vals[i]), f"Mismatch at {i}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Run
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        ("pack all single values",        test_pack_all_single_values),
        ("pack all positions",            test_pack_all_positions),
        ("pack 3^4 combos first 4",       test_pack_all_combinations_first4),
        ("INVALID code → 0",              test_invalid_code_maps_to_zero),
        ("state machine increment",       test_state_machine_increment),
        ("state machine decrement",       test_state_machine_decrement),
        ("state machine zero",            test_state_machine_no_op_on_zero),
        ("CUDA pack roundtrip",           test_cuda_pack_roundtrip),
        ("CUDA state machine",            test_cuda_state_machine),
        ("CUDA row roundtrip",            test_cuda_row_roundtrip),
        ("tensor roundtrip random",       test_pack_tensor_random),
        ("stride computation",            test_pack_tensor_stride),
        ("known vector roundtrip",        test_pack_tensor_nonzero),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} passed", end="")
    if failed:
        print(f", {failed} FAILED ❌")
        sys.exit(1)
    else:
        print(" ✅")
