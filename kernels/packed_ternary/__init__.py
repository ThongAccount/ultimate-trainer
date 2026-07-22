"""Packed ternary tensor — 16 ternary values per uint32_t.

Primitives for Phase 1 of the discrete-optimisation stack:

    PackedTernaryTensor
    ├── pack / unpack / load16 / get / set
    ├── increment / decrement (state-machine transitions)
    └── pack_row / unpack_row (host helpers)
"""

from __future__ import annotations

import ctypes
import math
import os
from typing import Optional

import numpy as np
import torch

_EXTENSION_LOADED = False
_pack_fn = None   # pack16 host wrapper
_unpack_fn = None  # unpack16 host wrapper

kWeightsPerWord = 16
kTernaryBits = 2

# ═══════════════════════════════════════════════════════════════════════════════
#  Host-side reference (pure Python, no CUDA needed)
# ═══════════════════════════════════════════════════════════════════════════════

LUT = {0b00: 0, 0b01: 1, 0b10: -1, 0b11: 0}  # 11 = INVALID → 0
CODE = {0: 0b00, 1: 0b01, -1: 0b10}


def pack16_reference(vals):
    """Pack 16 int8 values into one uint32."""
    word = 0
    for i, v in enumerate(vals):
        code = CODE.get(v, 0b11)  # clamp to INVALID
        word |= code << (kTernaryBits * i)
    return word


def unpack16_reference(word):
    """Decode one uint32 into 16 int8 values."""
    vals = []
    for i in range(16):
        code = (word >> (kTernaryBits * i)) & 3
        vals.append(LUT[code])
    return vals


# ═══════════════════════════════════════════════════════════════════════════════
#  CUDA extension (load_inline)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_extension():
    global _EXTENSION_LOADED, _pack_fn, _unpack_fn
    if _EXTENSION_LOADED:
        return

    HERE = os.path.dirname(os.path.abspath(__file__))
    CUH_PATH = os.path.join(HERE, "packed_ternary.cuh")

    with open(CUH_PATH) as f:
        cuh_source = f.read()

    cpp_source = """
    #include <cstdint>
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    #include <cstdio>

    namespace py = pybind11;

    """ + cuh_source + """

    // ── Host test: pack 16 values → verify roundtrip ──────────────────
    py::dict test_pack_roundtrip() {
        int8_t input[16];
        int8_t decoded[16];
        int errors = 0;

        // Test all {-1,0,+1} combinations for first 4 positions (3^4 = 81 combos)
        int8_t vals[] = {-1, 0, 1};
        int count = 0;
        for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
        for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d) {
            input[0]  = vals[a]; input[1]  = vals[b];
            input[2]  = vals[c]; input[3]  = vals[d];
            input[4]  = 0; input[5]  = 0; input[6]  = 0; input[7]  = 0;
            input[8]  = 0; input[9]  = 0; input[10] = 0; input[11] = 0;
            input[12] = 0; input[13] = 0; input[14] = 0; input[15] = 0;

            uint32_t word = pack16(input);
            unpack16(word, decoded);

            for (int i = 0; i < 16; ++i) {
                if (input[i] != decoded[i]) {
                    errors++;
                }
            }
            count++;
        }

        py::dict result;
        result["combinations"] = count;
        result["errors"] = errors;
        result["ok"] = (errors == 0);
        return result;
    }

    // ── Host test: increment / decrement state machine ────────────────
    py::dict test_state_machine() {
        uint32_t row[1] = {0};
        int errors = 0;

        // -1 → increment → 0 → increment → +1 → increment → +1 (clamped)
        set_weight(row, 0, -1);
        increment_weight(row, 0);
        if (get_weight(row, 0) != 0) errors++;
        increment_weight(row, 0);
        if (get_weight(row, 0) != 1) errors++;
        increment_weight(row, 0);
        if (get_weight(row, 0) != 1) errors++;  // clamped

        // +1 → decrement → 0 → decrement → -1 → decrement → -1 (clamped)
        set_weight(row, 0, 1);
        decrement_weight(row, 0);
        if (get_weight(row, 0) != 0) errors++;
        decrement_weight(row, 0);
        if (get_weight(row, 0) != -1) errors++;
        decrement_weight(row, 0);
        if (get_weight(row, 0) != -1) errors++;  // clamped

        py::dict result;
        result["errors"] = errors;
        result["ok"] = (errors == 0);
        return result;
    }

    // ── Host test: pack_row / unpack_row roundtrip ────────────────────
    py::dict test_row_roundtrip(int cols) {
        int n_words = (cols + 15) / 16;
        std::vector<float> src(cols);
        std::vector<float> dst(cols);
        std::vector<uint32_t> packed(n_words);

        // Fill with random FP32 values
        for (int i = 0; i < cols; ++i) {
            src[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
        }

        // Ternary quantization with gamma = 1.0
        float gamma = 1.0f;
        for (int i = 0; i < cols; ++i) {
            float q = src[i] / gamma;
            int8_t t = (int8_t)max(-1, min(1, (int)roundf(q)));
            src[i] = (float)t * gamma;  // expected after roundtrip
        }

        pack_row(packed.data(), src.data(), cols, gamma);
        unpack_row(packed.data(), dst.data(), cols, gamma);

        int errors = 0;
        for (int i = 0; i < cols; ++i) {
            if (fabs(src[i] - dst[i]) > 1e-6f) errors++;
        }

        py::dict result;
        result["cols"] = cols;
        result["n_words"] = n_words;
        result["errors"] = errors;
        result["ok"] = (errors == 0);
        return result;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("test_pack_roundtrip", &test_pack_roundtrip,
              "Exhaustive pack/unpack roundtrip for 3^4 combinations");
        m.def("test_state_machine", &test_state_machine,
              "increment / decrement state machine transitions");
        m.def("test_row_roundtrip", &test_row_roundtrip,
              "pack_row / unpack_row roundtrip for given cols");
    }
    """

    from torch.utils.cpp_extension import load_inline

    ext = load_inline(
        name="packed_ternary",
        cpp_sources=[cpp_source],
        functions=["test_pack_roundtrip", "test_state_machine", "test_row_roundtrip"],
        verbose=False,
        extra_cflags=["-O2"],
    )

    _pack_fn = ext
    _EXTENSION_LOADED = True


def get_extension():
    """Lazy-load and return the CUDA extension module."""
    if not _EXTENSION_LOADED:
        _load_extension()
    return _pack_fn


# ═══════════════════════════════════════════════════════════════════════════════
#  Tensor-level helpers (no CUDA required for these)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stride_words(cols: int) -> int:
    """Number of uint32 words needed per row (includes padding)."""
    return (cols + kWeightsPerWord - 1) // kWeightsPerWord


def pack_tensor(fp32_tensor: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Pack an FP32 matrix into a uint32 packed tensor.

    Args:
        fp32_tensor: Shape ``(rows, cols)`` FP32 tensor.
        gamma: Quantisation scale factor.

    Returns:
        uint32 tensor of shape ``(rows, stride_words)``.
    """
    rows, cols = fp32_tensor.shape
    stride = compute_stride_words(cols)
    fp32_np = fp32_tensor.cpu().numpy()
    packed = np.zeros((rows, stride), dtype=np.uint32)

    for r in range(rows):
        row_vals = []
        for c in range(cols):
            q = fp32_np[r, c] / gamma
            t = max(-1, min(1, int(round(q))))
            row_vals.append(t)
        for w in range(stride):
            chunk = row_vals[w * 16:(w + 1) * 16]
            chunk += [0] * (16 - len(chunk))
            packed[r, w] = pack16_reference(chunk)

    # Must be int32 so the CUDA kernel can read stride_words uint32 entries
    # per row without pointer-arithmetic mismatch (int64 → uint32 halves data).
    return torch.from_numpy(packed.astype(np.int32))


def unpack_tensor(packed: torch.Tensor, rows: int, cols: int, gamma: float = 1.0) -> torch.Tensor:
    """Unpack a ``(rows, stride_words)`` uint32 tensor back to FP32.

    This is for checkpoint / debugging only — never use in the forward path.
    """
    stride = packed.shape[1]
    fp32 = torch.zeros(rows, cols, dtype=torch.float32)
    packed_np = packed.cpu().numpy().astype(np.uint32, copy=False)

    for r in range(rows):
        for w in range(stride):
            word = int(packed_np[r, w])
            vals = unpack16_reference(word)
            for i, v in enumerate(vals):
                c = w * 16 + i
                if c < cols:
                    fp32[r, c] = float(v) * gamma

    return fp32


def count_parameters(packed: torch.Tensor) -> int:
    """Number of parameters stored in a packed tensor."""
    rows = packed.shape[0]
    stride = packed.shape[1]
    return rows * stride * kWeightsPerWord
