#pragma once

#include <cstdint>
#include <cassert>

// ═════════════════════════════════════════════════════════════════════════════
//  Packed ternary storage — 16 ternary values per uint32_t
//
//  Encoding  │ 00 =  0  │ 01 = +1  │ 10 = -1  │ 11 = INVALID
//
//  alignas(16) is recommended for row pointers to enable 128-bit vectorised
//  loads in GEMM kernels.
// ═════════════════════════════════════════════════════════════════════════════

constexpr int kTernaryBits = 2;
constexpr int kWeightsPerWord = 16;
constexpr uint32_t kCode0  = 0;  //  0
constexpr uint32_t kCodeP1 = 1;  // +1
constexpr uint32_t kCodeM1 = 2;  // -1
constexpr uint32_t kCodeXX = 3;  // INVALID (corruption sentinel)

// ── Storage descriptor ──────────────────────────────────────────────────────

struct PackedTernaryTensor {
    uint32_t* data;        // packed words
    int rows;              // number of rows (e.g. output features)
    int cols;              // number of columns (e.g. input features)
    int stride_words;      // stride in uint32_t words, >= ceil(cols / 16)
};

// ── Branchless 1-hot decode LUT ─────────────────────────────────────────────
//  Compiler turns this into a single vmov/vextract on most ISAs.

__device__ __host__ inline int8_t decode_ternary(uint32_t code_2bit) {
    constexpr int8_t kLUT[4] = {0, 1, -1, 0};
    return kLUT[code_2bit & 3];
}

// ── Pack 16 values into one uint32_t ────────────────────────────────────────

__host__ __device__ inline uint32_t pack16(const int8_t vals[16]) {
    uint32_t word = 0;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        uint32_t code;
        switch (vals[i]) {
            case  0: code = kCode0;  break;
            case  1: code = kCodeP1; break;
            case -1: code = kCodeM1; break;
            default: code = kCodeXX; break;  // clamp to INVALID
        }
        word |= (code << (kTernaryBits * i));
    }
    return word;
}

// ── Decode one word into 16 int8 values ─────────────────────────────────────

__host__ __device__ inline void unpack16(uint32_t word, int8_t out[16]) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        out[i] = decode_ternary(word >> (kTernaryBits * i));
    }
}

// ── Load 16 values from a row at a column offset (returns register array) ───
//  This is the recommended entry point for GEMM kernels — never materialise
//  an FP tensor from packed storage.

__device__ __host__ inline void load16(
    const uint32_t* row_base,
    int col_start,
    int8_t out[16])
{
    int wi = col_start / 16;
    unpack16(row_base[wi], out);
}

// ── Single-weight accessors (for host-side init / debug) ────────────────────

__host__ __device__ inline int8_t get_weight(
    const uint32_t* row, int col)
{
    int wi = col / kWeightsPerWord;
    int shift = (col % kWeightsPerWord) * kTernaryBits;
    return decode_ternary(row[wi] >> shift);
}

__host__ __device__ inline void set_weight(
    uint32_t* row, int col, int8_t val)
{
    int wi = col / kWeightsPerWord;
    int shift = (col % kWeightsPerWord) * kTernaryBits;
    uint32_t code;
    switch (val) {
        case  0: code = kCode0;  break;
        case  1: code = kCodeP1; break;
        case -1: code = kCodeM1; break;
        default: code = kCodeXX; break;
    }
    uint32_t mask = ~(3u << shift);
    row[wi] = (row[wi] & mask) | (code << shift);
}

// ── Discrete state-machine transitions (for fused backward kernel) ─────────

__host__ __device__ inline void increment_weight(uint32_t* row, int col) {
    int8_t w = get_weight(row, col);
    if      (w == -1) set_weight(row, col, 0);
    else if (w ==  0) set_weight(row, col, 1);
    // +1 stays +1 (saturated)
}

__host__ __device__ inline void decrement_weight(uint32_t* row, int col) {
    int8_t w = get_weight(row, col);
    if      (w ==  1) set_weight(row, col, 0);
    else if (w ==  0) set_weight(row, col, -1);
    // -1 stays -1 (saturated)
}

#ifdef __CUDACC__
__device__ inline void increment_weight_atomic(uint32_t* row, int col) {
    int wi = col / kWeightsPerWord;
    int shift = (col % kWeightsPerWord) * kTernaryBits;
    uint32_t* address = row + wi;
    uint32_t old_val = *address;
    uint32_t assumed;
    do {
        assumed = old_val;
        int8_t w = decode_ternary(assumed >> shift);
        int8_t new_w = w;
        if      (w == -1) new_w = 0;
        else if (w ==  0) new_w = 1;
        if (new_w == w) break;

        uint32_t code = (new_w == 1) ? kCodeP1 : ((new_w == -1) ? kCodeM1 : kCode0);
        uint32_t mask = ~(3u << shift);
        uint32_t updated = (assumed & mask) | (code << shift);

        old_val = atomicCAS(address, assumed, updated);
    } while (assumed != old_val);
}

__device__ inline void decrement_weight_atomic(uint32_t* row, int col) {
    int wi = col / kWeightsPerWord;
    int shift = (col % kWeightsPerWord) * kTernaryBits;
    uint32_t* address = row + wi;
    uint32_t old_val = *address;
    uint32_t assumed;
    do {
        assumed = old_val;
        int8_t w = decode_ternary(assumed >> shift);
        int8_t new_w = w;
        if      (w ==  1) new_w = 0;
        else if (w ==  0) new_w = -1;
        if (new_w == w) break;

        uint32_t code = (new_w == 1) ? kCodeP1 : ((new_w == -1) ? kCodeM1 : kCode0);
        uint32_t mask = ~(3u << shift);
        uint32_t updated = (assumed & mask) | (code << shift);

        old_val = atomicCAS(address, assumed, updated);
    } while (assumed != old_val);
}
#endif

// ── Pack a whole row from FP32 source ───────────────────────────────────────

__host__ void pack_row(
    uint32_t* dst_row,          // stride_words words
    const float* src,           // cols elements
    int cols,
    float gamma = 1.0f)        // scale factor (ternarize threshold)
{
    int n_words = (cols + kWeightsPerWord - 1) / kWeightsPerWord;
    for (int w = 0; w < n_words; ++w) {
        int8_t vals[16];
        for (int i = 0; i < 16; ++i) {
            int c = w * 16 + i;
            if (c < cols) {
                float q = src[c] / gamma;
                vals[i] = (int8_t)max(-1, min(1, (int)roundf(q)));
            } else {
                vals[i] = 0;  // padding
            }
        }
        dst_row[w] = pack16(vals);
    }
}

// ── Unpack a whole row to FP32 (for checkpoint / debug only) ────────────────

__host__ void unpack_row(
    const uint32_t* src_row,    // stride_words words
    float* dst,                 // cols elements
    int cols,
    float gamma = 1.0f)
{
    int n_words = (cols + kWeightsPerWord - 1) / kWeightsPerWord;
    for (int w = 0; w < n_words; ++w) {
        int8_t vals[16];
        unpack16(src_row[w], vals);
        for (int i = 0; i < 16; ++i) {
            int c = w * 16 + i;
            if (c < cols) {
                dst[c] = (float)vals[i] * gamma;
            }
        }
    }
}
