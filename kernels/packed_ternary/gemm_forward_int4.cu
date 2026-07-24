/**
 * gemm_forward_int4.cu — INT4 WMMA forward kernel (8×8×32 tiles).
 *
 * EXPERIMENTAL: Uses nvcuda::wmma::experimental::precision::s4 for 4-bit
 * signed integer Tensor Core operations.  Packed ternary weights {-1,0,+1}
 * are unpacked and stored as INT4 values in shared memory.
 *
 * Tile: 8×8×32 (M×N×K) — double the inner dimension of FP16 16×16×16.
 * This halves the number of K-loop trips, and the 4× more blocks (8×8 vs
 * 16×16 output) improve occupancy on small-to-moderate layers.
 *
 * Grid:   (ceil(batch/8), ceil(out_features/8))
 * Block:  64 threads (2 warps) — each warp handles one 8×8 tile
 *          (or 128 threads with 2 tiles per warp; TBD)
 *
 * W (packed uint32_t) × X (FP16) → Y (FP16), accumulated in FP32.
 *
 * Reference: CUDA Toolkit mma.h experimental sub-byte WMMA
 *   https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wmma-sub-byte
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "packed_ternary.cuh"
#include <mma.h>

namespace wmma = nvcuda::wmma;
namespace exp  = nvcuda::wmma::experimental;

constexpr int kM = 8;    // WMMA tile: batch
constexpr int kN = 8;    // WMMA tile: out_features
constexpr int kK = 32;   // WMMA tile: in_features (reduction dim)

// Each s4 element is 4 bits. 2 per byte.
constexpr int kS4PerWord = 8;  // 8 s4 values per uint32 (4 bits each)

/// Pack one ternary value (-1, 0, +1) to s4 (4-bit signed).
__device__ __forceinline__ uint8_t ternary_to_s4(int8_t t) {
    // s4 range is -8..7.  -1 → 0xF, 0 → 0x0, +1 → 0x1.
    return (uint8_t)(t & 0xF);
}

/// Pack two s4 values into one byte (lo in lower nibble, hi in upper).
__device__ __forceinline__ uint8_t pack_s4_pair(int8_t lo, int8_t hi) {
    return (uint8_t)((lo & 0xF) | ((hi & 0xF) << 4));
}

__global__ void packed_ternary_forward_int4_kernel(
    const uint32_t* __restrict__ W,    // (out_features, stride_words) packed ternary
    const half*     __restrict__ X,    // (batch, in_features)
    half*           __restrict__ Y,    // (batch, out_features)
    int batch_size,
    int in_features,
    int out_features,
    int stride_words)
{
    int b0 = blockIdx.x * kM;           // batch offset for this tile (8)
    int r0 = blockIdx.y * kN;           // out-feature row offset (8)
    int tid = threadIdx.x;              // 0..63 or 0..127

    // ── Shared memory ─────────────────────────────────────────────────
    // W tile: 8×32 s4 = 8 × 16 bytes = 128 bytes, aligned
    __shared__ uint8_t W_smem[kN][kK / 2];  // 8 rows × 16 bytes/row
    // X tile: 8×32 FP16 (loaded from global as-is) = 512 bytes
    __shared__ half   X_smem[kM][kK];       // 8 × 32 half
    // Accumulator (8×8 float) + output (8×8 half)
    __shared__ float  Y_float_smem[kN][kM]; // 8×8 float = 256 bytes
    __shared__ half   Y_smem[kN][kM];       // 8×8 half = 128 bytes

    // ── WMMA fragments (s4 × s4 → float accumulator) ─────────────────
    // matrix_a: row_major, M×K = 8×32, s4 elements
    exp::fragment<exp::matrix_a, kM, kN, kK, exp::precision::s4, exp::row_major> a_frag;
    // matrix_b: col_major, K×N = 32×8, s4 elements
    exp::fragment<exp::matrix_b, kM, kN, kK, exp::precision::s4, exp::col_major> b_frag;
    // accumulator: M×N = 8×8, float
    exp::fragment<exp::accumulator, kM, kN, kK, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // ── Outer loop over K (in_features) tiles ─────────────────────────
    for (int k0 = 0; k0 < in_features; k0 += kK) {
        int tile_k = min(kK, in_features - k0);

        // ── Load W tile → unpack ternary → pack as s4 in W_smem ─────
        // W_smem layout: rows [r][byte], each byte holds 2 s4 values.
        // With 64 threads: 64 × (8×32/2)/64 = 64 × 4 = 256 bytes to fill.
        // Each thread handles 4 bytes.
        #pragma unroll
        for (int i = tid; i < kN * (kK / 2); i += blockDim.x) {
            int r = i / (kK / 2);          // row 0..7
            int byte_in_row = i % (kK / 2); // byte 0..15

            int pos_lo = byte_in_row * 2;
            int pos_hi = byte_in_row * 2 + 1;

            int gr = r0 + r;
            if (gr < out_features) {
                uint8_t lo = 0, hi = 0;

                if (pos_lo < tile_k) {
                    int gc = k0 + pos_lo;
                    if (gc < in_features) {
                        int wi = gc / kWeightsPerWord;
                        if (wi < stride_words) {
                            uint32_t word = W[gr * stride_words + wi];
                            lo = ternary_to_s4(decode_ternary(
                                word >> (kTernaryBits * (gc % kWeightsPerWord))));
                        }
                    }
                }
                if (pos_hi < tile_k) {
                    int gc = k0 + pos_hi;
                    if (gc < in_features) {
                        int wi = gc / kWeightsPerWord;
                        if (wi < stride_words) {
                            uint32_t word = W[gr * stride_words + wi];
                            hi = ternary_to_s4(decode_ternary(
                                word >> (kTernaryBits * (gc % kWeightsPerWord))));
                        }
                    }
                }

                W_smem[r][byte_in_row] = (lo & 0xF) | ((hi & 0xF) << 4);
            } else {
                W_smem[r][byte_in_row] = 0;
            }
        }

        // ── Load X tile → X_smem (FP16, stays FP16) ──────────────────
        #pragma unroll
        for (int i = tid; i < kM * kK; i += blockDim.x) {
            int b = i / kK;               // 0..7
            int k = i % kK;               // 0..31
            half val = __float2half(0.0f);
            if (k < tile_k) {
                int gb = b0 + b;
                int gk = k0 + k;
                if (gb < batch_size && gk < in_features) {
                    val = X[gb * in_features + gk];
                }
            }
            X_smem[b][k] = val;
        }
        __syncthreads();

        // ── Load WMMA fragments from SMEM ────────────────────────────
        // For s4 matrix_a row_major: stride is bytes per row = 16
        exp::load_matrix_sync(a_frag, &W_smem[0][0], kK / 2);  // stride = 16 bytes
        // For FP16 matrix_b col_major (X stays FP16, not s4):
        // Actually, we need matrix_b to also be s4 for the s4×s4 WMMA.
        // But X is FP16 — we need to convert or use mixed precision.
        // ...
        // Hmm, s4×s4 WMMA expects both operands in s4 format.
        // X is FP16 activations — converting to INT4 would lose precision.
        // This needs a different approach...

        // ── Tensor-core matmul on 8×8×32 tile ────────────────────────
        // wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }
}
