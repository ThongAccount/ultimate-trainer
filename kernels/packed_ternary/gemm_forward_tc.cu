/**
 * gemm_forward_tc.cu — Tensor-Core packed ternary × FP16 forward GEMM.
 *
 * Uses wmma::mma_sync(m=16, n=16, k=16) on T4 Tensor Cores.
 * Packed ternary weights unpacked into __shared__ FP16 tiles.
 *
 * Grid:   (ceil(batch/16), ceil(out_features/16))
 * Block:  256 threads
 *
 * W (packed uint32_t) × X (FP16) → Y (FP16)
 *
 * Reference: the analysis at projects/ultimate-ai-model/ternary-gemm-t4-bottleneck.md
 * identifies the v2 kernel as instruction-issue limited (5.5 inst/FLOP → ~22 GFLOPS).
 * WMMA amortises decode across 256 FMAs per tile, moving bottleneck to memory BW.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "packed_ternary.cuh"
#include <nvcuda/wmma.h>

namespace wmma = nvcuda::wmma;

constexpr int kM = 16;   // WMMA tile sizes
constexpr int kN = 16;
constexpr int kK = 16;

// ═════════════════════════════════════════════════════════════════════════════
//  WMMA kernel
// ═════════════════════════════════════════════════════════════════════════════

__global__ void packed_ternary_tc_kernel(
    const uint32_t* __restrict__ W,
    const half*     __restrict__ X,
    half*           __restrict__ Y,
    int batch_size,
    int in_features,
    int out_features,
    int stride_words)
{
    int b0 = blockIdx.x * kM;     // batch offset for this tile
    int r0 = blockIdx.y * kN;     // output-row offset for this tile
    int tid = threadIdx.x;        // 0..255

    // ── Shared memory ─────────────────────────────────────────────────
    // W tile unpacked to FP16 (row-major): 16 rows × 16 cols
    __shared__ half W_smem[kN][kK];

    // Temporary output tile (row-major: 16 rows × 16 cols)
    __shared__ half Y_smem[kN][kM];

    // ── WMMA fragments ───────────────────────────────────────────────
    wmma::fragment<wmma::matrix_a, kM, kN, kK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kM, kN, kK, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, kM, kN, kK, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // ── Outer loop over K tiles ──────────────────────────────────────
    for (int k0 = 0; k0 < in_features; k0 += kK) {
        int tile_k = min(kK, in_features - k0);
        int tile_k_words = (tile_k + kWeightsPerWord - 1) / kWeightsPerWord;

        // ── Load W tile → unpack to FP16 → W_smem ────────────────────
        // Each of 256 threads loads exactly 1 element of the 16×16 tile.
        int r = tid / kK;                        // 0..15, output row in tile
        int c = tid % kK;                        // 0..15, input col in tile

        half w_val = __float2half(0.0f);
        if (r < kN && c < tile_k) {
            int gr = r0 + r;                     // global output row
            int gc = k0 + c;                     // global input col
            if (gr < out_features && gc < in_features) {
                int wi = gc / kWeightsPerWord;   // which packed word
                if (wi < stride_words) {
                    uint32_t word = W[gr * stride_words + wi];
                    int pos = gc % kWeightsPerWord;
                    int8_t t = decode_ternary(word >> (kTernaryBits * pos));
                    w_val = __float2half((float)t);
                }
            }
        }
        W_smem[r][c] = w_val;
        __syncthreads();

        // ── Load WMMA fragments from SMEM / global ───────────────────
        // A fragment: W tile row-major
        wmma::load_matrix_sync(a_frag, &W_smem[0][0], kK);

        // B fragment: X tile col_major loaded directly from global.
        // X layout in global: X[batch][in_features] row-major.
        // We want B(k, b) = X[b][k].  col_major load with stride=in_features
        // gives: B(k, b) = ptr[b * stride + k] = X[b * in_features + k] = X[b][k].
        wmma::load_matrix_sync(b_frag, &X[b0 * in_features + k0], in_features);

        // ── Tensor-core matmul on 16×16×16 tile ─────────────────────
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // ── Store accumulator to shared, then to global Y ────────────────
    // store_matrix_sync(row_major, stride=kM): c_frag(row, batch) → Y_smem[row][batch]
    wmma::store_matrix_sync(&Y_smem[0][0], c_frag, kM, wmma::mem_row_major);

    // ── Write Y_smem to global Y (transposing row,batch → batch,row) ─
    int r = tid / kM;               // 0..15
    int b = tid % kM;               // 0..15
    int gr = r0 + r;
    int gb = b0 + b;
    if (gr < out_features && gb < batch_size) {
        Y[gb * out_features + gr] = Y_smem[r][b];
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Host launch wrapper
// ═════════════════════════════════════════════════════════════════════════════

extern "C" void launch_packed_ternary_tc(
    const uint32_t* W,
    const void*     X_ptr,
    void*           Y_ptr,
    int batch_size,
    int in_features,
    int out_features,
    int stride_words,
    cudaStream_t stream)
{
    const half* X = static_cast<const half*>(X_ptr);
    half*       Y = static_cast<half*>(Y_ptr);

    dim3 grid((batch_size + kM - 1) / kM,
              (out_features + kN - 1) / kN);
    dim3 block(256);

    packed_ternary_tc_kernel<<<grid, block, 0, stream>>>(
        W, X, Y, batch_size, in_features, out_features, stride_words
    );
}
