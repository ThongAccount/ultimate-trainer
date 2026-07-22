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
#include <mma.h>

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
    __shared__ half   W_smem[kN][kK];     // W tile unpacked (row-major)
    __shared__ half   X_smem[kM][kK];     // X tile (row-major: batch × k)
    __shared__ float  Y_float_smem[kN][kM]; // output tile (WMMA must write to SMEM)
    __shared__ half   Y_smem[kN][kM];

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

        // ── Load X tile → X_smem ────────────────────────────────────────
        // Each of 256 threads loads 1 element from X into X_smem.
        {
            int xb = tid / kK;               // batch offset in tile
            int xk = tid % kK;               // k offset in tile
            half x_val = __float2half(0.0f);
            if (xb < kM && xk < tile_k) {
                int gb = b0 + xb;
                int gk = k0 + xk;
                if (gb < batch_size && gk < in_features) {
                    x_val = X[gb * in_features + gk];
                }
            }
            X_smem[xb][xk] = x_val;
        }
        __syncthreads();

        // ── Load WMMA fragments from SMEM ───────────────────────────
        // A fragment: W tile row_major
        wmma::load_matrix_sync(a_frag, &W_smem[0][0], kK);
        // B fragment: X tile col_major.  X_smem stores X[b][k] row-major,
        // col_major load with stride=kM gives B(k, b) = X_smem[b][k] = X[b][k].
        wmma::load_matrix_sync(b_frag, &X_smem[0][0], kM);

        // ── Tensor-core matmul on 16×16×16 tile ─────────────────────
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // ── Store accumulator to shared, then to global Y ────────────────
    // store_matrix_sync(row_major, stride=kM): c_frag(row, batch) → Y_smem[row][batch]
    // Store float accumulator → float SMEM (requires SMEM on Turing),
    // then convert to half.  __syncthreads() before any thread reads.
    wmma::store_matrix_sync(&Y_float_smem[0][0], c_frag, kM, wmma::mem_row_major);
    __syncthreads();
    if (tid < kM * kN) {
        ((half*)Y_smem)[tid] = __float2half(((float*)Y_float_smem)[tid]);
    }
    __syncthreads();

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
