/**
 * gemm_backward_dx_tc.cu — Tensor-Core backward dX via WMMA.
 *
 * Computes dX = dY @ W  (gradient w.r.t. input).
 * Same GEMM as forward but the weight matrix is on the right.
 *
 * Uses wmma::mma_sync(m=16, n=16, k=16) on T4 Tensor Cores.
 * Packed ternary weights are unpacked into __shared__ FP16 tiles.
 *
 * Grid:   (ceil(batch/16), ceil(in_features/16))
 * Block:  256 threads
 *
 * dY (FP16)  × W (packed ternary) → dX (FP16)
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "packed_ternary.cuh"
#include <mma.h>

namespace wmma = nvcuda::wmma;

constexpr int kM = 16;   // WMMA tile: batch
constexpr int kN = 16;   // WMMA tile: in_features
constexpr int kK = 16;   // WMMA tile: out_features (reduction dim)

__global__ void packed_ternary_backward_dx_tc_kernel(
    const uint32_t* __restrict__ W,
    const half*     __restrict__ dY,
    half*           __restrict__ dX,
    int batch_size,
    int in_features,
    int out_features,
    int stride_words)
{
    int b0 = blockIdx.x * kM;     // batch offset for this tile
    int c0 = blockIdx.y * kN;     // in-feature offset for this tile
    int tid = threadIdx.x;        // 0..255

    // ── Shared memory ─────────────────────────────────────────────────
    __shared__ half   dY_smem[kM][kK];     // dY tile (row-major: batch × out)
    __shared__ half   W_smem[kK][kN];      // W tile (row-major: out × in)
    __shared__ float  dX_float_smem[kM][kN]; // output tile (WMMA writes to SMEM)
    __shared__ half   dX_smem[kM][kN];

    // ── WMMA fragments ────────────────────────────────────────────────
    wmma::fragment<wmma::matrix_a, kM, kN, kK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kM, kN, kK, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, kM, kN, kK, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // ── Outer loop over R (out_features) tiles ────────────────────────
    for (int r0 = 0; r0 < out_features; r0 += kK) {
        int tile_r = min(kK, out_features - r0);
        int tile_r_words = (tile_r + kWeightsPerWord - 1) / kWeightsPerWord;

        // ── Load dY tile → dY_smem ────────────────────────────────────
        {
            int b = tid / kK;               // 0..15, batch idx in tile
            int r = tid % kK;               // 0..15, out row idx in tile
            half val = __float2half(0.0f);
            if (b < kM && r < tile_r) {
                int gb = b0 + b;
                int gr = r0 + r;
                if (gb < batch_size && gr < out_features) {
                    val = dY[gb * out_features + gr];
                }
            }
            dY_smem[b][r] = val;
        }

        // ── Load W tile → unpack to FP16 → W_smem ────────────────────
        {
            int r = tid / kN;               // 0..15, out row idx in tile
            int c = tid % kN;               // 0..15, in col idx in tile
            half w_val = __float2half(0.0f);
            if (r < tile_r && c < kN) {
                int gr = r0 + r;            // global out row
                int gc = c0 + c;            // global in col
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
        }
        __syncthreads();

        // ── Load WMMA fragments from SMEM ─────────────────────────────
        wmma::load_matrix_sync(a_frag, &dY_smem[0][0], kK);
        wmma::load_matrix_sync(b_frag, &W_smem[0][0], kN);

        // ── Tensor-core matmul on 16×16×16 tile ──────────────────────
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // ── Store accumulator to shared, then to global dX ────────────────
    // store_matrix_sync(row_major, stride=kN): c_frag[b][c] → dX_float_smem[b][c]
    wmma::store_matrix_sync(&dX_float_smem[0][0], c_frag, kN, wmma::mem_row_major);
    __syncthreads();
    if (tid < kM * kN) {
        ((half*)dX_smem)[tid] = __float2half(((float*)dX_float_smem)[tid]);
    }
    __syncthreads();

    // ── Write dX_smem to global dX (batch × in_features) ────────────
    {
        int b = tid / kN;               // 0..15
        int c = tid % kN;               // 0..15
        int gb = b0 + b;
        int gc = c0 + c;
        if (gb < batch_size && gc < in_features) {
            dX[gb * in_features + gc] = dX_smem[b][c];
        }
    }
}

extern "C" void launch_packed_ternary_backward_dx_tc(
    const uint32_t* W,
    const void*     dY_ptr,
    void*           dX_ptr,
    int batch_size,
    int in_features,
    int out_features,
    int stride_words,
    cudaStream_t stream)
{
    const half* dY = static_cast<const half*>(dY_ptr);
    half*       dX = static_cast<half*>(dX_ptr);

    dim3 grid((batch_size + kM - 1) / kM,
              (in_features + kN - 1) / kN);
    dim3 block(256);

    packed_ternary_backward_dx_tc_kernel<<<grid, block, 0, stream>>>(
        W, dY, dX, batch_size, in_features, out_features, stride_words
    );
}
