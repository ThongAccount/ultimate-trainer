/**
 * gemm_backward_dx_tc.cu — Tensor-Core backward dX via WMMA (4 warps).
 *
 * Computes dX = dY @ W  (gradient w.r.t. input).
 * Same GEMM as forward but the weight matrix is on the right.
 *
 * Uses wmma::mma_sync(m=16, n=16, k=16) on T4 Tensor Cores.
 * Each block processes a 32×32 super-tile with 4 warps, each warp
 * handling one 16×16 tile.  This improves occupancy vs the 1-warp version.
 *
 * Grid:   (ceil(batch/32), ceil(in_features/32))
 * Block:  128 threads (4 warps)
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
constexpr int kWarpsPerBlock = 4;
constexpr int kSuperM = 32;  // super-tile batch (2 × kM)
constexpr int kSuperN = 32;  // super-tile in   (2 × kN)

// ── Per-warp SMEM offsets ─────────────────────────────────────────────
#define DYS(w, b, r)   dY_smem[(w) * kM * kK + (b) * kK + (r)]
#define WS(w, r, c)    W_smem[(w) * kK * kN + (r) * kN + (c)]
#define DXF(w, b, c)   dX_float_smem[(w) * kM * kN + (b) * kN + (c)]
#define DXH(w, b, c)   dX_smem[(w) * kM * kN + (b) * kN + (c)]

__global__ __launch_bounds__(128) void packed_ternary_backward_dx_tc_kernel(
    const uint32_t* __restrict__ W,
    const half*     __restrict__ dY,
    half*           __restrict__ dX,
    int batch_size,
    int in_features,
    int out_features,
    int stride_words)
{
    int super_b0 = blockIdx.x * kSuperM;   // super-tile batch offset
    int super_c0 = blockIdx.y * kSuperN;   // super-tile in offset
    int warp_id = threadIdx.x / 32;        // 0..3
    int wtid    = threadIdx.x % 32;        // 0..31 (within-warp)

    // Each warp handles one 16×16 output tile within the 32×32 super-tile.
    // warp 0 → (b=0, c=0), warp 1 → (b=0, c=16),
    // warp 2 → (b=16, c=0), warp 3 → (b=16, c=16)
    int warp_b_off = (warp_id / 2) * kM;   // 0 or 16
    int warp_c_off = (warp_id % 2) * kN;   // 0 or 16

    int b0 = super_b0 + warp_b_off;
    int c0 = super_c0 + warp_c_off;

    // ── Shared memory (4 warps × independent tiles) ──────────────────
    __shared__ half   dY_smem[kWarpsPerBlock * kM * kK];
    __shared__ half   W_smem[kWarpsPerBlock * kK * kN];
    __shared__ float  dX_float_smem[kWarpsPerBlock * kM * kN];
    __shared__ half   dX_smem[kWarpsPerBlock * kM * kN];

    // ── WMMA fragments ──────────────────────────────────────────────
    wmma::fragment<wmma::matrix_a, kM, kN, kK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kM, kN, kK, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, kM, kN, kK, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // ── Outer loop over R (out_features) tiles ──────────────────────
    for (int r0 = 0; r0 < out_features; r0 += kK) {
        int tile_r = min(kK, out_features - r0);

        // ── Load dY tile → dY_smem (strided fill) ───────────────────
        #pragma unroll
        for (int i = wtid; i < kM * kK; i += 32) {
            int b = i / kK;               // 0..15
            int r = i % kK;               // 0..15
            half val = __float2half(0.0f);
            if (b < kM && r < tile_r) {
                int gb = b0 + b;
                int gr = r0 + r;
                if (gb < batch_size && gr < out_features) {
                    val = dY[gb * out_features + gr];
                }
            }
            DYS(warp_id, b, r) = val;
        }

        // ── Load W tile → unpack to FP16 (strided fill) ─────────────
        #pragma unroll
        for (int i = wtid; i < kK * kN; i += 32) {
            int r = i / kN;               // 0..15, out row idx in tile
            int c = i % kN;               // 0..15, in col idx in tile
            half w_val = __float2half(0.0f);
            if (r < tile_r && c < kN) {
                int gr = r0 + r;           // global out row
                int gc = c0 + c;           // global in col
                if (gr < out_features && gc < in_features) {
                    int wi = gc / kWeightsPerWord;
                    if (wi < stride_words) {
                        uint32_t word = W[gr * stride_words + wi];
                        int pos = gc % kWeightsPerWord;
                        int8_t t = decode_ternary(word >> (kTernaryBits * pos));
                        w_val = __float2half((float)t);
                    }
                }
            }
            WS(warp_id, r, c) = w_val;
        }
        __syncthreads();

        // ── Load WMMA fragments from SMEM ───────────────────────────
        wmma::load_matrix_sync(a_frag, &dY_smem[warp_id * kM * kK], kK);
        wmma::load_matrix_sync(b_frag, &W_smem[warp_id * kK * kN], kN);

        // ── Tensor-core matmul on 16×16×16 tile ─────────────────────
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // ── Store accumulator to shared, then to global dX ──────────────
    wmma::store_matrix_sync(&dX_float_smem[warp_id * kM * kN], c_frag, kN,
                            wmma::mem_row_major);
    __syncthreads();

    // 128 threads convert 1024 float→half elements (8 per thread)
    for (int i = threadIdx.x; i < kWarpsPerBlock * kM * kN; i += blockDim.x) {
        ((half*)dX_smem)[i] = __float2half(((float*)dX_float_smem)[i]);
    }
    __syncthreads();

    // Write dX_smem to global dX (batch × in_features)
    for (int i = threadIdx.x; i < kWarpsPerBlock * kM * kN; i += blockDim.x) {
        int w = i / (kM * kN);               // which warp's tile
        int local = i % (kM * kN);           // (b,c) within the 16×16 tile
        int b = local / kN;
        int c = local % kN;

        int warp_b_off_w = (w / 2) * kM;
        int warp_c_off_w = (w % 2) * kN;

        int gb = super_b0 + warp_b_off_w + b;
        int gc = super_c0 + warp_c_off_w + c;
        if (gb < batch_size && gc < in_features) {
            dX[gb * in_features + gc] = DXH(w, b, c);
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

    dim3 grid((batch_size + kSuperM - 1) / kSuperM,
              (in_features + kSuperN - 1) / kSuperN);
    dim3 block(128);  // 4 warps

    packed_ternary_backward_dx_tc_kernel<<<grid, block, 0, stream>>>(
        W, dY, dX, batch_size, in_features, out_features, stride_words
    );
}
