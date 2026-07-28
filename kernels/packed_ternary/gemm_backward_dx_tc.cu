/**
 * gemm_backward_dx_tc.cu — Backward dX GEMM with 64×64 tile (WMMA).
 *
 * Grid:  (ceil(B / 64), ceil(K / 64))
 * Block: 128 threads (4 warps)
 *
 * Each CTA computes dX[b:b+64, k:k+64] = SUM_n dY[b:b+64, n] * W[n, k:k+64]
 *
 * Outer loop over N (out_features) in steps of 16.
 * SMEM (6 KB):
 *   dY_smem[64][16]  — 2 KB
 *   W_smem[64][16]   — 2 KB
 *   spill[4][16][16] — 4 KB (float, reused)
 * Total: ~10 KB (fits T4 48 KB)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "packed_ternary.cuh"
#include <mma.h>

namespace wmma = nvcuda::wmma;

constexpr int kWMMA_M = 16;
constexpr int kWMMA_N = 16;
constexpr int kWMMA_K = 16;
constexpr int kSuperM = 64;
constexpr int kSuperN = 64;
constexpr int kWarps  = 4;
constexpr int kFragsPerWarp = 4;

__global__ __launch_bounds__(128) void packed_ternary_backward_dx_tc_64_kernel(
    const uint32_t* __restrict__ W,   // [N, stride] packed
    const half*     __restrict__ dY,  // [B, N] FP16
    half*           __restrict__ dX,  // [B, K] FP16
    int B, int K, int N, int stride_words)
{
    int super_b0 = blockIdx.x * kSuperM;
    int super_k0 = blockIdx.y * kSuperN;

    int warp_id = threadIdx.x / 32;
    int warp_b_off = (warp_id / 2) * 32;
    int warp_k_off = (warp_id % 2) * 32;

    __shared__ half dY_smem[kSuperM][kWMMA_K];  // [64][16]
    __shared__ half W_smem[kSuperN][kWMMA_K];   // [64][16]
    __shared__ float spill[kFragsPerWarp][kWMMA_M][kWMMA_N];

    wmma::fragment<wmma::matrix_a, kWMMA_M, kWMMA_N, kWMMA_K,
                   half, wmma::row_major> a_frag;  // dY
    wmma::fragment<wmma::matrix_b, kWMMA_M, kWMMA_N, kWMMA_K,
                   half, wmma::row_major> b_frag;  // W
    wmma::fragment<wmma::accumulator, kWMMA_M, kWMMA_N, kWMMA_K,
                   float> c_frag[kFragsPerWarp];

    #pragma unroll
    for (int f = 0; f < kFragsPerWarp; ++f)
        wmma::fill_fragment(c_frag[f], 0.0f);

    // Outer loop over N (out_features) in steps of 16
    for (int r0 = 0; r0 < N; r0 += kWMMA_K) {
        int tile_r = min(kWMMA_K, N - r0);

        // Load dY[super_b0:super_b0+64, r0:r0+16]
        {
            int n_total = kSuperM * kWMMA_K;
            for (int tid = threadIdx.x; tid < n_total; tid += 128) {
                int r = tid / kWMMA_K;
                int c = tid % kWMMA_K;
                int gb = super_b0 + r;
                int gn = r0 + c;
                if (gb < B && gn < N && c < tile_r) {
                    dY_smem[r][c] = dY[gb * N + gn];
                } else {
                    dY_smem[r][c] = __float2half(0.0f);
                }
            }
        }

        // Load W[r0:r0+16, super_k0:super_k0+64] (transposed: stored as [64][16])
        {
            int n_total = kSuperN * kWMMA_K;
            for (int tid = threadIdx.x; tid < n_total; tid += 128) {
                int r = tid / kWMMA_K;
                int c = tid % kWMMA_K;
                int gn = r0 + c;  // W row = out_feature index
                int gk = super_k0 + r;  // W col = in_feature index
                if (gn < N && gk < K && c < tile_r) {
                    int wi = gk / kWeightsPerWord;
                    int pos = gk % kWeightsPerWord;
                    uint32_t word = W[gn * stride_words + wi];
                    int8_t t = decode_ternary(word >> (2 * pos));
                    W_smem[r][c] = __int2half_rn(t);
                } else {
                    W_smem[r][c] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // WMMA: dX += dY_tile @ W_tile (WMMA M=16, dY row_major, W row_major)
        // Actually dX = dY @ W means:
        //   dY is [64×16] (batch x out), W is [64×16] (out x in)
        //   We need dY[b,n] * W[n,k] for each k in the 64-wide tile.
        // Each warp accumulates 4 fragments covering its 32×32 portion.
        #pragma unroll
        for (int fi = 0; fi < kFragsPerWarp; ++fi) {
            int frag_b_off = (fi / 2) * kWMMA_M;  // 0 or 16
            int frag_k_off = (fi % 2) * kWMMA_N;  // 0 or 16
            int b_base = warp_b_off + frag_b_off;
            int k_base = warp_k_off + frag_k_off;

            // a_frag = dY[b_base:b_base+15, 0:15]  (row_major)
            // b_frag = W[0:15, k_base:k_base+15]   (row_major)
            wmma::load_matrix_sync(a_frag, &dY_smem[b_base][0], kWMMA_K);
            wmma::load_matrix_sync(b_frag, &W_smem[k_base][0], kWMMA_K);
            wmma::mma_sync(c_frag[fi], a_frag, b_frag, c_frag[fi]);
        }

        __syncthreads();
    }

    // Store results to global dX
    #pragma unroll
    for (int fi = 0; fi < kFragsPerWarp; ++fi) {
        int frag_b_off = (fi / 2) * kWMMA_M;
        int frag_k_off = (fi % 2) * kWMMA_N;
        int b_base = warp_b_off + frag_b_off;
        int k_base = warp_k_off + frag_k_off;

        int gb0 = super_b0 + b_base;
        int gk0 = super_k0 + k_base;

        wmma::store_matrix_sync(&spill[fi][0][0], c_frag[fi],
                                kWMMA_N, wmma::mem_row_major);
        __syncthreads();

        int n_elems = kWMMA_M * kWMMA_N;
        for (int tid = threadIdx.x; tid < n_elems; tid += 128) {
            int r = tid / kWMMA_N;
            int c = tid % kWMMA_N;
            int gb = gb0 + r;
            int gk = gk0 + c;
            if (gb < B && gk < K) {
                dX[gb * K + gk] = __float2half_rn(spill[fi][r][c]);
            }
        }
        __syncthreads();
    }
}


extern "C" void launch_packed_ternary_backward_dx_tc_64(
    const uint32_t* W, const void* dY_ptr, void* dX_ptr,
    int batch_size, int in_features, int out_features,
    int stride_words, cudaStream_t stream)
{
    const half* dY = static_cast<const half*>(dY_ptr);
    half* dX = static_cast<half*>(dX_ptr);

    dim3 grid((batch_size + kSuperM - 1) / kSuperM,
              (in_features + kSuperN - 1) / kSuperN);
    dim3 block(128);

    packed_ternary_backward_dx_tc_64_kernel<<<grid, block, 0, stream>>>(
        W, dY, dX, batch_size, in_features, out_features, stride_words
    );
}
