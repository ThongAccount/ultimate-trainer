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

    // Session-10 K32 experiment: two WMMA K-slices per sync window.
    // dY tile [64][32] (4KB), W tile [32][64] (4KB) — smem still ~12KB on
    // a 64KB T4 SM, so occupancy should hold; measure, don't assume.
    constexpr int kK2 = 2 * kWMMA_K;  // 32
    __shared__ half dY_smem[kSuperM][kK2];  // [64][32]
    __shared__ half W_smem[kK2][kSuperN];   // [32][64] — reduction × output
    __shared__ float spill[kWarps][kWMMA_M][kWMMA_N];  // [4][16][16] per-warp spill

    wmma::fragment<wmma::matrix_a, kWMMA_M, kWMMA_N, kWMMA_K,
                   half, wmma::row_major> a_frag;  // dY
    wmma::fragment<wmma::matrix_b, kWMMA_M, kWMMA_N, kWMMA_K,
                   half, wmma::row_major> b_frag;  // W
    wmma::fragment<wmma::accumulator, kWMMA_M, kWMMA_N, kWMMA_K,
                   float> c_frag[kFragsPerWarp];

    #pragma unroll
    for (int f = 0; f < kFragsPerWarp; ++f)
        wmma::fill_fragment(c_frag[f], 0.0f);

    // Outer loop over N (out_features) in steps of 32 (K32 experiment)
    for (int r0 = 0; r0 < N; r0 += kK2) {
        int tile_r = min(kK2, N - r0);

        // Load dY[super_b0:super_b0+64, r0:r0+32]
        {
            int n_total = kSuperM * kK2;
            for (int tid = threadIdx.x; tid < n_total; tid += 128) {
                int r = tid / kK2;
                int c = tid % kK2;
                int gb = super_b0 + r;
                int gn = r0 + c;
                if (gb < B && gn < N && c < tile_r) {
                    dY_smem[r][c] = dY[gb * N + gn];
                } else {
                    dY_smem[r][c] = __float2half(0.0f);
                }
            }
        }

        // Load W[r0:r0+32, super_k0:super_k0+64] → W_smem[n][k]
        {
            int n_total = kSuperN * kK2;
            for (int tid = threadIdx.x; tid < n_total; tid += 128) {
                int r = tid / kSuperN;
                int c = tid % kSuperN;
                int gn = r0 + r;
                int gk = super_k0 + c;
                if (gn < N && gk < K && r < tile_r) {
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

        // 4 frags × 2 K-slices per sync window
        #pragma unroll
        for (int fi = 0; fi < kFragsPerWarp; ++fi) {
            int frag_b_off = (fi / 2) * kWMMA_M;
            int frag_k_off = (fi % 2) * kWMMA_N;
            int b_base = warp_b_off + frag_b_off;
            int k_base = warp_k_off + frag_k_off;

            #pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                wmma::load_matrix_sync(a_frag, &dY_smem[b_base][ks * kWMMA_K], kK2);
                wmma::load_matrix_sync(b_frag, &W_smem[ks * kWMMA_K][k_base], kSuperN);
                wmma::mma_sync(c_frag[fi], a_frag, b_frag, c_frag[fi]);
            }
        }

        __syncthreads();
    }

    // Store results to global dX
    #pragma unroll
    for (int fi = 0; fi < kFragsPerWarp; ++fi) {
        int frag_b_off = (fi / 2) * kWMMA_M;
        int frag_k_off = (fi % 2) * kWMMA_N;

        wmma::store_matrix_sync(&spill[warp_id][0][0], c_frag[fi],
                                kWMMA_N, wmma::mem_row_major);
        __syncthreads();

        // All 128 threads cooperate to write all 4 warps' 16×16 tiles
        int n_elems = kWarps * kWMMA_M * kWMMA_N;  // 1024
        for (int tid = threadIdx.x; tid < n_elems; tid += 128) {
            int w = tid / (kWMMA_M * kWMMA_N);
            int rem = tid % (kWMMA_M * kWMMA_N);
            int r = rem / kWMMA_N;
            int c = rem % kWMMA_N;
            int w_b_off = (w / 2) * 32;
            int w_k_off = (w % 2) * 32;
            int gb = super_b0 + w_b_off + frag_b_off + r;
            int gk = super_k0 + w_k_off + frag_k_off + c;
            if (gb < B && gk < K) {
                dX[gb * K + gk] = __float2half_rn(spill[w][r][c]);
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
