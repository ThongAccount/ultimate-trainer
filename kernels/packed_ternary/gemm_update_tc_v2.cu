/**
 * gemm_update_tc_v2.cu — Weight update with 64×64 tile (WMMA, vectorized counter).
 *
 * Grid:  (ceil(N / 64), ceil(K / 64))
 * Block: 128 threads (4 warps)
 *
 * Each CTA owns a 64×64 tile of W (N_sub × K_sub).
 * Computes: dW[n,k] = SUM_b dY[b,n] * X[b,k] over batch B.
 * Then: counter += sign(dW), flip when |cnt| > threshold.
 *
 * SMEM (6 KB):
 *   dY_smem[64][16]  — 2 KB
 *   X_smem[64][16]   — 2 KB
 *   dW_float_smem[64][64] — 16 KB (float, for WMMA output)
 * Total: ~20 KB (fits T4 48 KB)
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

__global__ __launch_bounds__(128) void packed_ternary_update_tc_v2_64_kernel(
    const half*     __restrict__ X,       // [B, K] FP16
    const half*     __restrict__ dY,      // [B, N] FP16
    uint32_t*       __restrict__ W,       // [N, stride] packed
    int16_t*        __restrict__ counter,  // [N, K] int16
    int B, int K, int N, int stride_words,
    int16_t threshold)
{
    int super_n0 = blockIdx.x * kSuperM;
    int super_k0 = blockIdx.y * kSuperN;

    int warp_id = threadIdx.x / 32;
    int warp_n_off = (warp_id / 2) * 32;
    int warp_k_off = (warp_id % 2) * 32;

    __shared__ half dY_smem[kSuperM][kWMMA_K];  // dY[b:tile, 0:16], reloaded per batch-step
    __shared__ half X_smem[kSuperM][kWMMA_K];   // X[b:tile, 0:16], reloaded per batch-step
    __shared__ float dW_frag[kFragsPerWarp][kWMMA_M][kWMMA_N];  // fragment spill

    wmma::fragment<wmma::matrix_a, kWMMA_M, kWMMA_N, kWMMA_K,
                   half, wmma::col_major> a_frag;  // dY transposed for dW = dY^T @ X
    wmma::fragment<wmma::matrix_b, kWMMA_M, kWMMA_N, kWMMA_K,
                   half, wmma::row_major> b_frag;  // X
    wmma::fragment<wmma::accumulator, kWMMA_M, kWMMA_N, kWMMA_K,
                   float> c_frag[kFragsPerWarp];

    #pragma unroll
    for (int f = 0; f < kFragsPerWarp; ++f)
        wmma::fill_fragment(c_frag[f], 0.0f);

    // Batch loop: accumulate dW over all batch tiles
    for (int b0 = 0; b0 < B; b0 += kSuperM) {
        int tile_b = min(kSuperM, B - b0);

        // Load dY[b0:b0+64, super_n0:super_n0+64] in K-slices
        // Actually for WMMA we load one K-slice at a time
        // dY is indexed as dY[b, n]; we need dY[b, n] for batch tile
        // The WMMA for update is: dW = dY^T @ X, so we batch over B
        // dY dimension: [B, N]. We load dY[b0:b0+64, super_n0:super_n0+64]
        // in steps of kWMMA_M=16 across the N dimension

        // For now, load [64, 16] slice of dY starting at super_n0 offset
        {
            int n_total = kSuperM * kWMMA_K;
            for (int tid = threadIdx.x; tid < n_total; tid += 128) {
                int r = tid / kWMMA_K;
                int c = tid % kWMMA_K;
                int gb = b0 + r;
                int gn = super_n0 + c;
                if (gb < B && gn < N && r < tile_b) {
                    dY_smem[r][c] = dY[gb * N + gn];
                } else {
                    dY_smem[r][c] = __float2half(0.0f);
                }
            }
        }

        // Load X[b0:b0+64, super_k0:super_k0+64] in K-slices
        {
            int n_total = kSuperM * kWMMA_K;
            for (int tid = threadIdx.x; tid < n_total; tid += 128) {
                int r = tid / kWMMA_K;
                int c = tid % kWMMA_K;
                int gb = b0 + r;
                int gk = super_k0 + c;
                if (gb < B && gk < K && r < tile_b) {
                    X_smem[r][c] = X[gb * K + gk];
                } else {
                    X_smem[r][c] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // WMMA: dW += dY[b0:tile, super_n0+0:16]^T @ X[b0:tile, super_k0+0:16]
        // Each warp accumulates 4 fragments (32×32 sub-tile)
        #pragma unroll
        for (int fi = 0; fi < kFragsPerWarp; ++fi) {
            int frag_n_off = (fi / 2) * kWMMA_M;  // 0 or 16
            int frag_k_off = (fi % 2) * kWMMA_N;  // 0 or 16
            int n_base = warp_n_off + frag_n_off;
            int k_base = warp_k_off + frag_k_off;

            // a_frag: dY[b, super_n0+n_base] — col_major (encoded in fragment type)
            wmma::load_matrix_sync(a_frag, &dY_smem[0][n_base], kWMMA_K);
            wmma::load_matrix_sync(b_frag, &X_smem[0][k_base], kWMMA_K);
            wmma::mma_sync(c_frag[fi], a_frag, b_frag, c_frag[fi]);
        }

        __syncthreads();
    }

    // Store accumulator fragments to local SMEM for counter processing
    #pragma unroll
    for (int fi = 0; fi < kFragsPerWarp; ++fi) {
        wmma::store_matrix_sync(&dW_frag[fi][0][0], c_frag[fi],
                                kWMMA_N, wmma::mem_row_major);
    }
    __syncthreads();

    // Process counter updates over the 64×64 tile
    int n_pairs = (kWMMA_K * kWMMA_M * kFragsPerWarp * kWarps) / 2;  // (64*64)/2 = 2048
    for (int i = threadIdx.x; i < n_pairs; i += 128) {
        // Map pair index -> (warp, frag, row, col)
        int idx2 = i * 2;
        int frag_global = idx2 / (kWMMA_M * kWMMA_N);  // 0..15
        int linear = idx2 % (kWMMA_M * kWMMA_N);
        int r = linear / kWMMA_N;
        int c = linear % kWMMA_N;

        int warp_idx = frag_global / kFragsPerWarp;
        int frag_local = frag_global % kFragsPerWarp;
        int frag_n_off = (frag_local / 2) * kWMMA_M;
        int frag_k_off = (frag_local % 2) * kWMMA_N;
        int warp_n_off_w = (warp_idx / 2) * 32;
        int warp_k_off_w = (warp_idx % 2) * 32;

        int gn = super_n0 + warp_n_off_w + frag_n_off + r;
        int gk = super_k0 + warp_k_off_w + frag_k_off + c;

        if (gn >= N || gk + 1 >= K) continue;

        float g0 = dW_frag[frag_global][r][c];
        float g1 = dW_frag[frag_global][r][c + 1];

        if (g0 == 0.0f && g1 == 0.0f) continue;

        int idx = gn * K + gk;
        // Vectorized int32 counter load (with alignment check)
        int16_t cnt0, cnt1;
        if ((idx * (int)sizeof(int16_t)) & 3) {
            cnt0 = counter[idx];
            cnt1 = counter[idx + 1];
        } else {
            int32_t cnt_pair = *(const int32_t*)&counter[idx];
            cnt0 = (int16_t)(cnt_pair & 0xFFFF);
            cnt1 = (int16_t)((cnt_pair >> 16) & 0xFFFF);
        }

        cnt0 += (g0 > 0.0f) ? -1 : (g0 < 0.0f) ? 1 : 0;
        cnt1 += (g1 > 0.0f) ? -1 : (g1 < 0.0f) ? 1 : 0;

        uint32_t* w_row = W + gn * stride_words;
        if (cnt0 > threshold) { increment_weight_atomic(w_row, gk); cnt0 = 0; }
        else if (cnt0 < -threshold) { decrement_weight_atomic(w_row, gk); cnt0 = 0; }
        if (cnt1 > threshold) { increment_weight_atomic(w_row, gk + 1); cnt1 = 0; }
        else if (cnt1 < -threshold) { decrement_weight_atomic(w_row, gk + 1); cnt1 = 0; }

        // Store
        if ((idx * (int)sizeof(int16_t)) & 3) {
            counter[idx] = cnt0;
            counter[idx + 1] = cnt1;
        } else {
            *(int32_t*)&counter[idx] = ((int32_t)cnt1 << 16) | ((int32_t)cnt0 & 0xFFFF);
        }
    }

    // Tail: handle last column when K is odd
    if (K & 1) {
        int last_gk = K - 1;
        int total_elems = kWarps * kFragsPerWarp * kWMMA_M * kWMMA_N;
        for (int i = threadIdx.x; i < total_elems; i += 128) {
            int frag_global = i / (kWMMA_M * kWMMA_N);
            int linear = i % (kWMMA_M * kWMMA_N);
            int r = linear / kWMMA_N;
            int c = linear % kWMMA_N;
            int warp_idx = frag_global / kFragsPerWarp;
            int frag_local = frag_global % kFragsPerWarp;
            int frag_n_off = (frag_local / 2) * kWMMA_M;
            int frag_k_off = (frag_local % 2) * kWMMA_N;
            int warp_n_off_w = (warp_idx / 2) * 32;
            int warp_k_off_w = (warp_idx % 2) * 32;
            // Only process threads mapping to the last column
            int gk = super_k0 + warp_k_off_w + frag_k_off + c;
            if (gk != last_gk) continue;
            int gn = super_n0 + warp_n_off_w + frag_n_off + r;
            if (gn >= N) continue;
            float g = dW_frag[frag_global][r][c];
            if (g == 0.0f) continue;
            int idx = gn * K + last_gk;
            int16_t cnt = counter[idx];
            cnt += (g > 0.0f) ? -1 : (g < 0.0f) ? 1 : 0;
            uint32_t* w_row = W + gn * stride_words;
            if (cnt > threshold) { increment_weight_atomic(w_row, last_gk); cnt = 0; }
            else if (cnt < -threshold) { decrement_weight_atomic(w_row, last_gk); cnt = 0; }
            counter[idx] = cnt;
        }
    }
}


extern "C" void launch_packed_ternary_update_tc_v2_64(
    const void* X_ptr, const void* dY_ptr,
    uint32_t* W, int16_t* counter,
    int batch_size, int in_features, int out_features,
    int stride_words, int16_t threshold,
    cudaStream_t stream)
{
    const half* X = static_cast<const half*>(X_ptr);
    const half* dY = static_cast<const half*>(dY_ptr);

    dim3 grid((out_features + kSuperM - 1) / kSuperM,
              (in_features + kSuperN - 1) / kSuperN);
    dim3 block(128);

    packed_ternary_update_tc_v2_64_kernel<<<grid, block, 0, stream>>>(
        X, dY, W, counter,
        batch_size, in_features, out_features, stride_words, threshold
    );
}
