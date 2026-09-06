/*
 * Status note: 64×64 update path was dead code until session 8 fixed the
 * WMMA addressing (see header below). Still not wired into the training
 * dispatch — reachable only via pack_update._load_up_tc_v2()/update() 64×64
 * gate. Production uses gemm_update_tc_v2_32.cu.
 */
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
 * WMMA addressing fixed (session 8): smem tiles are reduction-major,
 * [batch 16][output 64], fragment ldm = 64 (was ldm=16 against a 64-wide
 * row → column shift for every n_base/k_base > 0, and 4-warp race on a
 * single spill slot). Same defect class as 113b40b for backward_dx.
 *
 * SMEM (~12 KB):
 *   dY_smem[16][64]  — 2 KB
 *   X_smem[16][64]   — 2 KB
 *   spill[4][16][16] — 4 KB (per-warp, float)
 * Total: ~8 KB
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

    // Reduction-major tiles (fix for session-8 WMMA addressing bug, same
    // class as 113b40b): reduction dim (batch) rows × full 64-wide output.
    __shared__ half dY_smem[kWMMA_K][kSuperM];  // dY[b:16][n:64] reduction × rows
    __shared__ half X_smem[kWMMA_K][kSuperN];   // X[b:16][k:64] reduction × cols
    __shared__ float spill[kWarps][kWMMA_M][kWMMA_N];  // per-warp 16×16 spill

    wmma::fragment<wmma::matrix_a, kWMMA_M, kWMMA_N, kWMMA_K,
                   half, wmma::col_major> a_frag;  // dY transposed for dW = dY^T @ X
    wmma::fragment<wmma::matrix_b, kWMMA_M, kWMMA_N, kWMMA_K,
                   half, wmma::row_major> b_frag;  // X
    wmma::fragment<wmma::accumulator, kWMMA_M, kWMMA_N, kWMMA_K,
                   float> c_frag[kFragsPerWarp];

    #pragma unroll
    for (int f = 0; f < kFragsPerWarp; ++f)
        wmma::fill_fragment(c_frag[f], 0.0f);

    // Batch loop: accumulate dW over batch, 16 rows (one WMMA-K) at a time.
    for (int b0 = 0; b0 < B; b0 += kWMMA_K) {
        int tile_b = min(kWMMA_K, B - b0);

        // Load dY[b0:b0+64, super_n0:super_n0+64] in K-slices
        // Actually for WMMA we load one K-slice at a time
        // dY is indexed as dY[b, n]; we need dY[b, n] for batch tile
        // The WMMA for update is: dW = dY^T @ X, so we batch over B
        // dY dimension: [B, N]. We load dY[b0:b0+64, super_n0:super_n0+64]
        // in steps of kWMMA_M=16 across the N dimension

        // Load dY[b0:b0+16, super_n0:super_n0+64] → dY_smem[b][n]
        // (reduction × output, like the fixed bwd-dX kernel in 113b40b)
        {
            int n_total = kWMMA_K * kSuperM;
            for (int tid = threadIdx.x; tid < n_total; tid += 128) {
                int r = tid / kSuperM;   // reduction (batch, 0..15)
                int c = tid % kSuperM;   // output (out_features, 0..63)
                int gb = b0 + r;
                int gn = super_n0 + c;
                if (gb < B && gn < N && r < tile_b) {
                    dY_smem[r][c] = dY[gb * N + gn];
                } else {
                    dY_smem[r][c] = __float2half(0.0f);
                }
            }
        }

        // Load X[b0:b0+16, super_k0:super_k0+64] → X_smem[b][k]
        {
            int n_total = kWMMA_K * kSuperN;
            for (int tid = threadIdx.x; tid < n_total; tid += 128) {
                int r = tid / kSuperN;
                int c = tid % kSuperN;
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

            // a_frag = dY^T slice: col_major, element (m,kk) =
            // dY_smem[kk][n_base+m] via ld = kSuperM (NOT kWMMA_K=16 —
            // the old code's ld=16 hit smem row j+n_base/16: shift bug).
            wmma::load_matrix_sync(a_frag, &dY_smem[0][n_base], kSuperM);
            // b_frag = X slice: row_major, element (kk,j) =
            // X_smem[kk][k_base+j] via ld = kSuperN.
            wmma::load_matrix_sync(b_frag, &X_smem[0][k_base], kSuperN);
            wmma::mma_sync(c_frag[fi], a_frag, b_frag, c_frag[fi]);
        }

        __syncthreads();
    }

    // Counter phase, per fragment — per-warp spill (no race, cf. 113b40b C)
    #pragma unroll
    for (int fi = 0; fi < kFragsPerWarp; ++fi) {
        int frag_n_off = (fi / 2) * kWMMA_M;
        int frag_k_off = (fi % 2) * kWMMA_N;

        wmma::store_matrix_sync(&spill[warp_id][0][0], c_frag[fi],
                                kWMMA_N, wmma::mem_row_major);
        __syncthreads();

        // 128 threads process 4 warps × 16×16 elems = 512 pairs
        int n_pairs = (kWarps * kWMMA_M * kWMMA_N) / 2;  // 512
        for (int i = threadIdx.x; i < n_pairs; i += 128) {
            int idx2 = i * 2;
            int w = idx2 / (kWMMA_M * kWMMA_N);
            int linear = idx2 % (kWMMA_M * kWMMA_N);
            int r = linear / kWMMA_N;
            int c = linear % kWMMA_N;

            int w_n_off = (w / 2) * 32;
            int w_k_off = (w % 2) * 32;

            int gn = super_n0 + w_n_off + frag_n_off + r;
            int gk = super_k0 + w_k_off + frag_k_off + c;

            if (gn >= N || gk + 1 >= K) continue;

            float g0 = spill[w][r][c];
            float g1 = spill[w][r][c + 1];

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

        // Tail: last column when K is odd (same spill, same fi)
        if (K & 1) {
            int last_gk = K - 1;
            for (int i = threadIdx.x; i < kWarps * kWMMA_M * kWMMA_N; i += 128) {
                int w = i / (kWMMA_M * kWMMA_N);
                int linear = i % (kWMMA_M * kWMMA_N);
                int r = linear / kWMMA_N;
                int c = linear % kWMMA_N;
                int w_n_off = (w / 2) * 32;
                int w_k_off = (w % 2) * 32;
                int gk = super_k0 + w_k_off + frag_k_off + c;
                if (gk != last_gk) continue;
                int gn = super_n0 + w_n_off + frag_n_off + r;
                if (gn >= N) continue;
                float g = spill[w][r][c];
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
        __syncthreads();
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
