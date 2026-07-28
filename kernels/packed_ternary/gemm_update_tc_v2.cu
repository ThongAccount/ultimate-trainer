/**
 * gemm_update_tc_v2.cu — Optimized TC gradient → counter → bit-flip.
 *
 * v2 optimizations over v1:
 *   1. Skip counter read/write when grad == 0 (saves 32MB+ traffic)
 *   2. Vectorized counter access: int32 for two adjacent int16 counters
 *   3. Branchless counter increment/decrement
 *   4. Reduced __syncthreads() calls
 *
 * Grid:  (ceil(in_features / 32), ceil(out_features / 32))
 * Block: 128 threads (4 warps)
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "packed_ternary.cuh"
#include <mma.h>

namespace wmma = nvcuda::wmma;

constexpr int kM = 16;
constexpr int kN = 16;
constexpr int kK = 16;
constexpr int kWarpsPerBlock = 4;
constexpr int kSuperM = 32;
constexpr int kSuperN = 32;

#define DYS(w, b, r)  dY_smem[(w) * kK * kM + (b) * kM + (r)]
#define XS(w, b, c)   X_smem[(w) * kK * kN + (b) * kN + (c)]
#define DWF(w, r, c)  dW_float_smem[(w) * kM * kN + (r) * kN + (c)]

__global__ __launch_bounds__(128) void packed_ternary_update_tc_v2_kernel(
    const half*     __restrict__ X,
    const half*     __restrict__ dY,
    uint32_t*       __restrict__ W,
    int16_t*        __restrict__ counter,
    int batch_size,
    int in_features,
    int out_features,
    int stride_words,
    int16_t threshold)
{
    int super_c0 = blockIdx.x * kSuperN;
    int super_r0 = blockIdx.y * kSuperM;
    int warp_id = threadIdx.x / 32;
    int wtid    = threadIdx.x % 32;

    int warp_c_off = (warp_id / 2) * kN;
    int warp_r_off = (warp_id % 2) * kM;
    int c0 = super_c0 + warp_c_off;
    int r0 = super_r0 + warp_r_off;

    __shared__ half   dY_smem[kWarpsPerBlock * kK * kM];
    __shared__ half   X_smem[kWarpsPerBlock * kK * kN];
    __shared__ float  dW_float_smem[kWarpsPerBlock * kM * kN];

    wmma::fragment<wmma::matrix_a, kM, kN, kK, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, kM, kN, kK, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, kM, kN, kK, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // ── WMMA accumulation loop over batch tiles ─────────────────────
    for (int b0 = 0; b0 < batch_size; b0 += kK) {
        int tile_b = min(kK, batch_size - b0);

        // Load dY tile
        {
            int base = wtid * 8;
            for (int j = 0; j < 8; j += 2) {
                int i = base + j;
                int b = i / kM;
                int r = i % kM;
                if (b < tile_b) {
                    int gb = b0 + b;
                    int gr = r0 + r;
                    if (gb < batch_size && gr < out_features) {
                        int byte_off = (gb * out_features + gr) * (int)sizeof(half);
                        if ((byte_off & 3) == 0 && r + 1 < kM && gr + 1 < out_features) {
                            half2 v = ((const half2*)&dY[gb * out_features + gr])[0];
                            DYS(warp_id, b, r)     = v.x;
                            DYS(warp_id, b, r + 1) = v.y;
                        } else {
                            DYS(warp_id, b, r) = dY[gb * out_features + gr];
                            if (r + 1 < kM && gr + 1 < out_features) {
                                DYS(warp_id, b, r + 1) = dY[gb * out_features + gr + 1];
                            }
                        }
                    }
                }
            }
        }

        // Load X tile
        {
            int base = wtid * 8;
            for (int j = 0; j < 8; j += 2) {
                int i = base + j;
                int b = i / kN;
                int c = i % kN;
                if (b < tile_b) {
                    int gb = b0 + b;
                    int gc = c0 + c;
                    if (gb < batch_size && gc < in_features) {
                        int byte_off = (gb * in_features + gc) * (int)sizeof(half);
                        if ((byte_off & 3) == 0 && c + 1 < kN && gc + 1 < in_features) {
                            half2 v = ((const half2*)&X[gb * in_features + gc])[0];
                            XS(warp_id, b, c)     = v.x;
                            XS(warp_id, b, c + 1) = v.y;
                        } else {
                            XS(warp_id, b, c) = X[gb * in_features + gc];
                            if (c + 1 < kN && gc + 1 < in_features) {
                                XS(warp_id, b, c + 1) = X[gb * in_features + gc + 1];
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, &dY_smem[warp_id * kK * kM], kM);
        wmma::load_matrix_sync(b_frag, &X_smem[warp_id * kK * kN], kN);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // ── Store accumulator to SMEM ───────────────────────────────────
    wmma::store_matrix_sync(&dW_float_smem[warp_id * kM * kN], c_frag,
                            kN, wmma::mem_row_major);
    __syncthreads();

    // ── Counter update: vectorized int32 pairs, skip zero grads ─────
    //
    // Process 2 adjacent elements per iteration (int32 counter load).
    // Skip counter read+write when both gradients are zero.
    // 128 threads handle 1024 elements = 512 pairs.
    //
    int n_pairs = (kWarpsPerBlock * kM * kN) / 2;  // 512
    for (int i = threadIdx.x; i < n_pairs; i += blockDim.x) {
        // Map pair index → (warp, row, col) of the FIRST element
        int pair_w = (i * 2) / (kM * kN);
        int pair_linear = (i * 2) % (kM * kN);
        int r = pair_linear / kN;
        int c = pair_linear % kN;

        int warp_r_off_w = (pair_w % 2) * kM;
        int warp_c_off_w = (pair_w / 2) * kN;
        int gr = super_r0 + warp_r_off_w + r;
        int gc = super_c0 + warp_c_off_w + c;

        // Bounds check for the pair (both elements in same row)
        if (gr >= out_features || gc + 1 >= in_features) continue;

        // Read both gradients from SMEM
        float g0 = DWF(pair_w, r, c);
        float g1 = DWF(pair_w, r, c + 1);

        // Skip if both gradients are zero (saves counter read + write)
        if (g0 == 0.0f && g1 == 0.0f) continue;

        int idx = gr * in_features + gc;

        // Vectorized counter load: two int16 as one int32
        int32_t cnt_pair = *(const int32_t*)&counter[idx];
        int16_t cnt0 = (int16_t)(cnt_pair & 0xFFFF);
        int16_t cnt1 = (int16_t)((cnt_pair >> 16) & 0xFFFF);

        // Branchless counter update: -1 if grad>0, +1 if grad<0, 0 if grad==0
        cnt0 += (g0 > 0.0f) ? -1 : (g0 < 0.0f) ? 1 : 0;
        cnt1 += (g1 > 0.0f) ? -1 : (g1 < 0.0f) ? 1 : 0;

        // Weight flips for cnt0
        uint32_t* w_row = W + gr * stride_words;
        if (cnt0 > threshold) {
            increment_weight_atomic(w_row, gc);
            cnt0 = 0;
        } else if (cnt0 < -threshold) {
            decrement_weight_atomic(w_row, gc);
            cnt0 = 0;
        }

        // Weight flips for cnt1
        if (cnt1 > threshold) {
            increment_weight_atomic(w_row, gc + 1);
            cnt1 = 0;
        } else if (cnt1 < -threshold) {
            decrement_weight_atomic(w_row, gc + 1);
            cnt1 = 0;
        }

        // Vectorized counter store
        *(int32_t*)&counter[idx] = ((int32_t)cnt1 << 16) | ((int32_t)cnt0 & 0xFFFF);
    }
}

extern "C" void launch_packed_ternary_update_tc_v2(
    const void*     X_ptr,
    const void*     dY_ptr,
    uint32_t*       W,
    int16_t*        counter,
    int batch_size,
    int in_features,
    int out_features,
    int stride_words,
    int16_t threshold,
    cudaStream_t stream)
{
    const half* X  = static_cast<const half*>(X_ptr);
    const half* dY = static_cast<const half*>(dY_ptr);

    dim3 grid((in_features + kSuperN - 1) / kSuperN,
              (out_features + kSuperM - 1) / kSuperM);
    dim3 block(128);

    packed_ternary_update_tc_v2_kernel<<<grid, block, 0, stream>>>(
        X, dY, W, counter, batch_size, in_features, out_features,
        stride_words, threshold
    );
}
