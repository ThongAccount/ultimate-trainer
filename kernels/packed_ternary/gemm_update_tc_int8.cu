/**
 * gemm_update_tc_int8.cu — TC gradient → int8 counter → bit-flip.
 *
 * Same as gemm_update_tc_v2 but with int8 counters (halves memory traffic).
 * Counter traffic: 32MB read+write (vs 64MB for int16).
 * Limits threshold to 127 (fine for threshold=8).
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

__global__ __launch_bounds__(128) void packed_ternary_update_tc_int8_kernel(
    const half*     __restrict__ X,
    const half*     __restrict__ dY,
    uint32_t*       __restrict__ W,
    int8_t*         __restrict__ counter,  // int8 instead of int16
    int batch_size,
    int in_features,
    int out_features,
    int stride_words,
    int8_t threshold)  // int8 threshold
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

    // WMMA accumulation loop
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
                        if (r + 1 < kM) {
                            half2 v = ((const half2*)&dY[gb * out_features + gr])[0];
                            DYS(warp_id, b, r)     = v.x;
                            DYS(warp_id, b, r + 1) = v.y;
                        } else {
                            DYS(warp_id, b, r) = dY[gb * out_features + gr];
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
                        if (c + 1 < kN) {
                            half2 v = ((const half2*)&X[gb * in_features + gc])[0];
                            XS(warp_id, b, c)     = v.x;
                            XS(warp_id, b, c + 1) = v.y;
                        } else {
                            XS(warp_id, b, c) = X[gb * in_features + gc];
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

    // Store accumulator to SMEM
    wmma::store_matrix_sync(&dW_float_smem[warp_id * kM * kN], c_frag,
                            kN, wmma::mem_row_major);
    __syncthreads();

    // Counter update: vectorized int32 (4× int8), skip zero grads
    int n_pairs = (kWarpsPerBlock * kM * kN) / 4;  // 256 int8 per int32
    for (int i = threadIdx.x; i < n_pairs; i += blockDim.x) {
        int quad_w = (i * 4) / (kM * kN);
        int quad_linear = (i * 4) % (kM * kN);
        int r = quad_linear / kN;
        int c = quad_linear % kN;

        int warp_r_off_w = (quad_w % 2) * kM;
        int warp_c_off_w = (quad_w / 2) * kN;
        int gr = super_r0 + warp_r_off_w + r;
        int gc = super_c0 + warp_c_off_w + c;

        if (gr >= out_features || gc + 3 >= in_features) continue;

        // Read 4 gradients
        float g0 = DWF(quad_w, r, c);
        float g1 = DWF(quad_w, r, c + 1);
        float g2 = DWF(quad_w, r, c + 2);
        float g3 = DWF(quad_w, r, c + 3);

        // Skip if all zero
        if (g0 == 0.0f && g1 == 0.0f && g2 == 0.0f && g3 == 0.0f) continue;

        int idx = gr * in_features + gc;

        // Vectorized counter load: 4 int8 as one int32
        int32_t cnt_quad = *(const int32_t*)&counter[idx];
        int8_t cnt0 = (int8_t)(cnt_quad & 0xFF);
        int8_t cnt1 = (int8_t)((cnt_quad >> 8) & 0xFF);
        int8_t cnt2 = (int8_t)((cnt_quad >> 16) & 0xFF);
        int8_t cnt3 = (int8_t)((cnt_quad >> 24) & 0xFF);

        // Branchless counter update
        cnt0 += (g0 > 0.0f) ? -1 : (g0 < 0.0f) ? 1 : 0;
        cnt1 += (g1 > 0.0f) ? -1 : (g1 < 0.0f) ? 1 : 0;
        cnt2 += (g2 > 0.0f) ? -1 : (g2 < 0.0f) ? 1 : 0;
        cnt3 += (g3 > 0.0f) ? -1 : (g3 < 0.0f) ? 1 : 0;

        // Weight flips
        uint32_t* w_row = W + gr * stride_words;

        if (cnt0 > threshold) { increment_weight_atomic(w_row, gc);     cnt0 = 0; }
        else if (cnt0 < -threshold) { decrement_weight_atomic(w_row, gc);     cnt0 = 0; }
        if (cnt1 > threshold) { increment_weight_atomic(w_row, gc + 1); cnt1 = 0; }
        else if (cnt1 < -threshold) { decrement_weight_atomic(w_row, gc + 1); cnt1 = 0; }
        if (cnt2 > threshold) { increment_weight_atomic(w_row, gc + 2); cnt2 = 0; }
        else if (cnt2 < -threshold) { decrement_weight_atomic(w_row, gc + 2); cnt2 = 0; }
        if (cnt3 > threshold) { increment_weight_atomic(w_row, gc + 3); cnt3 = 0; }
        else if (cnt3 < -threshold) { decrement_weight_atomic(w_row, gc + 3); cnt3 = 0; }

        // Vectorized counter store: 4 int8 as one int32
        *(int32_t*)&counter[idx] = ((int32_t)(uint8_t)cnt3 << 24) |
                                   ((int32_t)(uint8_t)cnt2 << 16) |
                                   ((int32_t)(uint8_t)cnt1 << 8)  |
                                   ((int32_t)(uint8_t)cnt0);
    }
}

extern "C" void launch_packed_ternary_update_tc_int8(
    const void*     X_ptr,
    const void*     dY_ptr,
    uint32_t*       W,
    int8_t*         counter,
    int batch_size,
    int in_features,
    int out_features,
    int stride_words,
    int8_t threshold,
    cudaStream_t stream)
{
    const half* X  = static_cast<const half*>(X_ptr);
    const half* dY = static_cast<const half*>(dY_ptr);

    dim3 grid((in_features + kSuperN - 1) / kSuperN,
              (out_features + kSuperM - 1) / kSuperM);
    dim3 block(128);

    packed_ternary_update_tc_int8_kernel<<<grid, block, 0, stream>>>(
        X, dY, W, counter, batch_size, in_features, out_features,
        stride_words, threshold
    );
}
