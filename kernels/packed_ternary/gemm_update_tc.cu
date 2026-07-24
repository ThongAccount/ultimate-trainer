/**
 * gemm_update_tc.cu — Tensor-Core fused gradient → counter → bit-flip.
 *
 * Uses WMMA Tensor Cores to compute dW = dY^T @ X, then applies
 * sign → int16 counter → bit-flip from shared memory, all in one kernel.
 *
 * Optimizations:
 *   - SMEM bank-conflict padding (stride 16→17)
 *   - half2 vectorized X + dY loads
 *   - Block-contiguous fill (8 elements/thread, contiguous)
 *   - 4-warp 32×32 super-tile for occupancy
 *   - __launch_bounds__(128)
 *
 * Grid:  (ceil(in_features / 32), ceil(out_features / 32))
 * Block: 128 threads (4 warps)
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "packed_ternary.cuh"
#include <mma.h>

namespace wmma = nvcuda::wmma;

constexpr int kM = 16;   // WMMA: out_features tile
constexpr int kN = 16;   // WMMA: in_features tile
constexpr int kK = 16;   // WMMA: batch tile (reduction dim)
constexpr int kPad = 1;  // SMEM bank-conflict padding
constexpr int kWarpsPerBlock = 4;
constexpr int kSuperM = 32;  // super-tile out (2 × kM)
constexpr int kSuperN = 32;  // super-tile in  (2 × kN)

#define DYS(w, b, r)  dY_smem[(w) * kK * (kM + kPad) + (b) * (kM + kPad) + (r)]
#define XS(w, b, c)   X_smem[(w) * kK * (kN + kPad) + (b) * (kN + kPad) + (c)]
#define DWF(w, r, c)  dW_float_smem[(w) * kM * (kN + kPad) + (r) * (kN + kPad) + (c)]

__global__ __launch_bounds__(128) void packed_ternary_update_tc_kernel(
    const half*     __restrict__ X,       // (batch, in_features)
    const half*     __restrict__ dY,      // (batch, out_features)
    uint32_t*       __restrict__ W,       // (out_features, stride_words) — IN PLACE
    int16_t*        __restrict__ counter, // (out_features * in_features)
    int batch_size,
    int in_features,
    int out_features,
    int stride_words,
    int16_t threshold)
{
    int super_c0 = blockIdx.x * kSuperN;  // super-tile in offset
    int super_r0 = blockIdx.y * kSuperM;  // super-tile out offset
    int warp_id = threadIdx.x / 32;       // 0..3
    int wtid    = threadIdx.x % 32;       // 0..31

    int warp_c_off = (warp_id / 2) * kN;   // 0 or 16
    int warp_r_off = (warp_id % 2) * kM;   // 0 or 16
    int c0 = super_c0 + warp_c_off;
    int r0 = super_r0 + warp_r_off;

    // ── Shared memory with padding ────────────────────────────────────
    __shared__ half   dY_smem[kWarpsPerBlock * kK * (kM + kPad)];
    __shared__ half   X_smem[kWarpsPerBlock * kK * (kN + kPad)];
    __shared__ float  dW_float_smem[kWarpsPerBlock * kM * (kN + kPad)];

    // ── WMMA fragments ──────────────────────────────────────────────
    wmma::fragment<wmma::matrix_a, kM, kN, kK, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, kM, kN, kK, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, kM, kN, kK, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // ── Outer loop over B (batch) tiles ─────────────────────────────
    for (int b0 = 0; b0 < batch_size; b0 += kK) {
        int tile_b = min(kK, batch_size - b0);

        // ── Load dY tile → dY_smem (half2, block fill) ───────────────
        {
            int base = wtid * 8;
            for (int j = 0; j < 8; j += 2) {
                int i = base + j;
                int b = i / kM;               // 0..15
                int r = i % kM;               // 0..15
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

        // ── Load X tile → X_smem (half2, block fill) ─────────────────
        {
            int base = wtid * 8;
            for (int j = 0; j < 8; j += 2) {
                int i = base + j;
                int b = i / kN;               // 0..15
                int c = i % kN;               // 0..15
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

        // ── Load WMMA fragments from SMEM ───────────────────────────
        wmma::load_matrix_sync(a_frag, &dY_smem[warp_id * kK * (kM + kPad)], kM + kPad);
        wmma::load_matrix_sync(b_frag, &X_smem[warp_id * kK * (kN + kPad)], kN + kPad);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // ── Store accumulator → SMEM → sign→counter→flip ─────────────────
    wmma::store_matrix_sync(&dW_float_smem[warp_id * kM * (kN + kPad)], c_frag,
                            kN + kPad, wmma::mem_row_major);
    __syncthreads();

    // 128 threads handle 1024 elements (8 per thread)
    int n_elems = kWarpsPerBlock * kM * (kN + kPad);
    for (int i = threadIdx.x; i < n_elems; i += blockDim.x) {
        int w = i / (kM * (kN + kPad));
        int linear = i % (kM * (kN + kPad));
        int r = linear / (kN + kPad);
        int c = linear % (kN + kPad);
        if (r >= kM || c >= kN) continue;  // skip padding

        int warp_r_off_w = (w % 2) * kM;
        int warp_c_off_w = (w / 2) * kN;
        int gr = super_r0 + warp_r_off_w + r;
        int gc = super_c0 + warp_c_off_w + c;

        if (gr < out_features && gc < in_features) {
            float grad = DWF(w, r, c);

            int idx = gr * in_features + gc;
            int16_t cnt = counter[idx];

            // Gradient descent: positive dW → decrease weight → decrement
            if (grad > 0.0f)       cnt--;
            else if (grad < 0.0f)  cnt++;

            uint32_t* w_row = W + gr * stride_words;

            if (cnt > threshold) {
                increment_weight_atomic(w_row, gc);
                cnt = 0;
            } else if (cnt < -threshold) {
                decrement_weight_atomic(w_row, gc);
                cnt = 0;
            }

            counter[idx] = cnt;
        }
    }
}

extern "C" void launch_packed_ternary_update_tc(
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
    dim3 block(128);  // 4 warps

    packed_ternary_update_tc_kernel<<<grid, block, 0, stream>>>(
        X, dY, W, counter, batch_size, in_features, out_features,
        stride_words, threshold
    );
}
