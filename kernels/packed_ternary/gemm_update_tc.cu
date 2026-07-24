/**
 * gemm_update_tc.cu — Tensor-Core fused gradient → counter → bit-flip.
 *
 * Uses WMMA Tensor Cores to compute dW = dY^T @ X, then applies
 * sign → int16 counter → bit-flip from shared memory, all in one kernel.
 *
 * No dW tensor is ever materialised in global memory.
 *
 * Grid:  (ceil(in_features / 16), ceil(out_features / 16))
 * Block: 256 threads
 *
 * For each batch tile of 16:
 *   1. Load dY_tile (batch × out) into shared memory
 *   2. Load X_tile   (batch × in)  into shared memory
 *   3. WMMA: accumulate dW[r][c] += Σ_b dY[b][r] * X[b][c]
 * After all batch tiles:
 *   4. Store dW to shared memory (float → 16×16 tile)
 *   5. Each thread: sign → counter → atomically flip ternary bit
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "packed_ternary.cuh"
#include <mma.h>

namespace wmma = nvcuda::wmma;

constexpr int kM = 16;   // WMMA: out_features tile
constexpr int kN = 16;   // WMMA: in_features tile
constexpr int kK = 16;   // WMMA: batch tile (reduction dim)

__global__ void packed_ternary_update_tc_kernel(
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
    int c0 = blockIdx.x * kN;     // in-feature column offset for this tile
    int r0 = blockIdx.y * kM;     // out-feature row offset for this tile
    int tid = threadIdx.x;        // 0..255

    // ── Shared memory ─────────────────────────────────────────────────
    __shared__ half   dY_smem[kK][kM];     // dY tile (col-major-friendly: batch × out)
    __shared__ half   X_smem[kK][kN];      // X tile  (row-major: batch × in)
    __shared__ float  dW_float_smem[kM][kN]; // output tile (WMMA writes to SMEM)

    // ── WMMA fragments ────────────────────────────────────────────────
    // dY as matrix_a col_major:  a_frag[r][b] = dY_smem[b][r]
    // X  as matrix_b row_major:  b_frag[b][c] = X_smem[b][c]
    wmma::fragment<wmma::matrix_a, kM, kN, kK, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, kM, kN, kK, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, kM, kN, kK, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // ── Outer loop over B (batch) tiles ───────────────────────────────
    for (int b0 = 0; b0 < batch_size; b0 += kK) {
        int tile_b = min(kK, batch_size - b0);

        // ── Load dY tile → dY_smem (batch × out) ─────────────────────
        // Each thread: b = tid / kM (0..15), r = tid % kM (0..15)
        {
            int b = tid / kM;
            int r = tid % kM;
            half val = __float2half(0.0f);
            if (b < tile_b) {
                int gb = b0 + b;
                int gr = r0 + r;
                if (gb < batch_size && gr < out_features) {
                    val = dY[gb * out_features + gr];
                }
            }
            dY_smem[b][r] = val;
        }

        // ── Load X tile → X_smem (batch × in) ────────────────────────
        // Each thread: b = tid / kN (0..15), c = tid % kN (0..15)
        {
            int b = tid / kN;
            int c = tid % kN;
            half val = __float2half(0.0f);
            if (b < tile_b) {
                int gb = b0 + b;
                int gc = c0 + c;
                if (gb < batch_size && gc < in_features) {
                    val = X[gb * in_features + gc];
                }
            }
            X_smem[b][c] = val;
        }
        __syncthreads();

        // ── Load WMMA fragments from SMEM ─────────────────────────────
        // dY_smem[b][r] loaded as col_major → a_frag[r][b]
        // col_major stride = kM = 16:  element (r, b) at base + r + b*16
        // dY_smem[b][r] is at base + b*16 + r = base + r + b*16  ✓
        wmma::load_matrix_sync(a_frag, &dY_smem[0][0], kM);

        // X_smem[b][c] loaded as row_major → b_frag[b][c]
        // row_major stride = kN = 16:  element (b, c) at base + b*16 + c
        // X_smem[b][c] is at base + b*16 + c  ✓
        wmma::load_matrix_sync(b_frag, &X_smem[0][0], kN);

        // ── Tensor-core matmul: dW[r][c] += Σ_b dY[b][r] * X[b][c] ──
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // ── Store accumulator to shared memory ────────────────────────────
    wmma::store_matrix_sync(&dW_float_smem[0][0], c_frag, kN, wmma::mem_row_major);
    __syncthreads();

    // ── Apply sign → counter → bit-flip for each weight in tile ──────
    int r = tid / kN;    // 0..15, row within tile (out_features)
    int c = tid % kN;    // 0..15, col within tile (in_features)
    int gr = r0 + r;     // global out_feature index
    int gc = c0 + c;     // global in_feature index

    if (gr < out_features && gc < in_features) {
        float grad = dW_float_smem[r][c];

        int idx = gr * in_features + gc;
        int16_t cnt = counter[idx];

        // Gradient descent: positive dW → decrease weight → decrement counter
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

    dim3 block(256);      // 1D block of 256 threads
    dim3 grid((in_features + kN - 1) / kN,
              (out_features + kM - 1) / kM);

    packed_ternary_update_tc_kernel<<<grid, block, 0, stream>>>(
        X, dY, W, counter, batch_size, in_features, out_features,
        stride_words, threshold
    );
}
