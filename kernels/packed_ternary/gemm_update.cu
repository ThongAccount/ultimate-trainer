/**
 * gemm_update.cu — Fused gradient → counter → bit-flip.
 *
 * This is the core research contribution of the discrete optimisation stack.
 *
 * For each weight w[r][c]:
 *   1. Compute dW[r][c] = Σ_b dY[b][r] * X[b][c]   (in registers, never stored)
 *   2. sign = (dW > 0) - (dW < 0)
 *   3. counter[idx] += sign
 *   4. If |counter[idx]| > threshold: flip the ternary bit, reset counter
 *
 * No dW tensor is ever materialised in global memory.
 *
 * Grid:  (ceil(out_features / 256), 1)
 * Block: 256 threads, each handling one output row
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "packed_ternary.cuh"

__global__ void packed_ternary_update_kernel(
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
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= out_features) return;

    uint32_t* w_row = W + r * stride_words;
    int16_t*  cnt_row = counter + r * in_features;

    for (int wi = 0; wi < stride_words; wi++) {
        uint32_t word = w_row[wi];
        int base = wi * kWeightsPerWord;
        int limit = min(kWeightsPerWord, in_features - base);

        for (int i = 0; i < limit; i++) {
            int c = base + i;

            // ── Compute dW[r][c] = Σ_b dY[b][r] * X[b][c] ──────────
            float grad = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                grad += __half2float(dY[b * out_features + r]) *
                        __half2float(X[b * in_features + c]);
            }

            // ── sign → counter → flip ─────────────────────────────
            if (grad > 0.0f)       cnt_row[c]++;
            else if (grad < 0.0f)  cnt_row[c]--;

            if (cnt_row[c] > threshold) {
                increment_weight(w_row, c);
                cnt_row[c] = 0;
            } else if (cnt_row[c] < -threshold) {
                decrement_weight(w_row, c);
                cnt_row[c] = 0;
            }
        }
    }
}

extern "C" void launch_packed_ternary_update(
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
    int threads = 256;
    int blocks = (out_features + threads - 1) / threads;
    packed_ternary_update_kernel<<<blocks, threads, 0, stream>>>(
        X, dY, W, counter, batch_size, in_features, out_features,
        stride_words, threshold
    );
}
