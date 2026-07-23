/**
 * gemm_update.cu — Fused gradient → counter → bit-flip (Parallel 2D Grid).
 *
 * Parallelised across 2D grid of (in_features, out_features).
 * Each thread handles one weight (r, c), accumulating gradient over batch B.
 *
 * For each weight w[r][c]:
 *   1. Compute dW[r][c] = Σ_b dY[b][r] * X[b][c]   (in registers, never stored)
 *   2. sign = (dW > 0) - (dW < 0)
 *   3. counter[r][c] += sign
 *   4. If |counter[r][c]| > threshold: flip ternary bit atomically, reset counter
 *
 * No dW tensor is ever materialised in global memory.
 *
 * Grid:  (ceil(in_features / 16), ceil(out_features / 16))
 * Block: (16, 16) = 256 threads
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
    int c = blockIdx.x * blockDim.x + threadIdx.x; // input feature column
    int r = blockIdx.y * blockDim.y + threadIdx.y; // output feature row

    if (r >= out_features || c >= in_features) return;

    // ── Compute dW[r][c] = Σ_b dY[b][r] * X[b][c] ──────────
    float grad = 0.0f;
    #pragma unroll 4
    for (int b = 0; b < batch_size; b++) {
        grad += __half2float(dY[b * out_features + r]) *
                __half2float(X[b * in_features + c]);
    }

    // ── sign → counter → flip ─────────────────────────────
    int idx = r * in_features + c;
    int16_t cnt = counter[idx];

    // Gradient descent: positive dW → decrease weight → decrement counter
    if (grad > 0.0f)       cnt--;
    else if (grad < 0.0f)  cnt++;

    uint32_t* w_row = W + r * stride_words;

    if (cnt > threshold) {
        increment_weight_atomic(w_row, c);   // counter went + → increase weight
        cnt = 0;
    } else if (cnt < -threshold) {
        decrement_weight_atomic(w_row, c);   // counter went - → decrease weight
        cnt = 0;
    }

    counter[idx] = cnt;
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

    dim3 block(16, 16);
    dim3 grid((in_features + block.x - 1) / block.x,
              (out_features + block.y - 1) / block.y);

    packed_ternary_update_kernel<<<grid, block, 0, stream>>>(
        X, dY, W, counter, batch_size, in_features, out_features,
        stride_words, threshold
    );
}
