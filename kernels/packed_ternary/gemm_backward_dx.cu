/**
 * gemm_backward_dx.cu — Packed ternary backward: dX = W^T @ dY.
 *
 * Same GEMM as forward, just transposed: each thread accumulates
 * dX[b][c] = Σ_r W[r][c] * dY[b][r] over output rows r.
 * W is ternary {-1,0,+1} stored in the same packed format.
 *
 * Grid: (ceil(in_features / 256), 1) — one block per feature column
 * Block: 256 threads — each thread handles one feature column
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "packed_ternary.cuh"

__global__ void packed_ternary_backward_dx_kernel(
    const uint32_t* __restrict__ W,   // (out_features, stride_words) packed ternary
    const half*     __restrict__ dY,  // (batch, out_features)
    half*           __restrict__ dX,  // (batch, in_features)
    int batch_size,
    int in_features,
    int out_features,
    int stride_words)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= in_features) return;

    int wi = c / kWeightsPerWord;
    int pos = c % kWeightsPerWord;

    // Each thread handles all output rows for one input column c.
    // dX[b][c] = Σ_r W[r][c] * dY[b][r]
    for (int b = 0; b < batch_size; b++) {
        float acc = 0.0f;
        for (int r = 0; r < out_features; r++) {
            uint32_t word = W[r * stride_words + wi];
            int8_t t = decode_ternary(word >> (kTernaryBits * pos));
            if (t != 0) {
                acc += (float)t * __half2float(dY[b * out_features + r]);
            }
        }
        dX[b * in_features + c] = __float2half(acc);
    }
}

extern "C" void launch_packed_ternary_backward_dx(
    const uint32_t* W,
    const void*     dY_ptr,
    void*           dX_ptr,
    int batch_size,
    int in_features,
    int out_features,
    int stride_words,
    cudaStream_t stream)
{
    const half* dY = static_cast<const half*>(dY_ptr);
    half*       dX = static_cast<half*>(dX_ptr);
    int threads = 256;
    int blocks = (in_features + threads - 1) / threads;
    packed_ternary_backward_dx_kernel<<<blocks, threads, 0, stream>>>(
        W, dY, dX, batch_size, in_features, out_features, stride_words
    );
}
