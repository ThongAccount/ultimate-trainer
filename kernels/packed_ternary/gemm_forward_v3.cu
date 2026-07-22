/**
 * gemm_forward_v3.cu — Occupancy-optimised packed ternary × FP16 forward GEMM.
 *
 * Observation from v2: at 128 threads × 4 outputs/thread, only 8-64 blocks
 * are launched on a 20-SM T4.  Occupancy is ~2-20%.
 *
 * Fix: 1 output per thread, 256 threads/block → 2-4× more blocks.
 * Each thread reads its single W row independently, but the GPU hides
 * latency by having many more active warps.
 *
 *  W (packed uint32_t)  ×  X (FP16)  →  Y (FP16)
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "packed_ternary.cuh"

__global__ void packed_ternary_forward_kernel_v3(
    const uint32_t* __restrict__ W,
    const half*     __restrict__ X,
    half*           __restrict__ Y,
    int batch_size,
    int in_features,
    int out_features,
    int stride_words)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int b = idx / out_features;
    int r = idx % out_features;

    const uint32_t* w_row = W + r * stride_words;
    const half*     x_row = X + b * in_features;

    float acc = 0.0f;

    for (int w = 0; w < stride_words; ++w) {
        uint32_t word = w_row[w];
        int base = w * kWeightsPerWord;
        int limit = min(kWeightsPerWord, in_features - base);

        for (int i = 0; i < limit; ++i) {
            int c = base + i;
            int8_t t = decode_ternary(word >> (kTernaryBits * i));
            if (t == 1)       acc += __half2float(x_row[c]);
            else if (t == -1) acc -= __half2float(x_row[c]);
        }
    }

    Y[idx] = __float2half(acc);
}

extern "C" void launch_packed_ternary_forward_v3(
    const uint32_t* W,
    const void*     X_ptr,
    void*           Y_ptr,
    int batch_size,
    int in_features,
    int out_features,
    int stride_words,
    cudaStream_t stream)
{
    const half* X = static_cast<const half*>(X_ptr);
    half*       Y = static_cast<half*>(Y_ptr);

    int total = batch_size * out_features;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    packed_ternary_forward_kernel_v3<<<blocks, threads, 0, stream>>>(
        W, X, Y,
        batch_size, in_features, out_features, stride_words
    );
}
