/**
 * gemm_forward_v4.cu — Half-arithmetic packed ternary × FP16 forward GEMM.
 *
 * v2 peaks at ~19 GFLOPS on a T4 (0.25% utilisation). The bottleneck is
 * the inner loop doing half→float→add→float→half conversion every iteration.
 *
 * Fix from v2: accumulate in FP16 using __hadd / __hsub instead of float.
 * This eliminates 3 of ~10 instructions per inner iteration.
 *
 *  W (packed uint32_t)  ×  X (FP16)  →  Y (FP16)
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "packed_ternary.cuh"

__global__ void packed_ternary_forward_kernel_v4(
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

    half acc = __float2half(0.0f);

    for (int w = 0; w < stride_words; ++w) {
        uint32_t word = w_row[w];
        int base = w * kWeightsPerWord;
        int limit = min(kWeightsPerWord, in_features - base);

        for (int i = 0; i < limit; ++i) {
            int c = base + i;
            int8_t t = decode_ternary(word >> (kTernaryBits * i));
            if (t == 1)       acc = __hadd(acc, x_row[c]);
            else if (t == -1) acc = __hsub(acc, x_row[c]);
        }
    }

    Y[idx] = acc;
}

extern "C" void launch_packed_ternary_forward_v4(
    const uint32_t* W, const void* X_ptr, void* Y_ptr,
    int batch_size, int in_features, int out_features,
    int stride_words, cudaStream_t stream)
{
    const half* X = static_cast<const half*>(X_ptr);
    half*       Y = static_cast<half*>(Y_ptr);
    int total = batch_size * out_features;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    packed_ternary_forward_kernel_v4<<<blocks, threads, 0, stream>>>(
        W, X, Y, batch_size, in_features, out_features, stride_words
    );
}
