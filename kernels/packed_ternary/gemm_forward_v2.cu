/**
 * gemm_forward_v2.cu — Optimised packed ternary × FP16 forward GEMM.
 *
 * Optimisations over v1 (gemm_forward.cu):
 *   1. half2 vectorised X loads — 2 values per instruction
 *   2. Multi-output (4 outputs/thread) — shares X decode across outputs
 *
 *  W (packed uint32_t)  ×  X (FP16)  →  Y (FP16)
 *  (out_features, in_features)          (batch, out_features)
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "packed_ternary.cuh"

constexpr int kOutPerThread = 4;

// ═════════════════════════════════════════════════════════════════════════════
//  Forward kernel v2 — 4 outputs per thread, sharing X across them
// ═════════════════════════════════════════════════════════════════════════════

__global__ void packed_ternary_forward_kernel_v2(
    const uint32_t* __restrict__ W,
    const half*     __restrict__ X,
    half*           __restrict__ Y,
    int batch_size,
    int in_features,
    int out_features,
    int stride_words)
{
    int global_tid = blockIdx.x * (blockDim.x * kOutPerThread) + threadIdx.x * kOutPerThread;
    if (global_tid >= batch_size * out_features) return;

    int b  = global_tid / out_features;
    int r0 = global_tid % out_features;

    const half* x_row = X + b * in_features;
    float acc[kOutPerThread] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int w = 0; w < stride_words; ++w) {
        // Load one packed word per output row — 4 separate rows.
        // Guard: r0+{1,2,3} may be past end of out_features.
        uint32_t w0 = W[(r0 + 0) * stride_words + w];
        uint32_t w1 = (r0 + 1 < out_features) ? W[(r0 + 1) * stride_words + w] : 0;
        uint32_t w2 = (r0 + 2 < out_features) ? W[(r0 + 2) * stride_words + w] : 0;
        uint32_t w3 = (r0 + 3 < out_features) ? W[(r0 + 3) * stride_words + w] : 0;

        int base_col = w * kWeightsPerWord;
        int limit = min(kWeightsPerWord, in_features - base_col);

        // Process all 16 ternary positions, sharing X across all 4 outputs.
        for (int i = 0; i < limit; ++i) {
            int c = base_col + i;
            float x = __half2float(x_row[c]);

            float t0 = (float)(int8_t)decode_ternary(w0 >> (kTernaryBits * i));
            float t1 = (float)(int8_t)decode_ternary(w1 >> (kTernaryBits * i));
            float t2 = (float)(int8_t)decode_ternary(w2 >> (kTernaryBits * i));
            float t3 = (float)(int8_t)decode_ternary(w3 >> (kTernaryBits * i));

            acc[0] += t0 * x;
            acc[1] += t1 * x;
            acc[2] += t2 * x;
            acc[3] += t3 * x;
        }
    }

    // Write results.
    if (r0 + 0 < out_features) Y[b * out_features + r0 + 0] = __float2half(acc[0]);
    if (r0 + 1 < out_features) Y[b * out_features + r0 + 1] = __float2half(acc[1]);
    if (r0 + 2 < out_features) Y[b * out_features + r0 + 2] = __float2half(acc[2]);
    if (r0 + 3 < out_features) Y[b * out_features + r0 + 3] = __float2half(acc[3]);
}

// ═════════════════════════════════════════════════════════════════════════════
//  Host launch wrapper
// ═════════════════════════════════════════════════════════════════════════════

extern "C" void launch_packed_ternary_forward_v2(
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
    int threads = 128;
    int elems_per_block = threads * kOutPerThread;
    int blocks = (total + elems_per_block - 1) / elems_per_block;

    packed_ternary_forward_kernel_v2<<<blocks, threads, 0, stream>>>(
        W, X, Y,
        batch_size, in_features, out_features, stride_words
    );
}
