/**
 * gemm_forward.cu — Packed ternary × FP16 forward GEMM.
 *
 *  W (packed uint32_t)  ×  X (FP16)  →  Y (FP16)
 *  (out_features, in_features)          (batch, out_features)
 *
 * Correctness-first: each thread computes one output element.
 * No shared memory or tiling — those come in the optimization pass.
 *
 * Encoding: 00=0, 01=+1, 10=-1, 11=INVALID (treated as 0)
 * 16 ternary values per uint32_t.
 */

#include <cuda_runtime.h>
#include <cstdint>

// ── Include the packed ternary header ───────────────────────────────────────
#include "packed_ternary.cuh"

// ═════════════════════════════════════════════════════════════════════════════
//  Forward kernel
// ═════════════════════════════════════════════════════════════════════════════

__global__ void packed_ternary_forward_kernel(
    const uint32_t* __restrict__ W,      // (out_features, stride_words) packed
    const half*     __restrict__ X,      // (batch, in_features)
    half*           __restrict__ Y,      // (batch, out_features)
    int batch_size,
    int in_features,
    int out_features,
    int stride_words)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int b = idx / out_features;   // batch index
    int r = idx % out_features;   // output row (feature)

    const uint32_t* w_row = W + r * stride_words;
    const half*     x_row = X + b * in_features;

    float acc = 0.0f;

    // Iterate over packed words.
    for (int w = 0; w < stride_words; ++w) {
        uint32_t word = w_row[w];
        int base_col = w * kWeightsPerWord;

        // Manual unrolling for correctness — compiler will optimise later.
        #pragma unroll
        for (int i = 0; i < kWeightsPerWord; ++i) {
            int c = base_col + i;
            if (c >= in_features) break;

            int8_t w_val = decode_ternary(word >> (kTernaryBits * i));
            if (w_val == 1) {
                acc += __half2float(x_row[c]);
            } else if (w_val == -1) {
                acc -= __half2float(x_row[c]);
            }
            // w_val == 0 → no-op
        }
    }

    Y[idx] = __float2half(acc);
}

// ═════════════════════════════════════════════════════════════════════════════
//  Host launch wrapper (extern "C" for PyTorch load_inline)
// ═════════════════════════════════════════════════════════════════════════════

extern "C" void launch_packed_ternary_forward(
    const uint32_t* W,
    const void*     X_ptr,
    void*           Y_ptr,
    int batch_size,
    int in_features,
    int out_features,
    int stride_words,
    cudaStream_t stream)

// Cast void* back to half* inside the launch wrapper — the caller uses
// void* to avoid exposing the CUDA half type to host-side C++.
{
    const half* X = static_cast<const half*>(X_ptr);
    half*       Y = static_cast<half*>(Y_ptr);
{
    int total = batch_size * out_features;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    packed_ternary_forward_kernel<<<blocks, threads, 0, stream>>>(
        W, X, Y,
        batch_size, in_features, out_features, stride_words
    );
}
