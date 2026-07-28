/**
 * gemm_forward_packed.cu — CUDA core ternary GEMM (no unpack to FP16).
 *
 * Processes packed uint32 weights directly: 16 ternary values per word.
 * For each output element, iterates over K weights, decoding 2-bit values
 * and conditionally accumulating FP16 inputs.
 *
 * Optimizations:
 *   - No unpack to FP16 (reads packed uint32 directly)
 *   - 32× less W memory traffic (2-bit vs 16-bit)
 *   - Double-buffered SMEM for X tiles (overlaps load + compute)
 *   - Branchless decode: sign = (bits==1) - (bits==2)
 *
 * Grid:  (ceil(batch/32), ceil(out_features/32))
 * Block: 256 threads (32×8, each thread = 1 output element)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "packed_ternary.cuh"

__global__ __launch_bounds__(256) void packed_ternary_forward_packed_kernel(
    const uint32_t* __restrict__ W,   // (out_features, stride_words) packed
    const half*     __restrict__ X,   // (batch, in_features) FP16
    half*           __restrict__ Y,   // (batch, out_features) FP16
    int batch_size,
    int in_features,
    int out_features,
    int stride_words)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && n < out_features) {
        // ── Kernel body ─────────────────────────────────────────────

        // Double-buffered SMEM: load next tile while computing current
        __shared__ half X_tile[2][32][16];

        float acc = 0.0f;
        int b_local = threadIdx.x;
        int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;

        // Load first tile (buffer 0)
        {
            int tile_k = min(16, in_features);
            int total_elements = blockDim.x * tile_k;
            for (int i = linear_tid; i < total_elements; i += 256) {
                int lb = i / tile_k;
                int lk = i % tile_k;
                int gb = blockIdx.x * blockDim.x + lb;
                X_tile[0][lb][lk] = (gb < batch_size) ? X[gb * in_features + lk] : __float2half(0.0f);
            }
        }
        __syncthreads();

        int load_buf = 1;
        int compute_buf = 0;

        // Main loop: compute current tile, load next tile
        for (int k0 = 0; k0 < in_features; k0 += 16) {
            int tile_k = min(16, in_features - k0);

            // Asynchronously load NEXT tile into load_buf
            if (k0 + 16 < in_features) {
                int next_k = k0 + 16;
                int next_tile_k = min(16, in_features - next_k);
                int total_elements = blockDim.x * next_tile_k;
                for (int i = linear_tid; i < total_elements; i += 256) {
                    int lb = i / next_tile_k;
                    int lk = i % next_tile_k;
                    int gb = blockIdx.x * blockDim.x + lb;
                    X_tile[load_buf][lb][lk] = (gb < batch_size)
                        ? X[gb * in_features + next_k + lk]
                        : __float2half(0.0f);
                }
            }

            // Compute CURRENT tile
            if (k0 / 16 < stride_words) {
                uint32_t word = W[n * stride_words + k0 / 16];
                #pragma unroll 16
                for (int i = 0; i < 16 && i < tile_k; i++) {
                    int bits = (word >> (2 * i)) & 3;
                    int sign = (bits == 1) - (bits == 2);
                    if (sign != 0) {
                        acc += sign * __half2float(X_tile[compute_buf][b_local][i]);
                    }
                }
            }

            __syncthreads();

            // Swap buffers
            load_buf ^= 1;
            compute_buf ^= 1;
        }

        // Store result
        Y[b * in_features + out_features * n + 0] = __float2half(acc);
    }
}


extern "C" void launch_packed_ternary_forward_packed(
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

    dim3 block(32, 8);
    dim3 grid((batch_size + block.x - 1) / block.x,
              (out_features + block.y - 1) / block.y);

    packed_ternary_forward_packed_kernel<<<grid, block, 0, stream>>>(
        W, X, Y, batch_size, in_features, out_features, stride_words
    );
}
