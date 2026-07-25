/**
 * gemm_forward_packed.cu — CUDA core ternary GEMM (no unpack to FP16).
 *
 * Processes packed uint32 weights directly: 16 ternary values per word.
 * For each output element, iterates over K weights, decoding 2-bit values
 * and conditionally accumulating FP16 inputs.
 *
 * Why this instead of WMMA:
 *   - WMMA requires unpacking ternary→FP16 in SMEM (~0.3ms overhead)
 *   - This kernel reads packed weights directly from registers
 *   - 32× less W memory traffic (2-bit vs 16-bit)
 *   - No SMEM needed for W tiles
 *   - Compute is ~10× slower than TC but we're memory-bound anyway
 *
 * Grid:  (ceil(batch/32), ceil(out_features/32))
 * Block: 256 threads (8×32 thread block, each thread = 1 output element)
 */

#include <cuda_runtime.h>
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
    // Each thread computes one output element Y[b, n]
    int b = blockIdx.x * blockDim.x + threadIdx.x;  // batch index
    int n = blockIdx.y * blockDim.y + threadIdx.y;   // output feature

    if (b >= batch_size || n >= out_features) return;

    // Shared memory for X tile (reused across all output features in block)
    // Tile size: blockDim.x × 16 = 256 × 16 = 4096 half = 8 KB
    __shared__ half X_tile[256][16];

    float acc = 0.0f;

    // Iterate over K in chunks of 16 (one uint32 per chunk)
    for (int k0 = 0; k0 < in_features; k0 += 16) {
        int tile_k = min(16, in_features - k0);

        // Cooperative load: all threads load X tile into SMEM
        // 256 threads × 16 elements = 4096 elements, but we only need
        // blockDim.x × tile_k elements
        for (int i = threadIdx.x; i < blockDim.x * tile_k; i += blockDim.x) {
            int lb = i / tile_k;  // local batch index
            int lk = i % tile_k;  // local k index
            int gb = blockIdx.x * blockDim.x + lb;
            if (gb < batch_size) {
                X_tile[lb][lk] = X[gb * in_features + k0 + lk];
            }
        }
        __syncthreads();

        // Load packed weight word for this output feature
        if (k0 / 16 < stride_words) {
            uint32_t word = W[n * stride_words + k0 / 16];

            // Decode 16 ternary values and accumulate
            #pragma unroll 16
            for (int i = 0; i < 16 && i < tile_k; i++) {
                int bits = (word >> (2 * i)) & 3;
                // Branchless: +1 if bits==1, -1 if bits==2, 0 otherwise
                int sign = (bits == 1) - (bits == 2);
                if (sign != 0 && b < batch_size) {
                    acc += sign * __half2float(X_tile[threadIdx.x][i]);
                }
            }
        }

        __syncthreads();
    }

    // Store result
    Y[b * out_features + n] = __float2half(acc);
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

    dim3 block(32, 8);  // 256 threads, 32 batch × 8 features per block
    dim3 grid((batch_size + block.x - 1) / block.x,
              (out_features + block.y - 1) / block.y);

    packed_ternary_forward_packed_kernel<<<grid, block, 0, stream>>>(
        W, X, Y, batch_size, in_features, out_features, stride_words
    );
}
