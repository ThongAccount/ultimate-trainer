/**
 * gemm_fused.cu - Fused Forward + Backward + Update (CUDA core, packed ternary).
 *
 * One kernel launch for the full discrete optimizer step:
 *   1. Forward:  Y = X @ W^T  (packed ternary, no unpack)
 *   2. Loss:     dY = Y - Y_target (MSE gradient)
 *   3. Update:   dW = dY^T @ X -> sign -> counter -> flip
 *
 * Grid:  (ceil(batch/32), ceil(out_features/32))
 * Block: 256 threads (32x8)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "packed_ternary.cuh"

__global__ __launch_bounds__(256) void packed_ternary_fused_step_kernel(
    uint32_t*       __restrict__ W,          // [N, stride] packed ternary
    int16_t*        __restrict__ counter,     // [N, K] int16 counters
    const half*     __restrict__ X,           // [B, K] input
    const half*     __restrict__ Y_target,    // [B, N] target
    half*           __restrict__ Y_out,       // [B, N] output
    int B, int K, int N, int stride_words, int16_t threshold)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    // SMEM must be at block scope (all threads participate in barriers)
    __shared__ half X_tile[32][16];
    __shared__ float dW_smem[32][16];

    float y_acc = 0.0f;
    float dy_val = 0.0f;

    if (b < B && n < N) {
        // Phase 1: Forward -- Y[b,n] = SUM_k X[b,k] * W[n,k]
        for (int k0 = 0; k0 < K; k0 += 16) {
            int tile_k = min(16, K - k0);

            int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
            for (int i = linear_tid; i < blockDim.x * tile_k; i += 256) {
                int lb = i / tile_k;
                int lk = i % tile_k;
                int gb = blockIdx.x * blockDim.x + lb;
                X_tile[lb][lk] = (gb < B) ? X[gb * K + k0 + lk] : __float2half(0.0f);
            }
            __syncthreads();

            if (k0 / 16 < stride_words) {
                uint32_t word = W[n * stride_words + k0 / 16];
                #pragma unroll 16
                for (int i = 0; i < 16 && i < tile_k; i++) {
                    int bits = (word >> (2 * i)) & 3;
                    int sign = (bits == 1) - (bits == 2);
                    if (sign != 0) {
                        y_acc += sign * __half2float(X_tile[threadIdx.x][i]);
                    }
                }
            }
            __syncthreads();
        }

        Y_out[b * N + n] = __float2half(y_acc);

        // Phase 2: Loss gradient -- dY = (Y - target) * 2/N
        dy_val = (y_acc - __half2float(Y_target[b * N + n])) * (2.0f / N);
    }

    // Phase 3: Counter update -- dW = SUM_b dY[b,n] * X[b,k]
    for (int k0 = 0; k0 < K; k0 += 16) {
        int tile_k = min(16, K - k0);

        float partial_dw = 0.0f;
        if (b < B && n < N) {
            for (int i = 0; i < tile_k; i++) {
                float x_val = __half2float(X[b * K + k0 + i]);
                partial_dw += dy_val * x_val;
            }
        }

        int n_local = threadIdx.y;
        int k_local = threadIdx.x % tile_k;
        if (threadIdx.x < tile_k) {
            dW_smem[n_local][k_local] = 0.0f;
        }
        __syncthreads();

        if (b < B && n < N && threadIdx.x < tile_k) {
            atomicAdd(&dW_smem[n_local][k_local], partial_dw);
        }
        __syncthreads();

        if (threadIdx.x < tile_k && n < N && k0 + threadIdx.x < K) {
            float dw = dW_smem[n_local][k_local];
            int idx = n * K + k0 + threadIdx.x;
            int16_t cnt = counter[idx];

            if (dw > 0.0f) cnt--;
            else if (dw < 0.0f) cnt++;

            uint32_t* w_row = W + n * stride_words;
            int kc = k0 + threadIdx.x;

            if (cnt > threshold) {
                increment_weight_atomic(w_row, kc);
                cnt = 0;
            } else if (cnt < -threshold) {
                decrement_weight_atomic(w_row, kc);
                cnt = 0;
            }

            counter[idx] = cnt;
        }
        __syncthreads();
    }
}


extern "C" void launch_packed_ternary_fused_step(
    uint32_t*       W,
    int16_t*        counter,
    const void*     X_ptr,
    const void*     Y_target_ptr,
    void*           Y_out_ptr,
    int batch_size,
    int in_features,
    int out_features,
    int stride_words,
    int16_t threshold,
    cudaStream_t stream)
{
    const half* X = static_cast<const half*>(X_ptr);
    const half* Y_target = static_cast<const half*>(Y_target_ptr);
    half* Y_out = static_cast<half*>(Y_out_ptr);

    dim3 block(32, 8);
    dim3 grid((batch_size + block.x - 1) / block.x,
              (out_features + block.y - 1) / block.y);

    packed_ternary_fused_step_kernel<<<grid, block, 0, stream>>>(
        W, counter, X, Y_target, Y_out,
        batch_size, in_features, out_features, stride_words, threshold
    );
}
