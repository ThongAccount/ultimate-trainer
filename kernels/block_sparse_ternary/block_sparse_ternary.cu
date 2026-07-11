// block_sparse_ternary.cu
// Ternary matmul with block-skip bitmask. Extends dense ternary_matmul.cu.
// When block_mask[tile_n * num_k_tiles + tile_k] bit is 0, skip that K-tile
// entirely and leave the output tile at zero.
//
// Grid: (N/BN, M/BM)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

__global__ void block_sparse_ternary_kernel(
    const half* __restrict__ x_ptr,
    const float* __restrict__ w_ptr,
    half* __restrict__ y_ptr,
    const uint64_t* __restrict__ block_mask,
    float gamma,
    int M, int N, int K,
    int BM, int BN, int BK,
    int num_k_tiles
) {
    int pid_m = blockIdx.x * BM;
    int pid_n = blockIdx.y * BN;
    if (pid_m >= M || pid_n >= N) return;

    int tid = threadIdx.x;
    int total = blockDim.x;

    // Shared memory for tiles
    __shared__ float x_tile[32 * 32];  // BK x BM
    __shared__ float w_tile[32 * 32];  // BK x BN

    float acc[16][16] = {{0}};  // Accumulator for this tile

    for (int tk = 0; tk < num_k_tiles; tk++) {
        int block_bit = (pid_n / BN) * num_k_tiles + tk;
        int word_idx = block_bit / 64;
        int bit_idx = block_bit % 64;
        bool active = (block_mask[word_idx] >> bit_idx) & 1ULL;

        if (!active) continue;  // Skip zero block

        // Load and process this K-tile
        for (int kk = 0; kk < BK; kk += total) {
            int k = tk * BK + kk + tid;
            if (k >= K) break;
            for (int i = 0; i < BM && pid_m + i < M; i++) {
                x_tile[kk * BM + i] = __half2float(x_ptr[(pid_m + i) * K + k]);
            }
            for (int j = 0; j < BN && pid_n + j < N; j++) {
                w_tile[kk * BN + j] = w_ptr[(pid_n + j) * K + k];
            }
        }
        __syncthreads();

        // Compute ternary matmul for this tile
        for (int kk = 0; kk < BK; kk++) {
            for (int i = 0; i < BM && pid_m + i < M; i++) {
                float x_val = x_tile[kk * BM + i];
                for (int j = 0; j < BN && pid_n + j < N; j++) {
                    float w_val = w_tile[kk * BN + j] / gamma;
                    if (w_val > 0.5f) acc[i][j] += x_val;
                    else if (w_val < -0.5f) acc[i][j] -= x_val;
                    // |w_val| <= 0.5 => ternary 0, skip
                }
            }
        }
        __syncthreads();
    }

    // Write output
    for (int i = tid; i < BM && pid_m + i < M; i += total) {
        for (int j = 0; j < BN && pid_n + j < N; j++) {
            y_ptr[(pid_m + i) * N + pid_n + j] = __float2half(acc[i][j]);
        }
    }
}
