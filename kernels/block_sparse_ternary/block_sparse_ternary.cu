// block_sparse_ternary.cu
// Ternary matmul with block-skip bitmask.
// When block_mask[tile_n * num_k_tiles + tile_k] bit is 0, skip that K-tile
// entirely and leave the output tile at zero.
//
// Grid:  (ceil(N/BN), ceil(M/BM))
// Block: 16x16 threads (BM=BN=16), one output element per thread.
// Shared: x_tile[16][16], w_tile[16][16] loaded per K-chunk of BK=16.
//
// Output y is fp32; x input is fp32 (wrapper converts); weights fp32 (master).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define TILE 16

__global__ void block_sparse_ternary_kernel(
    const float* __restrict__ x_ptr,
    const float* __restrict__ w_ptr,
    float* __restrict__ y_ptr,
    const uint64_t* __restrict__ block_mask,
    float gamma,
    int M, int N, int K,
    int BM, int BN, int BK,
    int num_k_tiles
) {
    const int tx = threadIdx.x;          // 0..15 (N within tile)
    const int ty = threadIdx.y;          // 0..15 (M within tile)
    const int pid_m = blockIdx.x * BM;
    const int pid_n = blockIdx.y * BN;
    if (pid_m >= M || pid_n >= N) return;

    const int row = pid_m + ty;          // global M index
    const int col = pid_n + tx;          // global N index

    __shared__ float x_tile[TILE][TILE];  // [k][m]
    __shared__ float w_tile[TILE][TILE];  // [k][n]

    float acc = 0.0f;
    const int tid = ty * TILE + tx;
    const int total = TILE * TILE;

    for (int tk = 0; tk < num_k_tiles; tk++) {
        const int block_bit = (pid_n / BN) * num_k_tiles + tk;
        const int word_idx = block_bit >> 6;
        const int bit_idx = block_bit & 63;
        const bool active = (block_mask[word_idx] >> bit_idx) & 1ULL;

        if (!active) continue;  // whole K-chunk masked out

        // Cooperative load of the K-chunk (BK rows).
        for (int kk = tid; kk < BK; kk += total) {
            const int k = tk * BK + kk;
            if (k < K) {
                // x: x[row][k] -> x_tile[kk][ty]
                if (row < M) x_tile[kk][ty] = x_ptr[(long)row * K + k];
                // w: w[col][k] -> w_tile[kk][tx]
                if (col < N) w_tile[kk][tx] = w_ptr[(long)col * K + k];
            }
        }
        __syncthreads();

        // Accumulate this K-chunk. Ternary weights: +1 if w/gamma > 0.5,
        // -1 if < -0.5, else 0. All threads read the same (kk) rows.
        for (int kk = 0; kk < BK && tk * BK + kk < K; kk++) {
            const float x_val = x_tile[kk][ty];
            const float w_val = w_tile[kk][tx] / gamma;
            if (w_val > 0.5f) acc += x_val;
            else if (w_val < -0.5f) acc -= x_val;
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        y_ptr[(long)row * N + col] = acc;
    }
}
