/* Scratch experiment: does forward become faster if W arrives pre-decoded fp16?
 * Copy of packed_ternary_forward_tc_64_kernel with the W load loop changed to
 * read half W[gn*K+gk] directly (no decode, no __int2half_rn). Everything else
 * identical. Bench vs packed kernel in the same process.
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "packed_ternary.cuh"
#include <mma.h>

namespace wmma = nvcuda::wmma;

constexpr int kWMMA_M = 16;
constexpr int kWMMA_N = 16;
constexpr int kWMMA_K = 16;
constexpr int kSuperM = 64;
constexpr int kSuperN = 64;
constexpr int kWarps  = 4;
constexpr int kFragsPerWarp = 4;

__global__ __launch_bounds__(128) void packed_ternary_forward_tc_64_fp16w_kernel(
    const half*     __restrict__ W,   // [N, K] pre-decoded FP16
    const half*     __restrict__ X,   // [B, K] FP16
    half*           __restrict__ Y,   // [B, N] FP16
    int B, int K, int N)
{
    int super_b0 = blockIdx.x * kSuperM;
    int super_n0 = blockIdx.y * kSuperN;

    int warp_id = threadIdx.x / 32;
    int warp_b_off = (warp_id / 2) * 32;
    int warp_n_off = (warp_id % 2) * 32;

    __shared__ half W_smem[kWMMA_K][kSuperN];
    __shared__ half X_smem[kSuperM][kWMMA_K];
    __shared__ float spill[kWarps][kWMMA_M * kWMMA_N];

    wmma::fragment<wmma::matrix_a, kWMMA_M, kWMMA_N, kWMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kWMMA_M, kWMMA_N, kWMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, kWMMA_M, kWMMA_N, kWMMA_K, float> c_frag[kFragsPerWarp];

    #pragma unroll
    for (int f = 0; f < kFragsPerWarp; ++f)
        wmma::fill_fragment(c_frag[f], 0.0f);

    for (int k0 = 0; k0 < K; k0 += kWMMA_K) {
        {
            int n_total = kSuperN * kWMMA_K;
            for (int tid = threadIdx.x; tid < n_total; tid += 128) {
                int r = tid / kWMMA_K;
                int c = tid % kWMMA_K;
                int gn = super_n0 + r;
                int gk = k0 + c;
                float v = (gn < N && gk < K) ? __half2float(W[gn * K + gk]) : 0.0f;
                W_smem[c][r] = __float2half(v);
            }
        }
        {
            int n_total = kSuperM * kWMMA_K;
            for (int tid = threadIdx.x; tid < n_total; tid += 128) {
                int r = tid / kWMMA_K;
                int c = tid % kWMMA_K;
                int gb = super_b0 + r;
                int gk = k0 + c;
                X_smem[r][c] = (gb < B && gk < K) ? X[gb * K + gk] : __float2half(0.0f);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int fi = 0; fi < kFragsPerWarp; ++fi) {
            int frag_b_off = (fi / 2) * kWMMA_M;
            int frag_n_off = (fi % 2) * kWMMA_N;
            int b_base = warp_b_off + frag_b_off;
            int n_base = warp_n_off + frag_n_off;
            wmma::load_matrix_sync(a_frag, &X_smem[b_base][0], kWMMA_K);
            wmma::load_matrix_sync(b_frag, &W_smem[0][n_base], kSuperN);
            wmma::mma_sync(c_frag[fi], a_frag, b_frag, c_frag[fi]);
        }
        __syncthreads();
    }

    for (int fi = 0; fi < kFragsPerWarp; ++fi) {
        int frag_b_off = (fi / 2) * kWMMA_M;
        int frag_n_off = (fi % 2) * kWMMA_N;
        wmma::store_matrix_sync(&spill[warp_id][0], c_frag[fi], kWMMA_N, wmma::mem_row_major);
        __syncthreads();
        int n_elems = kWMMA_M * kWMMA_N;
        for (int tid = threadIdx.x; tid < n_elems * kWarps; tid += 128) {
            int slot_w = tid / n_elems;
            int off    = tid % n_elems;
            int r = off / kWMMA_N;
            int c = off % kWMMA_N;
            int gb = super_b0 + (slot_w / 2) * 32 + frag_b_off + r;
            int gn = super_n0 + (slot_w % 2) * 32 + frag_n_off + c;
            if (gb < B && gn < N)
                Y[gb * N + gn] = __float2half_rn(spill[slot_w][off]);
        }
        __syncthreads();
    }
}

extern "C" void launch_forward_tc_64_fp16w(
    const void* W_ptr, const void* X_ptr, void* Y_ptr,
    int batch_size, int in_features, int out_features, cudaStream_t stream)
{
    const half* W = static_cast<const half*>(W_ptr);
    const half* X = static_cast<const half*>(X_ptr);
    half* Y = static_cast<half*>(Y_ptr);
    dim3 grid((batch_size + kSuperM - 1) / kSuperM,
              (out_features + kSuperN - 1) / kSuperN);
    dim3 block(128);
    packed_ternary_forward_tc_64_fp16w_kernel<<<grid, block, 0, stream>>>(
        W, X, Y, batch_size, in_features, out_features);
}