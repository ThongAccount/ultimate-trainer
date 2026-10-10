/**
 * tests/fwd_swizzle.cu — fwd TC64 rasterization-swizzle probe kernels (C13 GO #2).
 *
 * PROBE FILE — never imported by production dispatch. Compiled only by
 * tests/probe_fwd_swizzle.py (old-vs-new bit-exact parity + 10-trial median).
 *
 * Mechanism (sweep #13/#20): production launches grid=(m_tiles, n_tiles) with
 * blockIdx.x = m-tile as the FASTEST dimension. The hardware scheduler
 * rasterizes CTAs in linear order (x fastest), so co-resident CTAs span
 * DIFFERENT m-tiles of the SAME n-tile: each re-streams its own 128KB X tile
 * from DRAM (25.7GB of X traffic at the head shape) while the packed W slice
 * gets the L2 reuse. Swizzle so co-resident CTAs span different n-tiles of
 * the SAME m-tile: they all share one 128KB X tile (X DRAM 25.7GB -> ~32MB)
 * and their distinct packed-W slices (~1.9MB working set) fit in the 4MB L2.
 *
 * Each CTA still computes a disjoint 64x64 Y tile with byte-identical inner
 * math, so parity vs production must be BIT-EXACT; only the CTA->tile map
 * changes, and both maps below are bijections (probe asserts coverage on
 * CPU for several (M,N)).
 *
 * Two formulations, both bit-exact by construction:
 *
 *  A) `..._swizzle_kernel` — launch config UNCHANGED (grid=(m_tiles,n_tiles));
 *     the kernel derives (m,n) from the swizzled linear CTA id
 *       linear = blockIdx.y * gridDim.x + blockIdx.x   (hardware order)
 *       m_tile = linear / num_n_tiles   n_tile = linear % num_n_tiles
 *     so n varies FASTEST within the co-resident window. num_n_tiles is
 *     passed as a kernel arg (host computes ceil(N/64)). Costs one integer
 *     divide in the CTA prologue.
 *
 *  B) `..._xpose_kernel` — "transposed launch" fallback: host launches
 *     grid=(n_tiles, m_tiles) and the kernel simply reads n from blockIdx.x
 *     and b from blockIdx.y. Same n-fastest schedule, zero extra kernel
 *     instructions (resources bit-identical to production by construction).
 *
 * The probe times both; whichever wins with identical resources ships.
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
constexpr int kSuperM = 64;     // CTA batch tile
constexpr int kSuperN = 64;     // CTA feature tile
constexpr int kWarps  = 4;
constexpr int kFragsPerWarp = 4;  // 2x2 grid per warp

// ── Formulation A: swizzled linear id, launch config unchanged ───────────────

__global__ __launch_bounds__(128) void packed_ternary_forward_tc_64_swizzle_kernel(
    const uint32_t* __restrict__ W,   // [N, stride] packed
    const half*     __restrict__ X,   // [B, K] FP16
    half*           __restrict__ Y,   // [B, N] FP16
    int B, int K, int N, int stride_words, int num_n_tiles)
{
    // grid = (m_tiles, n_tiles), block = 128 — identical to production launch.
    int linear = blockIdx.y * gridDim.x + blockIdx.x;        // hw raster order
    // Unsigned divide: linear and num_n_tiles are non-negative; cheaper code.
    int m_tile = (int)((unsigned)linear / (unsigned)num_n_tiles);
    int n_tile = (int)((unsigned)linear % (unsigned)num_n_tiles);
    int super_b0 = m_tile * kSuperM;
    int super_n0 = n_tile * kSuperN;

    int warp_id = threadIdx.x / 32;
    int wtid    = threadIdx.x % 32;

    // Per-warp position within the 64x64 CTA tile
    // warp 0: rows 0-31, cols 0-31
    // warp 1: rows 0-31, cols 32-63
    // warp 2: rows 32-63, cols 0-31
    // warp 3: rows 32-63, cols 32-63
    int warp_b_off = (warp_id / 2) * 32;  // 0 or 32
    int warp_n_off = (warp_id % 2) * 32;  // 0 or 32

    // Shared memory (identical layout to production)
    __shared__ half W_smem[kSuperN][kWMMA_K];  // [64][16]
    __shared__ half X_smem[kSuperM][kWMMA_K];  // [64][16]

    wmma::fragment<wmma::matrix_a, kWMMA_M, kWMMA_N, kWMMA_K,
                   half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kWMMA_M, kWMMA_N, kWMMA_K,
                   half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, kWMMA_M, kWMMA_N, kWMMA_K,
                   float> c_frag[kFragsPerWarp];

    #pragma unroll
    for (int f = 0; f < kFragsPerWarp; ++f)
        wmma::fill_fragment(c_frag[f], 0.0f);

    // Outer loop over K in steps of 16
    for (int k0 = 0; k0 < K; k0 += kWMMA_K) {
        int tile_k = min(kWMMA_K, K - k0);

        // ── Cooperative load W[super_n0:super_n0+64, k0:k0+16] ──
        {
            int n_total = kSuperN * kWMMA_K;  // 64x16 = 1024
            for (int tid = threadIdx.x; tid < n_total; tid += 128) {
                int r = tid / kWMMA_K;
                int c = tid % kWMMA_K;
                int gn = super_n0 + r;
                int gk = k0 + c;
                if (gn < N && gk < K && c < tile_k) {
                    int wi = gk / kWeightsPerWord;
                    int pos = gk % kWeightsPerWord;
                    uint32_t word = W[gn * stride_words + wi];
                    int8_t t = decode_ternary(word >> (2 * pos));
                    W_smem[r][c] = __int2half_rn(t);  // row-major: W_smem[n][k]
                } else {
                    W_smem[r][c] = __float2half(0.0f);
                }
            }
        }

        // ── Cooperative load X[super_b0:super_b0+64, k0:k0+16] ──
        {
            int n_total = kSuperM * kWMMA_K;
            for (int tid = threadIdx.x; tid < n_total; tid += 128) {
                int r = tid / kWMMA_K;
                int c = tid % kWMMA_K;
                int gb = super_b0 + r;
                int gk = k0 + c;
                if (gb < B && gk < K && c < tile_k) {
                    X_smem[r][c] = X[gb * K + gk];
                } else {
                    X_smem[r][c] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // ── Each warp computes its 4 fragments ──
        #pragma unroll
        for (int fi = 0; fi < kFragsPerWarp; ++fi) {
            int frag_b_off = (fi / 2) * kWMMA_M;    // 0 or 16
            int frag_n_off = (fi % 2) * kWMMA_N;    // 0 or 16

            int b_base = warp_b_off + frag_b_off;   // 0..48
            int n_base = warp_n_off + frag_n_off;   // 0..48

            wmma::load_matrix_sync(a_frag,
                &X_smem[b_base][0], kWMMA_K);

            wmma::load_matrix_sync(b_frag,
                &W_smem[n_base][0], kWMMA_K);

            wmma::mma_sync(c_frag[fi], a_frag, b_frag, c_frag[fi]);
        }

        __syncthreads();
    }

    // ── Store results (identical to production spill epilogue) ──
    __shared__ float spill[4][kFragsPerWarp][kWMMA_M * kWMMA_N];  // 16 KB

    #pragma unroll
    for (int fi = 0; fi < kFragsPerWarp; ++fi) {
        wmma::store_matrix_sync(&spill[warp_id][fi][0], c_frag[fi],
                                kWMMA_N, wmma::mem_row_major);
    }
    __syncthreads();

    constexpr int kTotalStore = kFragsPerWarp * kWMMA_M * kWMMA_N;  // 1024, per warp
    for (int tid = threadIdx.x; tid < kTotalStore * kWarps; tid += 128) {
        int slot_w = tid / kTotalStore;          // 0..3 which warp's tile
        int lin    = tid % kTotalStore;
        int fi     = lin / (kWMMA_M * kWMMA_N);
        int off    = lin % (kWMMA_M * kWMMA_N);
        int r = off / kWMMA_N;
        int c = off % kWMMA_N;

        int warp_b_off_s = (slot_w / 2) * 32;
        int warp_n_off_s = (slot_w % 2) * 32;

        int gb = super_b0 + warp_b_off_s + (fi / 2) * kWMMA_M + r;
        int gn = super_n0 + warp_n_off_s + (fi % 2) * kWMMA_N + c;

        if (gb < B && gn < N) {
            Y[gb * N + gn] = __float2half_rn(spill[slot_w][fi][off]);
        }
    }
}

// ── Formulation B: transposed launch, kernel reads n from the fast dim ──────

__global__ __launch_bounds__(128) void packed_ternary_forward_tc_64_xpose_kernel(
    const uint32_t* __restrict__ W,   // [N, stride] packed
    const half*     __restrict__ X,   // [B, K] FP16
    half*           __restrict__ Y,   // [B, N] FP16
    int B, int K, int N, int stride_words)
{
    // Host launches grid = (n_tiles, m_tiles): blockIdx.x is n-tile (fastest),
    // blockIdx.y is m-tile. Consecutive linear CTAs therefore share the same
    // m-tile (same X tile) and span distinct n-tiles (distinct packed-W rows).
    int super_n0 = blockIdx.x * kSuperN;
    int super_b0 = blockIdx.y * kSuperM;

    int warp_id = threadIdx.x / 32;
    int wtid    = threadIdx.x % 32;

    int warp_b_off = (warp_id / 2) * 32;  // 0 or 32
    int warp_n_off = (warp_id % 2) * 32;  // 0 or 32

    __shared__ half W_smem[kSuperN][kWMMA_K];  // [64][16]
    __shared__ half X_smem[kSuperM][kWMMA_K];  // [64][16]

    wmma::fragment<wmma::matrix_a, kWMMA_M, kWMMA_N, kWMMA_K,
                   half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kWMMA_M, kWMMA_N, kWMMA_K,
                   half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, kWMMA_M, kWMMA_N, kWMMA_K,
                   float> c_frag[kFragsPerWarp];

    #pragma unroll
    for (int f = 0; f < kFragsPerWarp; ++f)
        wmma::fill_fragment(c_frag[f], 0.0f);

    for (int k0 = 0; k0 < K; k0 += kWMMA_K) {
        int tile_k = min(kWMMA_K, K - k0);

        // ── Cooperative load W[super_n0:super_n0+64, k0:k0+16] ──
        {
            int n_total = kSuperN * kWMMA_K;
            for (int tid = threadIdx.x; tid < n_total; tid += 128) {
                int r = tid / kWMMA_K;
                int c = tid % kWMMA_K;
                int gn = super_n0 + r;
                int gk = k0 + c;
                if (gn < N && gk < K && c < tile_k) {
                    int wi = gk / kWeightsPerWord;
                    int pos = gk % kWeightsPerWord;
                    uint32_t word = W[gn * stride_words + wi];
                    int8_t t = decode_ternary(word >> (2 * pos));
                    W_smem[r][c] = __int2half_rn(t);
                } else {
                    W_smem[r][c] = __float2half(0.0f);
                }
            }
        }

        // ── Cooperative load X[super_b0:super_b0+64, k0:k0+16] ──
        {
            int n_total = kSuperM * kWMMA_K;
            for (int tid = threadIdx.x; tid < n_total; tid += 128) {
                int r = tid / kWMMA_K;
                int c = tid % kWMMA_K;
                int gb = super_b0 + r;
                int gk = k0 + c;
                if (gb < B && gk < K && c < tile_k) {
                    X_smem[r][c] = X[gb * K + gk];
                } else {
                    X_smem[r][c] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int fi = 0; fi < kFragsPerWarp; ++fi) {
            int frag_b_off = (fi / 2) * kWMMA_M;
            int frag_n_off = (fi % 2) * kWMMA_N;

            int b_base = warp_b_off + frag_b_off;
            int n_base = warp_n_off + frag_n_off;

            wmma::load_matrix_sync(a_frag,
                &X_smem[b_base][0], kWMMA_K);
            wmma::load_matrix_sync(b_frag,
                &W_smem[n_base][0], kWMMA_K);

            wmma::mma_sync(c_frag[fi], a_frag, b_frag, c_frag[fi]);
        }

        __syncthreads();
    }

    // ── Store results (identical to production spill epilogue) ──
    __shared__ float spill[4][kFragsPerWarp][kWMMA_M * kWMMA_N];

    #pragma unroll
    for (int fi = 0; fi < kFragsPerWarp; ++fi) {
        wmma::store_matrix_sync(&spill[warp_id][fi][0], c_frag[fi],
                                kWMMA_N, wmma::mem_row_major);
    }
    __syncthreads();

    constexpr int kTotalStore = kFragsPerWarp * kWMMA_M * kWMMA_N;
    for (int tid = threadIdx.x; tid < kTotalStore * kWarps; tid += 128) {
        int slot_w = tid / kTotalStore;
        int lin    = tid % kTotalStore;
        int fi     = lin / (kWMMA_M * kWMMA_N);
        int off    = lin % (kWMMA_M * kWMMA_N);
        int r = off / kWMMA_N;
        int c = off % kWMMA_N;

        int warp_b_off_s = (slot_w / 2) * 32;
        int warp_n_off_s = (slot_w % 2) * 32;

        int gb = super_b0 + warp_b_off_s + (fi / 2) * kWMMA_M + r;
        int gn = super_n0 + warp_n_off_s + (fi % 2) * kWMMA_N + c;

        if (gb < B && gn < N) {
            Y[gb * N + gn] = __float2half_rn(spill[slot_w][fi][off]);
        }
    }
}

// ── Launchers ────────────────────────────────────────────────────────────────

extern "C" void launch_packed_ternary_forward_tc_64_swizzle(
    const uint32_t* W, const void* X_ptr, void* Y_ptr,
    int batch_size, int in_features, int out_features,
    int stride_words, cudaStream_t stream)
{
    const half* X = static_cast<const half*>(X_ptr);
    half* Y = static_cast<half*>(Y_ptr);

    // Launch config IDENTICAL to production: grid=(m_tiles, n_tiles).
    dim3 grid((batch_size + kSuperM - 1) / kSuperM,
              (out_features + kSuperN - 1) / kSuperN);
    dim3 block(128);

    packed_ternary_forward_tc_64_swizzle_kernel<<<grid, block, 0, stream>>>(
        W, X, Y, batch_size, in_features, out_features, stride_words,
        (int)grid.y  // num_n_tiles = ceil(N / 64)
    );
}

extern "C" void launch_packed_ternary_forward_tc_64_xpose(
    const uint32_t* W, const void* X_ptr, void* Y_ptr,
    int batch_size, int in_features, int out_features,
    int stride_words, cudaStream_t stream)
{
    const half* X = static_cast<const half*>(X_ptr);
    half* Y = static_cast<half*>(Y_ptr);

    // Transposed launch: grid=(n_tiles, m_tiles) so the FASTEST blockIdx
    // dimension (x) is the n-tile.
    dim3 grid((out_features + kSuperN - 1) / kSuperN,
              (batch_size + kSuperM - 1) / kSuperM);
    dim3 block(128);

    packed_ternary_forward_tc_64_xpose_kernel<<<grid, block, 0, stream>>>(
        W, X, Y, batch_size, in_features, out_features, stride_words
    );
}
