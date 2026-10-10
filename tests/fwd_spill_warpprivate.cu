/**
 * fwd_spill_warpprivate.cu — forward TC64 with warp-private spill epilogue.
 *
 * GO item #1 from the C13 sweep (docs/speedpass/2026-10-10-20-agent-crack-sweep.md,
 * investigations #6/#13): the production forward kernel's 16 KB spill buffer
 * (spill[4][4][256], shared across warps) + CTA-barrier epilogue caps occupancy
 * at 3 CTAs/SM on the 64 KB T4 SM (20480 B static smem > 16384 B needed for 4).
 * dX's shipped warp-private epilogue (gemm_backward_dx_tc.cu, validated
 * bit-identical in C11) runs at 12 KB smem / 5 CTAs/SM.
 *
 * This file is the production kernel with ONLY the epilogue restructured to the
 * dX pattern:
 *   OLD: spill[4][4][256] (16 KB) — each warp fills all 4 of its fragment slots,
 *        one __syncthreads(), then a whole-CTA cooperative copy of 4096 floats.
 *   NEW: spill[4][256] (4 KB, warp-private slot reused per fragment) — each warp
 *        stores its fragment, copies its OWN 16x16 tile to global, __syncwarp
 *        between fragments. Zero CTA barriers in the epilogue.
 *
 * SMEM: 2 KB (W) + 2 KB (X) + 4 KB (spill) = 8 KB -> 8 CTAs/SM by smem alone
 * (capped by __launch_bounds__/regs; measured by ptxas below).
 *
 * Everything outside the epilogue is byte-identical to the production kernel.
 *
 * Parity: BIT-EXACT expected. The epilogue has no cross-warp data flow in either
 * form — OLD distributes the copy work but reads only the copying thread's own
 * slot values; NEW has each warp copy its own values. Same float values are
 * stored to the same Y addresses; only the ownership of the copy changes.
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
constexpr int kFragsPerWarp = 4;  // 2×2 grid per warp

__global__ __launch_bounds__(128) void packed_ternary_forward_tc_64_kernel(
    const uint32_t* __restrict__ W,   // [N, stride] packed
    const half*     __restrict__ X,   // [B, K] FP16
    half*           __restrict__ Y,   // [B, N] FP16
    int B, int K, int N, int stride_words)
{
    int super_b0 = blockIdx.x * kSuperM;
    int super_n0 = blockIdx.y * kSuperN;

    int warp_id = threadIdx.x / 32;
    int wtid    = threadIdx.x % 32;

    // Per-warp position within the 64×64 CTA tile
    // warp 0: rows 0-31, cols 0-31
    // warp 1: rows 0-31, cols 32-63
    // warp 2: rows 32-63, cols 0-31
    // warp 3: rows 32-63, cols 32-63
    int warp_b_off = (warp_id / 2) * 32;  // 0 or 32
    int warp_n_off = (warp_id % 2) * 32;  // 0 or 32

    // Shared memory
    // W_smem is [n][k] row-major (64×16): W[n0:64, k0:k0+16] stored untransposed.
    // b_frag loads from it as col_major (leading dim kWMMA_K=16) — the same
    // trick dX uses. Consecutive lanes write consecutive k (stride 1), killing
    // the 32-way store bank conflict of the old transposed W_smem[c][r].
    __shared__ half W_smem[kSuperN][kWMMA_K];  // [64][16]
    __shared__ half X_smem[kSuperM][kWMMA_K];  // [64][16]

    // One 16×16 warp-private accumulator slot (4 KB), same as dX's shipped
    // epilogue. Only the owning warp ever touches its slot, so the epilogue
    // needs no __syncthreads().
    __shared__ float spill[kWarps][kWMMA_M * kWMMA_N];  // [4][256]

    // WMMA fragments for 4 sub-tiles per warp
    // Each warp does 4 fragments: (b_off, n_off) offsets within warp's 32×32
    wmma::fragment<wmma::matrix_a, kWMMA_M, kWMMA_N, kWMMA_K,
                   half, wmma::row_major> a_frag;

    // b_frag is the W operand. In the row-major W_smem[n][k], a matrix_b
    // fragment reads [n][k] with the k dimension contiguous in memory — that
    // is WMMAs "col_major" (leading dim = kWMMA_K = 16). dX uses the identical
    // trick and is conflict-free on the same store direction.
    wmma::fragment<wmma::matrix_b, kWMMA_M, kWMMA_N, kWMMA_K,
                   half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, kWMMA_M, kWMMA_N, kWMMA_K,
                   float> c_frag[kFragsPerWarp];

    // Initialize accumulators
    #pragma unroll
    for (int f = 0; f < kFragsPerWarp; ++f)
        wmma::fill_fragment(c_frag[f], 0.0f);

    // Outer loop over K in steps of 16
    for (int k0 = 0; k0 < K; k0 += kWMMA_K) {
        int tile_k = min(kWMMA_K, K - k0);

        // ── Cooperative load W[super_n0:super_n0+64, k0:k0+16] ──
        {
            int n_total = kSuperN * kWMMA_K;  // 64×16 = 1024
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

            // Load X tile: rows b_base..b_base+15 × columns 0..kWMMA_K-1
            wmma::load_matrix_sync(a_frag,
                &X_smem[b_base][0], kWMMA_K);

            // Load W tile: rows n_base..n_base+15 × columns 0..kWMMA_K-1
            // W_smem is row-major [n][k]; matrix_b col_major reads [n][k]
            // with row stride = ldm = kWMMA_K (16). Same trick as dX kernel.
            wmma::load_matrix_sync(b_frag,
                &W_smem[n_base][0], kWMMA_K);

            wmma::mma_sync(c_frag[fi], a_frag, b_frag, c_frag[fi]);
        }

        __syncthreads();
    }

    // ── Store results to global Y ──
    // WARP-PRIVATE spill (one 16×16 slot per warp, see declaration above).
    // The production kernel spills all 4 fragments per warp to a 16 KB
    // spill[4][4][256] buffer, takes one CTA barrier, then copies all 4
    // warps' tiles with a whole-CTA cooperative loop. That 16 KB of smem is
    // what caps occupancy at 3 CTAs/SM on a 64 KB T4 SM. Instead, mirror the
    // shipped dX epilogue: each warp stores a fragment to its own 4 KB slot,
    // copies its own 16×16 tile to global, __syncwarp between fragments —
    // zero CTA barriers in the epilogue. Only the owning warp touches its
    // slot, so store_matrix_sync and the global stores are warp-synchronous
    // by construction.
    #pragma unroll
    for (int fi = 0; fi < kFragsPerWarp; ++fi) {
        wmma::store_matrix_sync(&spill[warp_id][0], c_frag[fi],
                                kWMMA_N, wmma::mem_row_major);

        const int frag_b_off = (fi / 2) * kWMMA_M;
        const int frag_n_off = (fi % 2) * kWMMA_N;

        // This warp copies its own 16×16 tile (32 lanes, 8 elements each).
        #pragma unroll
        for (int e = wtid; e < kWMMA_M * kWMMA_N; e += 32) {
            int r = e / kWMMA_N;
            int c = e % kWMMA_N;
            int gb = super_b0 + warp_b_off + frag_b_off + r;
            int gn = super_n0 + warp_n_off + frag_n_off + c;
            if (gb < B && gn < N) {
                Y[gb * N + gn] = __float2half_rn(spill[warp_id][r * kWMMA_N + c]);
            }
        }
        // store_matrix_sync + shared loads/stores are warp-synchronous; the next
        // iteration reuses the slot only after this warp's reads retire.
        __syncwarp();
    }
}


extern "C" void launch_packed_ternary_forward_tc_64(
    const uint32_t* W, const void* X_ptr, void* Y_ptr,
    int batch_size, int in_features, int out_features,
    int stride_words, cudaStream_t stream)
{
    const half* X = static_cast<const half*>(X_ptr);
    half* Y = static_cast<half*>(Y_ptr);

    dim3 grid((batch_size + kSuperM - 1) / kSuperM,
              (out_features + kSuperN - 1) / kSuperN);
    dim3 block(128);

    packed_ternary_forward_tc_64_kernel<<<grid, block, 0, stream>>>(
        W, X, Y, batch_size, in_features, out_features, stride_words
    );
}
