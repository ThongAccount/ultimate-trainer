/**
 * gemm_forward_tc_64.cu — Forward GEMM with 64×64 tile (WMMA).
 *
 * 64×64 tile reduces CTA count by 4× compared to 32×32 tiles:
 *   Head (B=16384, N=50272): 201K CTAs vs 804K CTAs
 *   Body  (B=16384, N=4096):  16K CTAs vs  66K CTAs
 *
 * Grid:  (ceil(B / 64), ceil(N / 64))
 * Block: 128 threads (4 warps)
 *
 * Each CTA computes a 64×64 output tile Y[b:b+64, n:n+64].
 * Each warp computes 4 WMMA fragments (2×2 grid) = 32×32 output.
 *
 * SMEM (20 KB total):
 *   W_smem[64][16]  — 2 KB — W decoded for current K-slice, row-major [n][k]
 *   X_smem[64][16]  — 2 KB — X tile for current K-slice
 *   spill[4][4][256] — 16 KB — float accumulators before half global store
 *
 * Targets sm_75+. Works on any SM with WMMA support.
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
        // W-DEDUP: one K-slice is 16 wide = exactly one packed word per row
        // (wi = gk/16 constant across c). Old code had all 16 lanes of a row
        // load the SAME word (1024 loads, 64 unique). Here 1 thread per row
        // loads the word once and decodes all 16 weights via unpack16
        // (64 loads, 16x less global traffic).
        {
            for (int r = threadIdx.x; r < kSuperN; r += 128) {
                int gn = super_n0 + r;
                if (gn < N && k0 < K) {
                    int wi = k0 / kWeightsPerWord;
                    uint32_t word = W[gn * stride_words + wi];
                    int8_t vals[kWMMA_K];
                    unpack16(word, vals);
                    #pragma unroll
                    for (int c = 0; c < kWMMA_K; ++c) {
                        if (c < tile_k && k0 + c < K) {
                            W_smem[r][c] = __int2half_rn(vals[c]);
                        } else {
                            W_smem[r][c] = __float2half(0.0f);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int c = 0; c < kWMMA_K; ++c) {
                        W_smem[r][c] = __float2half(0.0f);
                    }
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

    // ── Store results ──
    // c_frag is float accumulator; Y is half.  Spill each of the 4 fragments
    // to a per-(warp, fragment) slot first (all disjoint — no syncthreads
    // needed between stores), one barrier, then one cooperative copy of all
    // 2048 elements.  (Old code reused one spill[warp_id] buffer across
    // fragments with per-fragment syncs — race window between store_matrix_sync
    // and the copy produced scattered nondeterministic NaN.)
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
