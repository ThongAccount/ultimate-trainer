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
constexpr int kK4 = 4 * kWMMA_K;  // 64 — probe fwd_uint4 K-window

__global__ __launch_bounds__(128) void probe_fwd_uint4_kernel(
    const uint32_t* __restrict__ W,   // [N, stride] packed
    const half*     __restrict__ X,   // [B, K] FP16
    half*           __restrict__ Y,   // [B, N] FP16
    int B, int K, int N, int stride_words)
{
    // feat/fwd-xpose: grid is launched (n_tiles, m_tiles) so blockIdx.x is the
    // n-tile (fastest hw raster order) — co-resident CTAs share one X tile
    // (X DRAM 25.7 GB -> ~32 MB at the head shape). Bit-exact vs the old map;
    // measured head fwd -19.9% (docs/speedpass/2026-11-01-c13-probe-batch-results.md).
    int super_n0 = blockIdx.x * kSuperN;
    int super_b0 = blockIdx.y * kSuperM;

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
    __shared__ half W_smem[kSuperN][kK4];  // [64][64] — probe fwd_uint4
    __shared__ half X_smem[kSuperM][kK4];  // [64][64] — probe fwd_uint4

    // feat/fwd-spill: one 16×16 warp-private accumulator slot (4 KB), the
    // shipped dX epilogue pattern. Only the owning warp ever touches its
    // slot, so the epilogue needs no __syncthreads(). 8 KB total static smem
    // instead of 20 KB -> occupancy 3 -> 6+ CTAs/SM on the 64 KB T4 SM.
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

    // Outer loop over K in steps of 64 (probe fwd_uint4: one uint4 W-load per row per window)
    for (int k0 = 0; k0 < K; k0 += kK4) {
        int tile_k = min(kK4, K - k0);

        // ── Cooperative load W[super_n0:super_n0+64, k0:k0+64] via uint4 ──
        // 64 rows x 4 word-groups = 256 uint4 (16 B) loads; k0 % 64 == 0 and
        // stride_words % 4 == 0 at the shipped shapes, so every uint4 is
        // 16 B-aligned. Tail tile (K % 64 != 0) falls back to per-word guards.
        {
            int n_groups = kSuperN * 4;  // 256
            for (int tid = threadIdx.x; tid < n_groups; tid += 128) {
                int r = tid / 4;
                int w = tid % 4;
                int gn = super_n0 + r;
                int wb = (k0 / kWeightsPerWord) + w * 4;  // first word of this group
                uint32_t words[4] = {0u, 0u, 0u, 0u};
                bool fast = (gn < N) && (stride_words % 4 == 0) && (wb % 4 == 0) && (wb + 3 < stride_words) && (w * 16 < tile_k);
                if (fast) {
                    uint4 v4 = *reinterpret_cast<const uint4*>(&W[gn * stride_words + wb]);
                    words[0] = v4.x; words[1] = v4.y; words[2] = v4.z; words[3] = v4.w;
                } else {
                    for (int j = 0; j < 4; ++j) {
                        int wi = wb + j;
                        if (gn < N && wi < stride_words && (w * 16 + j * kWeightsPerWord) < tile_k)
                            words[j] = W[gn * stride_words + wi];
                    }
                }
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    #pragma unroll
                    for (int pos = 0; pos < kWeightsPerWord; ++pos) {
                        int c = w * 16 + j * kWeightsPerWord + pos;
                        int gk = k0 + c;
                        if (gn < N && gk < K && c < tile_k) {
                            int8_t t = decode_ternary(words[j] >> (2 * pos));
                            W_smem[r][c] = __int2half_rn(t);
                        } else {
                            W_smem[r][c] = __float2half(0.0f);
                        }
                    }
                }
            }
        }

        // ── Cooperative load X[super_b0:super_b0+64, k0:k0+16] ──
        {
            int n_total = kSuperM * kK4;
            for (int tid = threadIdx.x; tid < n_total; tid += 128) {
                int r = tid / kK4;
                int c = tid % kK4;
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

            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                // Load X tile: rows b_base..b_base+15 × cols ks*16..ks*16+15
                wmma::load_matrix_sync(a_frag,
                    &X_smem[b_base][ks * kWMMA_K], kK4);

                // Load W tile: rows n_base..n_base+15 × cols ks*16..ks*16+15
                // W_smem is row-major [n][k]; matrix_b col_major reads [n][k]
                // with row stride = ldm = kK4 (64).
                wmma::load_matrix_sync(b_frag,
                    &W_smem[n_base][ks * kWMMA_K], kK4);

                wmma::mma_sync(c_frag[fi], a_frag, b_frag, c_frag[fi]);
            }
        }

        __syncthreads();
    }

    // ── Store results to global Y ──
    // WARP-PRIVATE spill (feat/fwd-spill): each warp stores a fragment to its
    // own 4 KB slot, copies its own 16×16 tile to global, __syncwarp between
    // fragments — zero CTA barriers in the epilogue. Bit-exact vs the old
    // cooperative copy: no cross-warp data flow in either form, same float
    // values to the same Y addresses (probe ut-c13-probes: fc1 −14.4%,
    // fc2 −14.0%, head −7.0%).
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


extern "C" void probe_launch_fwd_uint4(
    const uint32_t* W, const void* X_ptr, void* Y_ptr,
    int batch_size, int in_features, int out_features,
    int stride_words, cudaStream_t stream)
{
    const half* X = static_cast<const half*>(X_ptr);
    half* Y = static_cast<half*>(Y_ptr);

    // Transposed launch: grid=(n_tiles, m_tiles). gridDim.y caps at 65535 —
    // m_tiles = ceil(B/64) is 256 at B=16384, 512 at B=32768, safe.
    dim3 grid((out_features + kSuperN - 1) / kSuperN,
              (batch_size + kSuperM - 1) / kSuperM);
    dim3 block(128);

    probe_fwd_uint4_kernel<<<grid, block, 0, stream>>>(
        W, X, Y, batch_size, in_features, out_features, stride_words
    );
}
