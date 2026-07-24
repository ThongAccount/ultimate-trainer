/**
 * gemm_forward_tc.cu — Tensor-Core packed ternary × FP16 forward GEMM.
 *
 * Uses wmma::mma_sync(m=16, n=16, k=16) on T4 Tensor Cores.
 * Packed ternary weights unpacked into __shared__ FP16 tiles.
 *
 * Each block processes a 32×32 super-tile of the output with 4 warps,
 * each warp handling one 16×16 WMMA tile.  This keeps all 4 warps
 * usefully occupied (vs the previous single-tile approach that
 * wasted 7/8 warps on redundant computation).
 *
 * Grid:   (ceil(batch/32), ceil(out_features/32))
 * Block:  128 threads (4 warps)
 *
 * W (packed uint32_t) × X (FP16) → Y (FP16)
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "packed_ternary.cuh"
#include <mma.h>

namespace wmma = nvcuda::wmma;

constexpr int kM = 16;   // WMMA tile size (batch)
constexpr int kN = 16;   // WMMA tile size (out_features)
constexpr int kK = 16;   // WMMA tile size (in_features / reduction)

constexpr int kWarpsPerBlock = 4;
constexpr int kSuperM = 32;  // super-tile batch (2 × kM)
constexpr int kSuperN = 32;  // super-tile out  (2 × kN)

// ── Per-warp SMEM offsets ─────────────────────────────────────────────

#define W_SMEM(w, r, k)   W_smem[(w) * kN * kK + (r) * kK + (k)]
#define X_SMEM(w, b, k)   X_smem[(w) * kM * kK + (b) * kK + (k)]
#define YF_SMEM(w, r, b)  Y_float_smem[(w) * kN * kM + (r) * kM + (b)]
#define YH_SMEM(w, r, b)  Y_smem[(w) * kN * kM + (r) * kM + (b)]

__global__ void packed_ternary_tc_kernel(
    const uint32_t* __restrict__ W,
    const half*     __restrict__ X,
    half*           __restrict__ Y,
    int batch_size,
    int in_features,
    int out_features,
    int stride_words)
{
    int super_b0 = blockIdx.x * kSuperM;   // super-tile batch offset
    int super_r0 = blockIdx.y * kSuperN;   // super-tile output-row offset
    int warp_id = threadIdx.x / 32;        // 0..3
    int wtid    = threadIdx.x % 32;        // 0..31 (within-warp)

    // Each warp handles one 16×16 output tile within the 32×32 super-tile.
    // warp 0 → (b=0, r=0), warp 1 → (b=0, r=16),
    // warp 2 → (b=16, r=0), warp 3 → (b=16, r=16)
    int warp_b_off = (warp_id / 2) * kM;   // 0 or 16
    int warp_r_off = (warp_id % 2) * kN;   // 0 or 16

    int b0 = super_b0 + warp_b_off;
    int r0 = super_r0 + warp_r_off;

    // ── Shared memory (4 warps × independent tiles) ──────────────────
    __shared__ half   W_smem[kWarpsPerBlock * kN * kK];
    __shared__ half   X_smem[kWarpsPerBlock * kM * kK];
    __shared__ float  Y_float_smem[kWarpsPerBlock * kN * kM];
    __shared__ half   Y_smem[kWarpsPerBlock * kN * kM];

    // ── WMMA fragments (per-warp, in registers) ──────────────────────
    wmma::fragment<wmma::matrix_a, kM, kN, kK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kM, kN, kK, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, kM, kN, kK, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // ── Outer loop over K tiles ──────────────────────────────────────
    for (int k0 = 0; k0 < in_features; k0 += kK) {
        int tile_k = min(kK, in_features - k0);

        // ── Load W tile → unpack to FP16 (per-warp, strided fill) ────
        // Each warp has 32 threads, must fill 16×16 = 256 SMEM elements.
        // Strided loop: each thread handles 8 elements.
        for (int i = wtid; i < kN * kK; i += 32) {
            int r = i / kK;               // 0..15
            int c = i % kK;               // 0..15
            half w_val = __float2half(0.0f);
            if (c < tile_k) {
                int gr = r0 + r;           // global output row
                int gc = k0 + c;           // global input col
                if (gr < out_features && gc < in_features) {
                    int wi = gc / kWeightsPerWord;
                    if (wi < stride_words) {
                        uint32_t word = W[gr * stride_words + wi];
                        int pos = gc % kWeightsPerWord;
                        int8_t t = decode_ternary(word >> (kTernaryBits * pos));
                        w_val = __float2half((float)t);
                    }
                }
            }
            W_SMEM(warp_id, r, c) = w_val;
        }

        // ── Load X tile → X_smem (per-warp, strided fill) ────────────
        for (int i = wtid; i < kM * kK; i += 32) {
            int xb = i / kK;              // 0..15
            int xk = i % kK;              // 0..15
            half x_val = __float2half(0.0f);
            if (xk < tile_k) {
                int gb = b0 + xb;
                int gk = k0 + xk;
                if (gb < batch_size && gk < in_features) {
                    x_val = X[gb * in_features + gk];
                }
            }
            X_SMEM(warp_id, xb, xk) = x_val;
        }
        __syncthreads();

        // ── Load WMMA fragments from SMEM ────────────────────────────
        wmma::load_matrix_sync(a_frag, &W_smem[warp_id * kN * kK], kK);
        wmma::load_matrix_sync(b_frag, &X_smem[warp_id * kM * kK], kM);

        // ── Tensor-core matmul on 16×16×16 tile ──────────────────────
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // ── Store accumulator to shared, then to global Y ────────────────
    wmma::store_matrix_sync(&Y_float_smem[warp_id * kN * kM], c_frag, kM,
                            wmma::mem_row_major);
    __syncthreads();
    // 128 threads convert 1024 float→half elements (8 per thread)
    for (int idx = threadIdx.x; idx < kWarpsPerBlock * kM * kN; idx += blockDim.x) {
        ((half*)Y_smem)[idx] =
            __float2half(((float*)Y_float_smem)[idx]);
    }
    __syncthreads();

    // ── Write Y_smem to global Y (transposing row,batch → batch,row) ─
    // 128 threads, 1024 elements, 8 per thread.
    for (int idx = threadIdx.x; idx < kWarpsPerBlock * kM * kN; idx += blockDim.x) {
            int w = idx / (kM * kN);        // which warp's tile
            int local = idx % (kM * kN);    // (r,b) within the 16×16 tile
            int r = local / kM;
            int b = local % kM;

            int warp_b_off_w = (w / 2) * kM;
            int warp_r_off_w = (w % 2) * kN;

            int gr = super_r0 + warp_r_off_w + r;
            int gb = super_b0 + warp_b_off_w + b;
            if (gr < out_features && gb < batch_size) {
                Y[gb * out_features + gr] = YH_SMEM(w, r, b);
            }
        }
    }

extern "C" void launch_packed_ternary_tc(
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

    dim3 grid((batch_size + kSuperM - 1) / kSuperM,
              (out_features + kSuperN - 1) / kSuperN);
    dim3 block(128);  // 4 warps

    packed_ternary_tc_kernel<<<grid, block, 0, stream>>>(
        W, X, Y, batch_size, in_features, out_features, stride_words
    );
}
