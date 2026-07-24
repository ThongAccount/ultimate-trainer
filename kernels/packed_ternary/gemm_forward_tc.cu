/**
 * gemm_forward_tc.cu — Tensor-Core packed ternary × FP16 forward GEMM.
 *
 * Uses wmma::mma_sync(m=16, n=16, k=16) on T4 Tensor Cores.
 * Packed ternary weights unpacked into __shared__ FP16 tiles.
 *
 * Each block processes a 32×32 super-tile of the output with 4 warps,
 * each warp handling one 16×16 WMMA tile.
 *
 * Optimizations:
 *   - Block-contiguous fill with decode4 (4 ternary vals/load)
 *   - half2 vectorized X loads
 *   - Shared X loads between warps with same batch offset
 *   - __launch_bounds__(128)
 *
 * Grid:   (ceil(batch/32), ceil(out_features/32))
 * Block:  128 threads (4 warps)
 *
 * W (packed uint32_t) × X (FP16) → Y (FP16)
 *
 * NOTE: WMMA load/store_matrix_sync require stride to be a multiple of 16
 * on sm_75.  Do NOT pad SMEM strides for bank conflicts — it breaks WMMA.
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

// ── Per-warp SMEM offsets (stride = kK = 16 — must be multiple of 16) ─
#define W_SMEM(w, r, k)   W_smem[(w) * kN * kK + (r) * kK + (k)]
#define X_SMEM(w, b, k)   X_smem[(w/2) * kM * kK + (b) * kK + (k)]
#define YF_SMEM(w, r, b)  Y_float_smem[(w) * kN * kM + (r) * kM + (b)]
#define YH_SMEM(w, r, b)  Y_smem[(w) * kN * kM + (r) * kM + (b)]

__global__ __launch_bounds__(128) void packed_ternary_tc_kernel(
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

    // ── Shared memory (no padding — stride must be multiple of 16) ────
    __shared__ half   W_smem[kWarpsPerBlock * kN * kK];
    __shared__ half   X_smem[2 * kM * kK];
    __shared__ float  Y_float_smem[kWarpsPerBlock * kN * kM];
    __shared__ half   Y_smem[kWarpsPerBlock * kN * kM];

    // ── WMMA fragments ──────────────────────────────────────────────
    wmma::fragment<wmma::matrix_a, kM, kN, kK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kM, kN, kK, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, kM, kN, kK, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // ── Outer loop over K tiles ──────────────────────────────────────
    for (int k0 = 0; k0 < in_features; k0 += kK) {
        int tile_k = min(kK, in_features - k0);

        // ── Load W tile → unpack to FP16 (block fill + decode4) ──────
        // Block partition: 256 elements / 32 threads = 8 per thread.
        // Each thread loads 1 uint32 word and decodes 8 ternary values
        // via 2× decode4 (4 values each).
        {
            int base = wtid * 8;
            // First 4: positions base .. base+3
            int i0 = base;
            int c0_ = i0 % kK;
            if (c0_ < tile_k) {
                int r  = i0 / kK;
                int gr = r0 + r;
                int gc = k0 + c0_;
                if (gr < out_features && gc < in_features) {
                    int wi = gc / kWeightsPerWord;
                    if (wi < stride_words) {
                        uint32_t word = W[gr * stride_words + wi];
                        int pos = gc % kWeightsPerWord;
                        int8_t t0, t1, t2, t3;
                        decode4(word, pos, &t0, &t1, &t2, &t3);
                        W_SMEM(warp_id, r, c0_    ) = __float2half((float)t0);
                        W_SMEM(warp_id, r, c0_ + 1) = __float2half((float)t1);
                        W_SMEM(warp_id, r, c0_ + 2) = __float2half((float)t2);
                        if (c0_ + 3 < tile_k)
                            W_SMEM(warp_id, r, c0_ + 3) = __float2half((float)t3);
                    }
                }
            }
            // Second 4: positions base+4 .. base+7
            int i4 = base + 4;
            int c4_ = i4 % kK;
            if (c4_ < tile_k) {
                int r  = i4 / kK;
                int gr = r0 + r;
                int gc = k0 + c4_;
                if (gr < out_features && gc < in_features) {
                    int wi = gc / kWeightsPerWord;
                    if (wi < stride_words) {
                        uint32_t word = W[gr * stride_words + wi];
                        int pos = gc % kWeightsPerWord;
                        int8_t t0, t1, t2, t3;
                        decode4(word, pos, &t0, &t1, &t2, &t3);
                        W_SMEM(warp_id, r, c4_    ) = __float2half((float)t0);
                        W_SMEM(warp_id, r, c4_ + 1) = __float2half((float)t1);
                        W_SMEM(warp_id, r, c4_ + 2) = __float2half((float)t2);
                        if (c4_ + 3 < tile_k)
                            W_SMEM(warp_id, r, c4_ + 3) = __float2half((float)t3);
                    }
                }
            }
        }

        // ── Load X tile → X_smem (half2 vectorized, block fill) ──────
        // Warps 0&1 share X slot 0 (b_off=0), warps 2&3 share slot 1.
        // Only warps 0 and 2 load X from global memory.
        if (warp_id % 2 == 0) {
            int xslot = warp_id / 2;
            int base = wtid * 8;
            // 4× half2 pairs covering 8 elements
            for (int j = 0; j < 8; j += 2) {
                int i = base + j;
                int xb = i / kK;
                int xk = i % kK;
                int gb = b0 + xb;
                int gk = k0 + xk;
                if (xk < tile_k && gb < batch_size && gk < in_features) {
                    if (xk + 1 < tile_k && gk + 1 < in_features) {
                        // half2 vectorized: 2 elements in one 4B load
                        half2 v = ((const half2*)&X[gb * in_features + gk])[0];
                        X_SMEM(warp_id, xb, xk)     = v.x;
                        X_SMEM(warp_id, xb, xk + 1) = v.y;
                    } else {
                        X_SMEM(warp_id, xb, xk) = X[gb * in_features + gk];
                    }
                }
            }
        }
        __syncthreads();

        // ── Load WMMA fragments from SMEM ───────────────────────────
        // stride = kK + kPad = 17 (padded)
        wmma::load_matrix_sync(a_frag, &W_smem[warp_id * kN * kK], kK);
        wmma::load_matrix_sync(b_frag, &X_smem[(warp_id / 2) * kM * kK], kM);

        // ── Tensor-core matmul on 16×16×16 tile ─────────────────────
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // ── Store accumulator to shared, then to global Y ────────────────
    wmma::store_matrix_sync(&Y_float_smem[warp_id * kN * kM], c_frag, kM,
                            wmma::mem_row_major);
    __syncthreads();

    // 128 threads convert 1024 float→half elements (8 per thread)
    for (int idx = threadIdx.x; idx < kWarpsPerBlock * kN * kM; idx += blockDim.x) {
        ((half*)Y_smem)[idx] = __float2half(((float*)Y_float_smem)[idx]);
    }
    __syncthreads();

    // ── Write Y_smem to global Y (transposing row,batch → batch,row) ─
    for (int idx = threadIdx.x; idx < kWarpsPerBlock * kN * kM; idx += blockDim.x) {
        int w = idx / (kN * kM);      // which warp's tile
        int linear = idx % (kN * kM); // within tile
        int r = linear / kM;
        int b = linear % kM;

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
