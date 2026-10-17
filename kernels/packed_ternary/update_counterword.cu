/**
 * gemm_update_tc_v2.cu — Optimized TC gradient → counter → bit-flip.
 *
 * v2 optimizations over v1:
 *   1. Skip counter read/write when grad == 0 (saves 32MB+ traffic)
 *   2. Vectorized counter access: int32 for two adjacent int16 counters
 *   3. Branchless counter increment/decrement
 *   4. Reduced __syncthreads() calls
 *
 * Grid:  (ceil(in_features / 32), ceil(out_features / 32))
 * Block: 128 threads (4 warps)
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "packed_ternary.cuh"
#include <mma.h>

namespace wmma = nvcuda::wmma;

constexpr int kM = 16;
constexpr int kN = 16;
constexpr int kK = 16;
constexpr int kWarpsPerBlock = 4;
constexpr int kSuperM = 32;
constexpr int kSuperN = 32;

#define DYS(w, b, r)  dY_smem[(w) * kK * kM + (b) * kM + (r)]
#define XS(w, b, c)   X_smem[(w) * kK * kN + (b) * kN + (c)]
#define DWF(w, r, c)  dW_float_smem[(w) * kM * kN + (r) * kN + (c)]

__global__ __launch_bounds__(128) void packed_ternary_update_counterword_kernel(
    const half*     __restrict__ X,
    const half*     __restrict__ dY,
    uint32_t*       __restrict__ W,
    int16_t*        __restrict__ counter,
    int batch_size,
    int in_features,
    int out_features,
    int stride_words,
    int16_t threshold)
{
    int super_c0 = blockIdx.x * kSuperN;
    int super_r0 = blockIdx.y * kSuperM;
    int warp_id = threadIdx.x / 32;
    int wtid    = threadIdx.x % 32;

    int warp_c_off = (warp_id / 2) * kN;
    int warp_r_off = (warp_id % 2) * kM;
    int c0 = super_c0 + warp_c_off;
    int r0 = super_r0 + warp_r_off;

    __shared__ half   dY_smem[kWarpsPerBlock * kK * kM];
    __shared__ half   X_smem[kWarpsPerBlock * kK * kN];
    __shared__ float  dW_float_smem[kWarpsPerBlock * kM * kN];

    wmma::fragment<wmma::matrix_a, kM, kN, kK, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, kM, kN, kK, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, kM, kN, kK, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // ── WMMA accumulation loop over batch tiles ─────────────────────
    for (int b0 = 0; b0 < batch_size; b0 += kK) {
        int tile_b = min(kK, batch_size - b0);

        // Load dY tile
        {
            // Coalesced: consecutive lanes cover consecutive half2 pairs of the
            // row-major tile (measured -44% vs strided wtid*8 mapping on T4;
            // bit-exact, see experiment 2026-08-26). 128 pairs / 32 lanes = 4 iters.
            #pragma unroll
            for (int q = wtid; q < kK * kM / 2; q += 32) {
                int i = q * 2;
                int b = i / kM;
                int r = i % kM;
                if (b >= tile_b) continue;
                int gb = b0 + b;
                int gr = r0 + r;
                if (gb >= batch_size || gr >= out_features) continue;
                int byte_off = (gb * out_features + gr) * (int)sizeof(half);
                if ((byte_off & 3) == 0 && r + 1 < kM && gr + 1 < out_features) {
                    half2 v = ((const half2*)&dY[gb * out_features + gr])[0];
                    // Vectorized half2 store: r is always even (i=q*2), so the
                    // target is 4-byte aligned. A single 32-bit store puts 32
                    // lanes on 32 distinct banks (conflict-free); two separate
                    // 16-bit stores alias adjacent halves into one bank (2-way).
                    *reinterpret_cast<half2*>(&DYS(warp_id, b, r)) = v;
                } else {
                    DYS(warp_id, b, r) = dY[gb * out_features + gr];
                    if (r + 1 < kM && gr + 1 < out_features)
                        DYS(warp_id, b, r + 1) = dY[gb * out_features + gr + 1];
                }
            }
        }

        // Load X tile
        {
            // Coalesced: same lane->pair mapping as the dY tile above.
            #pragma unroll
            for (int q = wtid; q < kK * kN / 2; q += 32) {
                int i = q * 2;
                int b = i / kN;
                int c = i % kN;
                if (b >= tile_b) continue;
                int gb = b0 + b;
                int gc = c0 + c;
                if (gb >= batch_size || gc >= in_features) continue;
                int byte_off = (gb * in_features + gc) * (int)sizeof(half);
                if ((byte_off & 3) == 0 && c + 1 < kN && gc + 1 < in_features) {
                    half2 v = ((const half2*)&X[gb * in_features + gc])[0];
                    *reinterpret_cast<half2*>(&XS(warp_id, b, c)) = v;
                } else {
                    XS(warp_id, b, c) = X[gb * in_features + gc];
                    if (c + 1 < kN && gc + 1 < in_features)
                        XS(warp_id, b, c + 1) = X[gb * in_features + gc + 1];
                }
            }
        }
        __syncthreads();

        // ── Zero unused batch rows in dY/X SMEM (partial batch) ──
        if (tile_b < kK) {
            half zero = __float2half(0.0f);
            int n_total = kWarpsPerBlock * kK * kM;
            for (int tid = threadIdx.x; tid < n_total; tid += 128) {
                int w = tid / (kK * kM);
                int rem = tid % (kK * kM);
                int b = rem / kM;
                int r = rem % kM;
                if (b >= tile_b) DYS(w, b, r) = zero;
            }
            n_total = kWarpsPerBlock * kK * kN;
            for (int tid = threadIdx.x; tid < n_total; tid += 128) {
                int w = tid / (kK * kN);
                int rem = tid % (kK * kN);
                int b = rem / kN;
                int c = rem % kN;
                if (b >= tile_b) XS(w, b, c) = zero;
            }
            __syncthreads();
        }

        wmma::load_matrix_sync(a_frag, &dY_smem[warp_id * kK * kM], kM);
        wmma::load_matrix_sync(b_frag, &X_smem[warp_id * kK * kN], kN);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // ── Store accumulator to SMEM ───────────────────────────────────
    wmma::store_matrix_sync(&dW_float_smem[warp_id * kM * kN], c_frag,
                            kN, wmma::mem_row_major);
    __syncthreads();

    // ── Counter + flip: ONE THREAD PER PACKED W WORD (no atomics) ─────
    //
    // Phase 2 variant. Every (gr, gc) element of this CTA's 32x32 tile is
    // owned by exactly one thread of this CTA (disjoint grid), so no atomic
    // is required: plain loads/stores plus ONE word-level read-modify-write
    // of W per 16 elements instead of 16 atomicCAS on shared words.
    // Counter math is IDENTICAL to v2_32: counter += (g>0)?-1:(g<0)?+1:0,
    // saturating clamp to ±32767, strict flip at |cnt|>threshold, reset 0,
    // increment_weight (-1->0->+1 saturating) / decrement_weight mirror.

    const int words_per_row = stride_words;
    const int n_rows_here = min(kSuperM, out_features - super_r0);

    for (int li = threadIdx.x; li < n_rows_here * words_per_row;
         li += blockDim.x) {
        int r_local = li / words_per_row;
        int w = li % words_per_row;
        int gr = super_r0 + r_local;
        if (gr >= out_features) continue;

        // columns owned by this word, clamped to tile + matrix
        int c_lo = w * 16;
        int c_hi = min(c_lo + 16, in_features - super_c0);
        if (c_lo >= kSuperN || c_hi <= c_lo) continue;

        uint32_t word = W[gr * stride_words + w];   // ONE load for 16 elems
        uint32_t new_word = word;
        bool changed = false;

        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            int c = c_lo + j;
            if (c >= c_hi) break;

            // (r_local, c) lives in warp w' = (r_local/kM)*2 + c/kN,
            // elements (r_local%kM, c%kN) of that warp's accumulator
            int wq = (r_local / kM) * 2 + (c / kN);
            int r_in = r_local % kM;
            int c_in = c % kN;

            float g = DWF(wq, r_in, c_in);
            // skip zero-grad elements exactly like v2_32 (no counter touch)
            if (g == 0.0f) continue;

            int gc = super_c0 + c;
            int16_t cnt = counter[gr * in_features + gc];
            cnt += (g > 0.0f) ? -1 : (g < 0.0f) ? 1 : 0;
            cnt = max(-32767, min(32767, cnt));

            if (cnt > threshold || cnt < -threshold) {
                uint32_t code = (word >> (2 * j)) & 3u;
                if (cnt > threshold) {          // increment_weight
                    if (code == 2u) code = 0u;       // -1 -> 0
                    else if (code == 0u) code = 1u;  // 0 -> +1
                    else { /* saturated +1 */ }
                } else {                        // decrement_weight
                    if (code == 1u) code = 0u;       // +1 -> 0
                    else if (code == 0u) code = 2u;  // 0 -> -1
                    else { /* saturated -1 */ }
                }
                new_word = (new_word & ~(3u << (2 * j))) | (code << (2 * j));
                changed = true;
                cnt = 0;
            }
            counter[gr * in_features + gc] = cnt;
        }
        if (changed) {
            W[gr * stride_words + w] = new_word;   // ONE plain store per word
        }
    }

    // ── Tail: handle last column when in_features is odd ────────
    if (in_features & 1) {
        int last_gc = in_features - 1;
        for (int i = threadIdx.x; i < kWarpsPerBlock * kM * kN; i += blockDim.x) {
            int w = i / (kM * kN);
            int linear = i % (kM * kN);
            int r = linear / kN;
            int c = linear % kN;
            if (c != (kN - 1)) continue;  // last column in warp tile

            int warp_c_off_w = (w / 2) * kN;
            int gc = super_c0 + warp_c_off_w + c;
            if (gc != last_gc) continue;

            int warp_r_off_w = (w % 2) * kM;
            int gr = super_r0 + warp_r_off_w + r;

            if (gr >= out_features) continue;

            float g = DWF(w, r, c);
            if (g == 0.0f) continue;

            int idx = gr * in_features + gc;
            int16_t cnt = counter[idx];
            cnt += (g > 0.0f) ? -1 : (g < 0.0f) ? 1 : 0;
            // Saturating clamp before the threshold check (see pair path above).
            cnt = max(-32767, min(32767, cnt));

            // the last column is the only element its word-group owner
            // touches here — plain RMW of the 2-bit field (no atomic)
            int word_j = (in_features - 1) % 16;
            uint32_t* w_row = W + gr * stride_words;
            int wcol = (in_features - 1) / 16;
            if (cnt > threshold || cnt < -threshold) {
                uint32_t word = w_row[wcol];
                uint32_t code = (word >> (2 * word_j)) & 3u;
                if (cnt > threshold) {                 // increment_weight
                    if (code == 2u) code = 0u;
                    else if (code == 0u) code = 1u;
                } else {                               // decrement_weight
                    if (code == 1u) code = 0u;
                    else if (code == 0u) code = 2u;
                }
                w_row[wcol] = (word & ~(3u << (2 * word_j)))
                              | (code << (2 * word_j));
                cnt = 0;
            }
            counter[idx] = cnt;
        }
    }
}

extern "C" void launch_packed_ternary_update_counterword(
    const void*     X_ptr,
    const void*     dY_ptr,
    uint32_t*       W,
    int16_t*        counter,
    int batch_size,
    int in_features,
    int out_features,
    int stride_words,
    int16_t threshold,
    cudaStream_t stream)
{
    const half* X  = static_cast<const half*>(X_ptr);
    const half* dY = static_cast<const half*>(dY_ptr);

    dim3 grid((in_features + kSuperN - 1) / kSuperN,
              (out_features + kSuperM - 1) / kSuperM);
    dim3 block(128);

    packed_ternary_update_counterword_kernel<<<grid, block, 0, stream>>>(
        X, dY, W, counter, batch_size, in_features, out_features,
        stride_words, threshold
    );
}
