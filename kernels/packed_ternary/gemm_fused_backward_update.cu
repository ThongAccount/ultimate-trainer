/**
 * gemm_fused_backward_update.cu — Fused Backward dX + Weight Update (WMMA).
 *
 * Single CUDA kernel computing both:
 *   1. dX_partial = dY × W  (gradient w.r.t. input, via atomicAdd)
 *   2. dW_accum   = dY^T × X (manual reduction, per-weight register accum)
 *   3. counter += sign(dW); if |counter| > threshold → atomicCAS bit-flip
 *
 * ── Grid ──  (ceil(N / 32), ceil(K / 32))
 * ── Block ── 128 threads (4 warps)
 *
 * Each CTA owns a 32×32 tile of W (N_sub × K_sub):
 *   - W_smem[32][32] decoded once and reused across all batch tiles
 *   - dY and X streamed in BATCH_STEP=16 batch tiles
 *   - dX: WMMA row_major(dY) × row_major(W) → atomicAdd to global
 *   - dW: manual reduction over batch dimension, per-(n,k) register accum
 *   - Counter update: full batch sum, fixed ±1 (matching scalar/v2 semantics)
 *
 * Targets sm_75+ (Tesla T4). Scales to sm_80+ / sm_90+.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "packed_ternary.cuh"
#include <mma.h>

namespace wmma = nvcuda::wmma;

// ── Tile constants ──────────────────────────────────────────────────────────

constexpr int kWMMA_M = 16;
constexpr int kWMMA_N = 16;
constexpr int kWMMA_K = 16;
constexpr int kWarps     = 4;
constexpr int kSuperM    = 32;  // CTA tile: N dimension
constexpr int kSuperN    = 32;  // CTA tile: K dimension
constexpr int kBatchStep = 16;  // batch tile size

// ── Shared memory ───────────────────────────────────────────────────────────
//  W_smem:  32×32 half      = 2048 B
//  dY_smem: 4×16×16 half    = 2048 B   (raw, for dX WMMA & dW manual reduction)
//  X_smem:  4×16×16 half    = 2048 B   (for dW manual reduction)
//  spill:   4×16×16 float   = 4096 B   (dX WMMA ↔ global bridge)
//  Total: 10240 B ✓

// ── Global state (per-CTA, in/out via pointers) ─────────────────────────────
//   dY[B,N]  — read   — gradient w.r.t. output
//   X[B,K]   — read   — input activations
//   W_read[N,stride]  — read   — packed ternary snapshot for dX
//   dX[B,K]  — write  — gradient w.r.t. input (must be zero-initialised)
//   W_mut[N,stride]   — write  — same allocation, mutated with atomicCAS
//   counter[N,K]      — write  — int16 counters mutated in-place

// ═════════════════════════════════════════════════════════════════════════════
//  Kernel
// ═════════════════════════════════════════════════════════════════════════════

__global__ __launch_bounds__(128) void packed_ternary_backward_update_fused_kernel(
    const half*     __restrict__ dY,
    const half*     __restrict__ X,
    const uint32_t* __restrict__ W_read,
    half*           __restrict__ dX,
    uint32_t*       __restrict__ W_mut,
    int16_t*        __restrict__ counter,
    int B, int K, int N, int stride_words,
    int16_t threshold)
{
    // ── CTA extents ─────────────────────────────────────────────────────
    int super_n0 = blockIdx.x * kSuperM;  // global N offset
    int super_k0 = blockIdx.y * kSuperN;  // global K offset
    int n_max = min(super_n0 + kSuperM, N);
    int k_max = min(super_k0 + kSuperN, K);

    int warp_id = threadIdx.x / 32;
    int wtid    = threadIdx.x % 32;

    // Per-warp offsets within 32×32 CTA tile
    int warp_n_off = (warp_id / 2) * kWMMA_M;  // 0 or 16
    int warp_k_off = (warp_id % 2) * kWMMA_N;  // 0 or 16
    int gn_warp = super_n0 + warp_n_off;
    int gk_warp = super_k0 + warp_k_off;

    // ── Shared memory ──────────────────────────────────────────────────
    __shared__ half  W_smem[kSuperM][kSuperN];           // decoded W, once
    __shared__ half  dY_smem[kWarps][kWMMA_M][kWMMA_K];  // dY batch subtile
    __shared__ half  X_smem[kWarps][kWMMA_M][kWMMA_N];   // X batch subtile

    // Reused: dX float spill → atomicAdd, then dW shared accum
    __shared__ float spill[kWarps][kWMMA_M][kWMMA_N];

    // ── WMMA fragments for dX ─────────────────────────────────────────
    wmma::fragment<wmma::matrix_a, kWMMA_M, kWMMA_N, kWMMA_K,
                   half, wmma::row_major> a_frag;   // dY [b×n]
    wmma::fragment<wmma::matrix_b, kWMMA_M, kWMMA_N, kWMMA_K,
                   half, wmma::row_major> b_frag;   // W  [n×k]
    wmma::fragment<wmma::accumulator, kWMMA_M, kWMMA_N, kWMMA_K,
                   float> c_frag;                    // dX partial [b×k]

    // ── Per-thread dW registers ────────────────────────────────────────
    // Each thread handles 8 dW elements (1024 per CTA / 128 threads)
    float dw_reg[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) dw_reg[i] = 0.0f;

    // ═══════════════════════════════════════════════════════════════════
    //  Phase 0: Load packed W → W_smem (decoded FP16, once)
    // ═══════════════════════════════════════════════════════════════════

    {
        // Full SMEM zero-init for out-of-range rows/cols
        half zero = __float2half(0.0f);
        for (int tid = threadIdx.x; tid < kSuperM * kSuperN; tid += 128) {
            int r = tid / kSuperN;
            int c = tid % kSuperN;
            int gn = super_n0 + r;
            int gk = super_k0 + c;
            if (gn < N && gk < K) {
                // Decode from packed W_read
                int wi = (super_k0 + c) / kWeightsPerWord;
                int pos = (super_k0 + c) % kWeightsPerWord;
                uint32_t word = W_read[gn * stride_words + wi];
                int8_t t = decode_ternary(word >> (2 * pos));
                W_smem[r][c] = __int2half_rn(t);
            } else {
                W_smem[r][c] = zero;
            }
        }
    }
    __syncthreads();

    // ═══════════════════════════════════════════════════════════════════
    //  Phase 1 + 2: Batch loop — dX partial + dW register accum
    // ═══════════════════════════════════════════════════════════════════

    for (int b0 = 0; b0 < B; b0 += kBatchStep) {
        int Bt = min(kBatchStep, B - b0);

        // ── Load dY[b0:b0+Bt, super_n0:super_n0+32] → dY_smem ──
        {
            int base = wtid * 8;
            #pragma unroll
            for (int j = 0; j < 8; j += 2) {
                int flat = base + j;
                int b_loc = flat / kWMMA_K;   // 0..15
                int n_loc = flat % kWMMA_K;   // 0..15
                if (b_loc < Bt && n_loc < kWMMA_K) {
                    int gb = b0 + b_loc;
                    int gn = gn_warp + n_loc;
                    if (gb < B && gn < N) {
                        int byte_offset = (gb * N + gn) * (int)sizeof(half);
                        if ((byte_offset & 3) == 0 && n_loc + 1 < kWMMA_K && gn + 1 < N) {
                            half2 v = ((const half2*)&dY[gb * N + gn])[0];
                            dY_smem[warp_id][b_loc][n_loc]     = v.x;
                            dY_smem[warp_id][b_loc][n_loc + 1] = v.y;
                        } else {
                            dY_smem[warp_id][b_loc][n_loc] = dY[gb * N + gn];
                            if (n_loc + 1 < kWMMA_K && gn + 1 < N) {
                                dY_smem[warp_id][b_loc][n_loc + 1] = dY[gb * N + gn + 1];
                            }
                        }
                    }
                }
            }
        }

        // ── Load X[b0:b0+Bt, super_k0:super_k0+32] → X_smem ──
        {
            int base = wtid * 8;
            #pragma unroll
            for (int j = 0; j < 8; j += 2) {
                int flat = base + j;
                int b_loc = flat / kWMMA_N;   // 0..15
                int k_loc = flat % kWMMA_N;   // 0..15
                if (b_loc < Bt && k_loc < kWMMA_N) {
                    int gb = b0 + b_loc;
                    int gk = gk_warp + k_loc;
                    if (gb < B && gk < K) {
                        int byte_offset = (gb * K + gk) * (int)sizeof(half);
                        if ((byte_offset & 3) == 0 && k_loc + 1 < kWMMA_N && gk + 1 < K) {
                            half2 v = ((const half2*)&X[gb * K + gk])[0];
                            X_smem[warp_id][b_loc][k_loc]     = v.x;
                            X_smem[warp_id][b_loc][k_loc + 1] = v.y;
                        } else {
                            X_smem[warp_id][b_loc][k_loc] = X[gb * K + gk];
                            if (k_loc + 1 < kWMMA_N && gk + 1 < K) {
                                X_smem[warp_id][b_loc][k_loc + 1] = X[gb * K + gk + 1];
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();

        // ── Zero unused batch rows in dY/X SMEM (for partial batch) ──
        if (Bt < kBatchStep) {
            half zero = __float2half(0.0f);
            int n_total = kWarps * kWMMA_M * kWMMA_K;
            for (int tid = threadIdx.x; tid < n_total; tid += 128) {
                int w = tid / (kWMMA_M * kWMMA_K);
                int rem = tid % (kWMMA_M * kWMMA_K);
                int b = rem / kWMMA_K;
                int r = rem % kWMMA_K;
                if (b >= Bt) dY_smem[w][b][r] = zero;
            }
            n_total = kWarps * kWMMA_M * kWMMA_N;
            for (int tid = threadIdx.x; tid < n_total; tid += 128) {
                int w = tid / (kWMMA_M * kWMMA_N);
                int rem = tid % (kWMMA_M * kWMMA_N);
                int b = rem / kWMMA_N;
                int c = rem % kWMMA_N;
                if (b >= Bt) X_smem[w][b][c] = zero;
            }
            __syncthreads();
        }

        // ── Phase 1: dX_partial = dY_tile × W_tile (WMMA) ─────────
        {
            wmma::load_matrix_sync(a_frag,
                &dY_smem[warp_id][0][0], kWMMA_K);

            wmma::load_matrix_sync(b_frag,
                &W_smem[warp_n_off][warp_k_off], kSuperN);

            wmma::fill_fragment(c_frag, 0.0f);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

            // Spill float result to reuse SMEM
            wmma::store_matrix_sync(
                &spill[warp_id][0][0], c_frag,
                kWMMA_N, wmma::mem_row_major);
        }
        __syncthreads();

        // ── AtomicAdd dX partial to global ────────────────────────
        {
            int n_total = kWarps * kWMMA_M * kWMMA_N;
            for (int tid = threadIdx.x; tid < n_total; tid += 128) {
                int w  = tid / (kWMMA_M * kWMMA_N);
                int rem = tid % (kWMMA_M * kWMMA_N);
                int b_loc = rem / kWMMA_N;
                int k_loc = rem % kWMMA_N;

                int gb = b0 + b_loc;
                int gk = super_k0 + (w % 2) * kWMMA_N + k_loc;

                if (gb < B && gk < K) {
                    float val = spill[w][b_loc][k_loc];
                    if (val != 0.0f) {
                        atomicAdd(&dX[gb * K + gk], __float2half_rn(val));
                    }
                }
            }
        }

        // ── Phase 2: dW register accum — manual reduction ────────
        // Each thread owns 8 (n,k) pairs mapped from its warp's tile.
        // It reads dY_smem[warp][b][n] and X_smem[warp][b][k] for
        // each b in [0, Bt-1] and accumulates into dw_reg[8].
        //
        // Thread i (0..127) handles elements at positions:
        //   global_idx = i + {0, 128, 256, ..., 896}
        //   → (w, r, c) mapping as in counter update below
        //
        {
            int n_per_warp = kWMMA_M * kWMMA_N;  // 256
            int n_per_thread = (kWarps * n_per_warp) / 128;  // 8

            int base_idx = threadIdx.x * n_per_thread;
            for (int ei = 0; ei < n_per_thread; ++ei) {
                int idx = base_idx + ei;
                int w  = idx / n_per_warp;
                int rem = idx % n_per_warp;
                int r  = rem / kWMMA_N;   // n offset in warp tile 0..15
                int c  = rem % kWMMA_N;   // k offset in warp tile 0..15

                // Accumulate over batch
                float sum = 0.0f;
                #pragma unroll
                for (int b = 0; b < Bt; ++b) {
                    float dy_val = __half2float(dY_smem[w][b][r]);
                    float x_val  = __half2float(X_smem[w][b][c]);
                    sum += dy_val * x_val;
                }
                dw_reg[ei] += sum;
            }
        }
        __syncthreads();
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Phase 3: dw_reg → counter → bit-flip via atomicCAS
    // ═══════════════════════════════════════════════════════════════════

    // Write dW from registers to spill SMEM to enable vectorised pairs
    {
        int n_per_warp = kWMMA_M * kWMMA_N;
        int n_per_thread = (kWarps * n_per_warp) / 128;  // 8

        int base_idx = threadIdx.x * n_per_thread;
        for (int ei = 0; ei < n_per_thread; ++ei) {
            int idx = base_idx + ei;
            int w  = idx / n_per_warp;
            int rem = idx % n_per_warp;
            int r  = rem / kWMMA_N;
            int c  = rem % kWMMA_N;
            spill[w][r][c] = dw_reg[ei];
        }
    }
    __syncthreads();

    // Process counter updates: vectorized int32 pairs, skip zero grads
    {
        int n_pairs = (kWarps * kWMMA_M * kWMMA_N) / 2;  // 512
        for (int i = threadIdx.x; i < n_pairs; i += 128) {
            int pair_w = (i * 2) / (kWMMA_M * kWMMA_N);
            int pair_linear = (i * 2) % (kWMMA_M * kWMMA_N);
            int r = pair_linear / kWMMA_N;
            int c = pair_linear % kWMMA_N;

            int gn_w = super_n0 + (pair_w / 2) * kWMMA_M + r;
            int gk_w = super_k0 + (pair_w % 2) * kWMMA_N + c;

            if (gn_w >= N || gk_w + 1 >= K) continue;

            float g0 = spill[pair_w][r][c];
            float g1 = spill[pair_w][r][c + 1];

            if (g0 == 0.0f && g1 == 0.0f) continue;

            int idx = gn_w * K + gk_w;

            // Counter load: use vectorized int32 when aligned, scalar when not
            int16_t cnt0, cnt1;
            if ((idx * (int)sizeof(int16_t)) & 3) {
                // Misaligned: load as two int16 scalars
                cnt0 = counter[idx];
                cnt1 = counter[idx + 1];
            } else {
                // Aligned: vectorized int32 load
                int32_t cnt_pair = *(const int32_t*)&counter[idx];
                cnt0 = (int16_t)(cnt_pair & 0xFFFF);
                cnt1 = (int16_t)((cnt_pair >> 16) & 0xFFFF);
            }

            // Sign-based update: grad>0 → decrement (descent)
            if (g0 > 0.0f)       cnt0--;
            else if (g0 < 0.0f)  cnt0++;

            if (g1 > 0.0f)       cnt1--;
            else if (g1 < 0.0f)  cnt1++;

            uint32_t* w_row = W_mut + gn_w * stride_words;

            if (cnt0 > threshold) {
                increment_weight_atomic(w_row, gk_w);
                cnt0 = 0;
            } else if (cnt0 < -threshold) {
                decrement_weight_atomic(w_row, gk_w);
                cnt0 = 0;
            }

            if (cnt1 > threshold) {
                increment_weight_atomic(w_row, gk_w + 1);
                cnt1 = 0;
            } else if (cnt1 < -threshold) {
                decrement_weight_atomic(w_row, gk_w + 1);
                cnt1 = 0;
            }

            // Counter store: use vectorized int32 when aligned, scalar when not
            if ((idx * (int)sizeof(int16_t)) & 3) {
                counter[idx]     = cnt0;
                counter[idx + 1] = cnt1;
            } else {
                *(int32_t*)&counter[idx] =
                    ((int32_t)cnt1 << 16) | ((int32_t)cnt0 & 0xFFFF);
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Host launch wrapper
// ═════════════════════════════════════════════════════════════════════════════

extern "C" void launch_packed_ternary_backward_update_fused(
    const void*     dY_ptr,
    const void*     X_ptr,
    const uint32_t* W_read,
    void*           dX_ptr,
    uint32_t*       W_mut,
    int16_t*        counter,
    int B, int K, int N, int stride_words,
    int16_t threshold,
    cudaStream_t stream)
{
    const half* dY = static_cast<const half*>(dY_ptr);
    const half* X  = static_cast<const half*>(X_ptr);
    half*       dX = static_cast<half*>(dX_ptr);

    dim3 grid((N + kSuperM - 1) / kSuperM,
              (K + kSuperN - 1) / kSuperN);
    dim3 block(128);

    packed_ternary_backward_update_fused_kernel<<<grid, block, 0, stream>>>(
        dY, X, W_read, dX, W_mut, counter,
        B, K, N, stride_words, threshold
    );
}
