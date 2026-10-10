/**
 * gemm_update_tc_v2_32 double-buffer probe (GO item #3, sweep investigation #20).
 *
 * Mechanism under test: the production batch loop is a single-buffered
 * lockstep pipeline. Per batch iteration the warp pays the FULL exposed
 * global-load latency of the dY/X tile loads (the dependent smem stores
 * stall until the data arrives), then a CTA barrier, then the WMMA, then a
 * second CTA barrier — the sweep measured ~2200 cycles of exposed latency
 * per iteration (1.8% of DRAM roofline, 12% of TC peak; the machine idles
 * at the barriers). This variant software-pipelines the loop with TWO smem
 * buffer sets and a REGISTER-STAGED load path:
 *
 *   prologue:  LDG tile 0 -> regs, STS -> buffer 0, barrier
 *   iteration t (steady state, ONE barrier per trip):
 *     1. LDG tile t+1 into registers      (no consumer yet -> in flight)
 *     2. mma(t) on buffer cur             (overlaps the in-flight loads)
 *     3. STS registers -> buffer nxt      (first stall lands HERE, after
 *                                          the mma already executed)
 *        + zero-fill of the partial-tile tail rows of buffer nxt
 *     4. the ONE barrier                  (orders STS(t+1) before mma(t+1)
 *                                          and mma(t) reads before the
 *                                          loads of t+2 overwrite cur)
 *     swap cur/nxt
 *
 * Register staging is what makes the prefetch real with plain LDG: if the
 * smem store directly consumes the load (LDG;STS back-to-back before the
 * mma), the warp stalls on the load data BEFORE the mma issues and nothing
 * overlaps. Splitting the consumer STS to after the mma keeps the loads in
 * flight across it. sm_75 has cp.async (LDGSTS), which would move even the
 * STS stall into the LSU, but a 2-stage cp.async pipeline still exposes
 * (L_load - mma - barrier) per iteration — the residual overlap is the
 * same order as this plain-LDG variant — so per the campaign constraint the
 * plain-load form is used rather than fighting WMMA/cp.async interop under
 * torch load_inline. Adding a SECOND per-iteration barrier is forbidden
 * (that is the kSub=2 failure class); steady state here has exactly one.
 *
 * PRE-REGISTERED RISK (sweep #20): +4KB smem (8192 -> 12288 B/block). On
 * sm_75, floor(65536/12288) = 5 CTAs/SM (smem-limited) vs production's 8
 * (which is the 1024-thread/SM cap) — the cliff is 8 -> 5, WORSE than the
 * sweep's nominal 8 -> 6. The probe exists to measure whether the latency
 * overlap beats that loss. Falsifier (pre-registered): fc1 delta within
 * +/-2% or slower -> the pipelining class is CLOSED permanently.
 *
 * Parity argument (W + counter BIT-EXACT vs production):
 *   - X and dY are only ever READ by this kernel, and each counter/W
 *     element is owned by exactly one CTA, so both arms observe identical
 *     global inputs.
 *   - The lane->address mapping and all store predicates are copied
 *     verbatim from production, so the same tile values land in smem
 *     (in a different buffer). Skipped stores leave stale smem exactly as
 *     production does; stale/uninitialized lanes can only sit at positions
 *     whose mma contributions fall in rows/cols that the epilogue bounds-
 *     checks away (gr >= out_features, gc >= in_features, b >= tile_b is
 *     zero-filled identically in both), so they never reach W or counter.
 *   - The mma accumulation order per warp is the same t = 0..n_tiles-1
 *     sequence over identical fragments -> every dW float is bit-identical.
 *   - The epilogue (store_matrix_sync, counter update, odd-in tail) is
 *     byte-identical to production.
 *   - Only scheduling changes: which buffer, and when loads are issued.
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
constexpr int kNumBufs = 2;   // double buffer
constexpr int kPairsPerThread = 4;  // 128 half2 pairs / 32 lanes

// Store-kind per pair, mirroring the production predicates exactly.
enum { KS_NONE = 0, KS_LO = 1, KS_BOTH = 2 };

// ── dY tile: register-staged load ──────────────────────────────────────
// Lane->address mapping copied verbatim from the production coalesced
// half2-pair scheme (experiment 2026-08-26): thread handles pairs
// q = wtid, wtid+32, wtid+64, wtid+96; pair q covers linear elements
// 2q (row b=q/8, col r=2q%16) and 2q+1 (r+1).
__device__ __forceinline__ void ldg_dy(
    half2* pay, unsigned* mask,
    const half* __restrict__ dY,
    int b0, int tile_b, int r0,
    int batch_size, int out_features, int wtid)
{
    #pragma unroll
    for (int j = 0; j < kPairsPerThread; ++j) {
        int q = wtid + 32 * j;
        int i = q * 2;
        int b = i / kM;
        int r = i % kM;
        mask[j] = KS_NONE;
        if (b >= tile_b) continue;
        int gb = b0 + b;
        int gr = r0 + r;
        if (gb >= batch_size || gr >= out_features) continue;
        int byte_off = (gb * out_features + gr) * (int)sizeof(half);
        if ((byte_off & 3) == 0 && r + 1 < kM && gr + 1 < out_features) {
            pay[j] = ((const half2*)&dY[gb * out_features + gr])[0];
            mask[j] = KS_BOTH;
        } else {
            half lo = dY[gb * out_features + gr];
            half hi = lo;
            if (r + 1 < kM && gr + 1 < out_features) {
                hi = dY[gb * out_features + gr + 1];
                mask[j] = KS_BOTH;
            } else {
                mask[j] = KS_LO;
            }
            pay[j] = __halves2half2(lo, hi);
        }
    }
}

// ── X tile: register-staged load (same mapping, in_features side) ──────
__device__ __forceinline__ void ldg_x(
    half2* pay, unsigned* mask,
    const half* __restrict__ X,
    int b0, int tile_b, int c0,
    int batch_size, int in_features, int wtid)
{
    #pragma unroll
    for (int j = 0; j < kPairsPerThread; ++j) {
        int q = wtid + 32 * j;
        int i = q * 2;
        int b = i / kN;
        int c = i % kN;
        mask[j] = KS_NONE;
        if (b >= tile_b) continue;
        int gb = b0 + b;
        int gc = c0 + c;
        if (gb >= batch_size || gc >= in_features) continue;
        int byte_off = (gb * in_features + gc) * (int)sizeof(half);
        if ((byte_off & 3) == 0 && c + 1 < kN && gc + 1 < in_features) {
            pay[j] = ((const half2*)&X[gb * in_features + gc])[0];
            mask[j] = KS_BOTH;
        } else {
            half lo = X[gb * in_features + gc];
            half hi = lo;
            if (c + 1 < kN && gc + 1 < in_features) {
                hi = X[gb * in_features + gc + 1];
                mask[j] = KS_BOTH;
            } else {
                mask[j] = KS_LO;
            }
            pay[j] = __halves2half2(lo, hi);
        }
    }
}

// ── STS phase: drain registers into the target buffer ──────────────────
__device__ __forceinline__ void sts_tiles(
    half* dys, half* xs,
    const half2* pay_dy, const unsigned* mask_dy,
    const half2* pay_x, const unsigned* mask_x,
    int wtid, int warp_id)
{
    #pragma unroll
    for (int j = 0; j < kPairsPerThread; ++j) {
        int q = wtid + 32 * j;
        int i = q * 2;
        int b_dy = i / kM;
        int r = i % kM;
        if (mask_dy[j] != KS_NONE) {
            half* dst = dys + warp_id * kK * kM + b_dy * kM + r;
            if (mask_dy[j] == KS_BOTH)
                *reinterpret_cast<half2*>(dst) = pay_dy[j];
            else
                *dst = __low2half(pay_dy[j]);
        }
        int b_x = i / kN;
        int c = i % kN;
        if (mask_x[j] != KS_NONE) {
            half* dst = xs + warp_id * kK * kN + b_x * kN + c;
            if (mask_x[j] == KS_BOTH)
                *reinterpret_cast<half2*>(dst) = pay_x[j];
            else
                *dst = __low2half(pay_x[j]);
        }
    }
}

// Zero the unused tail rows (b >= tile_b) of one buffer, CTA-wide.
// Disjoint from the loaded rows (loads skip b >= tile_b), so it may run
// in the same phase as the STS drain into the same buffer; the loop's
// single barrier orders both before the mma that reads the buffer.
__device__ __forceinline__ void zero_fill_bufs(
    half* dys, half* xs, int tile_b)
{
    half zero = __float2half(0.0f);
    int n_total = kWarpsPerBlock * kK * kM;
    for (int tid = threadIdx.x; tid < n_total; tid += 128) {
        int w = tid / (kK * kM);
        int rem = tid % (kK * kM);
        if (rem / kM >= tile_b) dys[w * kK * kM + rem] = zero;
    }
    n_total = kWarpsPerBlock * kK * kN;
    for (int tid = threadIdx.x; tid < n_total; tid += 128) {
        int w = tid / (kK * kN);
        int rem = tid % (kK * kN);
        if (rem / kN >= tile_b) xs[w * kK * kN + rem] = zero;
    }
}

__global__ __launch_bounds__(128) void packed_ternary_update_tc_v2_kernel(
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

    // DOUBLE BUFFER: 2 x (2KB dY + 2KB X) + 4KB dW = 12KB (prod: 8KB).
    __shared__ half   dY_smem[kNumBufs][kWarpsPerBlock * kK * kM];
    __shared__ half   X_smem[kNumBufs][kWarpsPerBlock * kK * kN];
    __shared__ float  dW_float_smem[kWarpsPerBlock * kM * kN];

    wmma::fragment<wmma::matrix_a, kM, kN, kK, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, kM, kN, kK, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, kM, kN, kK, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    const int n_tiles = (batch_size + kK - 1) / kK;
    int cur = 0;

    // ── Prologue: fill buffer 0 with tile 0 (no overlap needed here) ──
    if (n_tiles > 0) {
        half2 pay_dy[kPairsPerThread], pay_x[kPairsPerThread];
        unsigned mask_dy[kPairsPerThread], mask_x[kPairsPerThread];
        int tile_b = min(kK, batch_size);
        ldg_dy(pay_dy, mask_dy, dY, 0, tile_b, r0,
               batch_size, out_features, wtid);
        ldg_x(pay_x, mask_x, X, 0, tile_b, c0,
              batch_size, in_features, wtid);
        sts_tiles(&dY_smem[0][0], &X_smem[0][0],
                  pay_dy, mask_dy, pay_x, mask_x, wtid, warp_id);
        if (tile_b < kK)
            zero_fill_bufs(&dY_smem[0][0], &X_smem[0][0], tile_b);
        __syncthreads();
    }

    // ── Software-pipelined WMMA accumulation over batch tiles ──────────
    // Steady state: [LDG t+1 -> regs] -> [mma t] -> [STS t+1 -> buf nxt]
    //               -> [ONE barrier] -> swap.
    for (int t = 0; t < n_tiles; ++t) {
        int nxt = 1 - cur;
        int t1 = t + 1;
        bool have_next = t1 < n_tiles;
        int b0n = t1 * kK;
        int tile_bn = have_next ? min(kK, batch_size - b0n) : 0;

        // 1. Issue tile t+1 loads into registers. No consumer until the
        //    STS phase, so these stay in flight across the mma below.
        half2 pay_dy[kPairsPerThread], pay_x[kPairsPerThread];
        unsigned mask_dy[kPairsPerThread], mask_x[kPairsPerThread];
        if (have_next) {
            ldg_dy(pay_dy, mask_dy, dY, b0n, tile_bn, r0,
                   batch_size, out_features, wtid);
            ldg_x(pay_x, mask_x, X, b0n, tile_bn, c0,
                  batch_size, in_features, wtid);
        }

        // 2. mma on the full buffer (tile t). Its smem data was made
        //    visible by the barrier at the end of iteration t-1.
        wmma::load_matrix_sync(a_frag, &dY_smem[cur][warp_id * kK * kM], kM);
        wmma::load_matrix_sync(b_frag, &X_smem[cur][warp_id * kK * kN], kN);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // 3. Drain tile t+1 registers into the idle buffer. The first
        //    dependent STS is where any residual load latency stalls the
        //    warp — AFTER the mma has already executed.
        if (have_next) {
            sts_tiles(&dY_smem[nxt][0], &X_smem[nxt][0],
                      pay_dy, mask_dy, pay_x, mask_x, wtid, warp_id);
            if (tile_bn < kK)
                zero_fill_bufs(&dY_smem[nxt][0], &X_smem[nxt][0], tile_bn);
        }

        // 4. THE one barrier per iteration. It orders:
        //    - all warps' STS(t+1)/zero-fill(t+1) writes to buffer nxt
        //      before mma(t+1) reads them, and
        //    - all warps' mma(t) reads of buffer cur before iteration t+2
        //      issues stores into buffer cur (nxt flips each trip).
        __syncthreads();
        cur = nxt;
    }

    // ── Store accumulator to SMEM ───────────────────────────────────
    wmma::store_matrix_sync(&dW_float_smem[warp_id * kM * kN], c_frag,
                            kN, wmma::mem_row_major);
    __syncthreads();

    // ── Counter update: vectorized int32 pairs, skip zero grads ─────
    //  (byte-identical to production from here down)
    int n_pairs = (kWarpsPerBlock * kM * kN) / 2;  // 512
    for (int i = threadIdx.x; i < n_pairs; i += blockDim.x) {
        int pair_w = (i * 2) / (kM * kN);
        int pair_linear = (i * 2) % (kM * kN);
        int r = pair_linear / kN;
        int c = pair_linear % kN;

        int warp_r_off_w = (pair_w % 2) * kM;
        int warp_c_off_w = (pair_w / 2) * kN;
        int gr = super_r0 + warp_r_off_w + r;
        int gc = super_c0 + warp_c_off_w + c;

        if (gr >= out_features || gc + 1 >= in_features) continue;

        float g0 = dW_float_smem[pair_w * kM * kN + r * kN + c];
        float g1 = dW_float_smem[pair_w * kM * kN + r * kN + c + 1];

        if (g0 == 0.0f && g1 == 0.0f) continue;

        int idx = gr * in_features + gc;

        int16_t cnt0, cnt1;
        if ((idx * (int)sizeof(int16_t)) & 3) {
            cnt0 = counter[idx];
            cnt1 = counter[idx + 1];
        } else {
            int32_t cnt_pair = *(const int32_t*)&counter[idx];
            cnt0 = (int16_t)(cnt_pair & 0xFFFF);
            cnt1 = (int16_t)((cnt_pair >> 16) & 0xFFFF);
        }

        cnt0 += (g0 > 0.0f) ? -1 : (g0 < 0.0f) ? 1 : 0;
        cnt1 += (g1 > 0.0f) ? -1 : (g1 < 0.0f) ? 1 : 0;

        uint32_t* w_row = W + gr * stride_words;
        if (cnt0 > threshold) {
            increment_weight_atomic(w_row, gc);
            cnt0 = 0;
        } else if (cnt0 < -threshold) {
            decrement_weight_atomic(w_row, gc);
            cnt0 = 0;
        }

        if (cnt1 > threshold) {
            increment_weight_atomic(w_row, gc + 1);
            cnt1 = 0;
        } else if (cnt1 < -threshold) {
            decrement_weight_atomic(w_row, gc + 1);
            cnt1 = 0;
        }

        if ((idx * (int)sizeof(int16_t)) & 3) {
            counter[idx]     = cnt0;
            counter[idx + 1] = cnt1;
        } else {
            *(int32_t*)&counter[idx] = ((int32_t)cnt1 << 16) | ((int32_t)cnt0 & 0xFFFF);
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
            if (c != (kN - 1)) continue;

            int warp_c_off_w = (w / 2) * kN;
            int gc = super_c0 + warp_c_off_w + c;
            if (gc != last_gc) continue;

            int warp_r_off_w = (w % 2) * kM;
            int gr = super_r0 + warp_r_off_w + r;

            if (gr >= out_features) continue;

            float g = dW_float_smem[w * kM * kN + r * kN + c];
            if (g == 0.0f) continue;

            int idx = gr * in_features + gc;
            int16_t cnt = counter[idx];
            cnt += (g > 0.0f) ? -1 : (g < 0.0f) ? 1 : 0;

            uint32_t* w_row = W + gr * stride_words;
            if (cnt > threshold) {
                increment_weight_atomic(w_row, gc);
                cnt = 0;
            } else if (cnt < -threshold) {
                decrement_weight_atomic(w_row, gc);
                cnt = 0;
            }
            counter[idx] = cnt;
        }
    }
}

extern "C" void launch_packed_ternary_update_tc_v2(
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

    packed_ternary_update_tc_v2_kernel<<<grid, block, 0, stream>>>(
        X, dY, W, counter, batch_size, in_features, out_features,
        stride_words, threshold
    );
}
