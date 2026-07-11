/**
 * subqsa_combine_kernel.cu -- Fused gate MLP -> 3-way blend -> RMSNorm -> O projection.
 *
 * Grid: (B, T)  -- one thread block per token position.
 * Block: 256 threads.
 * Shared mem (dynamic): D * sizeof(float) + max(12*H, THREADS/WARP_SIZE*4) bytes.
 *
 * Shared memory layout:
 *   [0 .. D*sizeof(float))        : s_blended (float[D]) -- phases 5-8
 *     sub-range [0 .. D*sizeof(half))   : s_x (half[D])     -- phase 1
 *     sub-range [D*sizeof(half) .. D*sizeof(half)+GATE_HIDDEN*4) : s_hidden -- phase 2-3
 *   [D*sizeof(float) .. end]      : s_gate (float[3*H])  -- phases 3-5
 *                                    / s_reduce           -- phase 6
 *
 * Computation per token:
 *   1. gate MLP:     Linear(D -> 64) -> SiLU -> Linear(64 -> 3*H)
 *   2. per-head sigmoid + L1 normalize across 3 branches
 *   3. weighted blend: g_cmp*o_cmp + g_slc*o_slc + g_win*o_win
 *   4. RMSNorm on blended output
 *   5. Ternary O projection (inline quantisation)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// Error-checking macro for host launch helpers
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                       \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)
#endif

#define GATE_HIDDEN 64
#define THREADS 256
#define WARP_SIZE 32

// ═══════════════════════════════════════════════════════════════════════════
//  Forward kernel
// ═══════════════════════════════════════════════════════════════════════════

__global__ void subqsa_combine_kernel(
    const half* __restrict__ x,
    const half* __restrict__ o_cmp,
    const half* __restrict__ o_slc,
    const half* __restrict__ o_win,
    const half* __restrict__ gate_w1,
    const half* __restrict__ gate_w2,
    const half* __restrict__ out_norm_weight,
    const float* __restrict__ o_proj_weight,
    half* __restrict__ y,
    float gamma,
    int B, int T, int H, int D)
{
    int b = blockIdx.x;
    int t = blockIdx.y;
    if (b >= B || t >= T) return;

    int tid = threadIdx.x;
    int D_head = D / H;

    extern __shared__ char shared_mem[];

    // ── Phase 1: Load x[b, t, :] into shared memory ──────────────────
    half* s_x = reinterpret_cast<half*>(shared_mem);
    long x_base = (long)b * T * D + (long)t * D;
    for (int i = tid; i < D; i += THREADS) {
        s_x[i] = __ldg(&x[x_base + i]);
    }
    __syncthreads();

    // ── Phase 2: Gate MLP Layer 1: Linear(D -> 64) + SiLU ───────────
    float* s_hidden = reinterpret_cast<float*>(shared_mem + D * sizeof(half));
    if (tid < GATE_HIDDEN) {
        float sum = 0.0f;
        for (int j = 0; j < D; j++) {
            sum += __half2float(__ldg(&gate_w1[(long)tid * D + j]))
                 * __half2float(s_x[j]);
        }
        // SiLU: x * sigmoid(x)
        sum = sum * (1.0f / (1.0f + expf(-sum)));
        s_hidden[tid] = sum;
    }
    __syncthreads();

    // ── Phase 3: Gate MLP Layer 2: Linear(64 -> 3*H) ────────────────
    float* s_gate = reinterpret_cast<float*>(shared_mem + D * sizeof(float));
    int gate_size = 3 * H;
    for (int i = tid; i < gate_size; i += THREADS) {
        float sum = 0.0f;
        for (int j = 0; j < GATE_HIDDEN; j++) {
            sum += __half2float(__ldg(&gate_w2[(long)i * GATE_HIDDEN + j]))
                 * s_hidden[j];
        }
        s_gate[i] = sum;
    }
    __syncthreads();

    // ── Phase 4: Per-head sigmoid + L1 normalise across 3 branches ──
    for (int h = tid; h < H; h += THREADS) {
        float g0 = 1.0f / (1.0f + expf(-s_gate[h * 3 + 0]));
        float g1 = 1.0f / (1.0f + expf(-s_gate[h * 3 + 1]));
        float g2 = 1.0f / (1.0f + expf(-s_gate[h * 3 + 2]));
        float sum = g0 + g1 + g2 + 1e-8f;
        s_gate[h * 3 + 0] = g0 / sum;
        s_gate[h * 3 + 1] = g1 / sum;
        s_gate[h * 3 + 2] = g2 / sum;
    }
    __syncthreads();

    // ── Phase 5: 3-way blend + accumulate squared sum for RMSNorm ───
    float* s_blended = reinterpret_cast<float*>(shared_mem);
    float local_sum_sq = 0.0f;

    for (int i = tid; i < D; i += THREADS) {
        int h = i / D_head;
        int d_head = i % D_head;

        long o_base = (long)b * H * T * D_head
                    + (long)h * T * D_head
                    + (long)t * D_head
                    + d_head;

        float v_cmp = __half2float(__ldg(&o_cmp[o_base]));
        float v_slc = __half2float(__ldg(&o_slc[o_base]));
        float v_win = __half2float(__ldg(&o_win[o_base]));

        float g_cmp_val = s_gate[h * 3 + 0];
        float g_slc_val = s_gate[h * 3 + 1];
        float g_win_val = s_gate[h * 3 + 2];

        float blended = g_cmp_val * v_cmp + g_slc_val * v_slc + g_win_val * v_win;
        s_blended[i] = blended;
        local_sum_sq += blended * blended;
    }

    // ── Phase 6: Parallel reduction for RMS ──────────────────────────
    float sum_sq = local_sum_sq;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
    }

    // Reuse s_gate space for reduction buffer (safe: s_gate consumed)
    float* s_reduce = reinterpret_cast<float*>(shared_mem + D * sizeof(float));
    if (tid % WARP_SIZE == 0) {
        s_reduce[tid / WARP_SIZE] = sum_sq;
    }
    __syncthreads();

    if (tid < THREADS / WARP_SIZE) {
        sum_sq = s_reduce[tid];
        for (int offset = (THREADS / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
        }
        if (tid == 0) {
            s_reduce[0] = sum_sq / D;
        }
    }
    __syncthreads();

    float rms = sqrtf(s_reduce[0] + 1e-5f);

    // ── Phase 7: RMSNorm scaling ─────────────────────────────────────
    for (int i = tid; i < D; i += THREADS) {
        s_blended[i] = s_blended[i] / rms * __half2float(__ldg(&out_norm_weight[i]));
    }
    __syncthreads();

    // ── Phase 8: Ternary O projection ────────────────────────────────
    // y[out] = sum_{in} blended[in] * qweight[out,in]
    // where qweight = clamp(round(o_proj_weight / gamma), -1, 1) * gamma
    long y_base = (long)b * T * D + (long)t * D;
    for (int out_idx = tid; out_idx < D; out_idx += THREADS) {
        float sum = 0.0f;
        for (int in_idx = 0; in_idx < D; in_idx++) {
            float w = __ldg(&o_proj_weight[(long)out_idx * D + in_idx]);
            float w_q = roundf(w / gamma);
            w_q = fminf(fmaxf(w_q, -1.0f), 1.0f);
            w_q = w_q * gamma;
            sum += s_blended[in_idx] * w_q;
        }
        y[y_base + out_idx] = __float2half(sum);
    }
}


// ═══════════════════════════════════════════════════════════════════════════
//  Host launch functions (extern "C" for PyTorch JIT / ctypes)
// ═══════════════════════════════════════════════════════════════════════════

extern "C" {

void launch_subqsa_combine_forward(
    const half* x,
    const half* o_cmp,
    const half* o_slc,
    const half* o_win,
    const half* gate_w1,
    const half* gate_w2,
    const half* out_norm_weight,
    const float* o_proj_weight,
    half* y,
    float gamma,
    int B, int T, int H, int D,
    cudaStream_t stream)
{
    // Shared memory: max of
    //   phase 1-4: D*half + 64*float + 3*H*float = 2*D + 256 + 12*H
    //   phase 5-8: D*float + 8*float               = 4*D + 32
    size_t phase1_size = (size_t)D * sizeof(half) + GATE_HIDDEN * sizeof(float) + (size_t)(3 * H) * sizeof(float);
    size_t phase5_size = (size_t)D * sizeof(float) + (THREADS / WARP_SIZE) * sizeof(float);
    size_t shared_mem = phase1_size > phase5_size ? phase1_size : phase5_size;

    dim3 grid(B, T);
    subqsa_combine_kernel<<<grid, THREADS, shared_mem, stream>>>(
        x, o_cmp, o_slc, o_win,
        gate_w1, gate_w2,
        out_norm_weight, o_proj_weight,
        y, gamma,
        B, T, H, D);

    CUDA_CHECK(cudaGetLastError());
}

}  // extern "C"
