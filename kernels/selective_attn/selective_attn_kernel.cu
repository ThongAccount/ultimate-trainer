// selective_attn_kernel.cu
// Two-phase fused selective attention kernel.
// Phase 1: Top-K selection from scores_agg using iterative max-reduction.
// Phase 2: Causal FlashAttention over selected K/V blocks (online softmax,
//          causal mask via position comparison, no explicit mask tensor).
//
// Grid layout:
//   Phase 1: (B, H)  — each block selects top-K indices for one (b,h)
//   Phase 2: (B, H)  — each block computes causal attention for one (b,h)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// ────────────────────────────────────────────────────────────────────────────
// Phase 1: Top-K Selection
// ────────────────────────────────────────────────────────────────────────────
//
// Grid: (B, H)
// Each block loads scores_agg[n_sel] into shared memory, then iteratively
// finds the maximum score (serial scan by thread 0), records the index,
// and sets that score to -inf.  topk_actual = min(topk, n_sel) prevents
// selecting garbage when topk exceeds the number of available blocks.

__global__ void fused_selective_attn_phase1_kernel(
    const float* __restrict__ scores_agg,  // (B, H, n_sel)
    long* __restrict__ top_idx,            // (B, H, topk) output indices
    int B, int H, int n_sel, int topk
) {
    int pid_b = blockIdx.x;
    int pid_h = blockIdx.y;

    if (pid_b >= B || pid_h >= H) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    extern __shared__ char shared_mem[];
    float* s_scores = (float*)shared_mem;
    int* s_indices = (int*)(s_scores + n_sel);

    long base = (long)pid_b * H * n_sel + (long)pid_h * n_sel;

    // Load scores and indices into shared memory
    for (int i = tid; i < n_sel; i += nthreads) {
        s_scores[i] = __ldg(&scores_agg[base + i]);
        s_indices[i] = i;
    }
    __syncthreads();

    int topk_actual = topk < n_sel ? topk : n_sel;

    for (int k = 0; k < topk_actual; k++) {
        // Thread 0 does a serial scan to find the max
        if (tid == 0) {
            float best_val = -1e38f;
            int best_idx = -1;
            for (int i = 0; i < n_sel; i++) {
                if (s_scores[i] > best_val) {
                    best_val = s_scores[i];
                    best_idx = s_indices[i];
                }
            }
            long out_idx = (long)pid_b * H * topk + (long)pid_h * topk + k;
            top_idx[out_idx] = (long)best_idx;
            s_scores[best_idx] = -1e38f;  // prevent reselection
        }
        __syncthreads();
    }
}


// ────────────────────────────────────────────────────────────────────────────
// Phase 2: Causal FlashAttention over Selected Blocks
// ────────────────────────────────────────────────────────────────────────────
//
// Grid: (B, H)
// Each block loads top_idx[topk], then iterates over all T query positions.
// For each query position, attention is computed (online softmax) over the
// selected K/V blocks.  Causal masking is applied by comparing the original
// key position against the query position: if key_pos > query_pos, the score
// is set to -inf (no explicit mask tensor is built).
//
// Shared memory layout:
//   [0 .. D*half)                    : current query vector (half)
//   [D*half .. D*half + nthreads*float) : reduction buffer for dot product
//   [D*half + nthreads*float .. )   : topk indices buffer (long)

__global__ void fused_selective_attn_phase2_kernel(
    const half* __restrict__ q_ptr,        // (B, H, T, D)
    const half* __restrict__ k_ptr,        // (B, H, T, D)
    const half* __restrict__ v_ptr,        // (B, H, T, D)
    const long* __restrict__ top_idx,      // (B, H, topk)
    half* __restrict__ attn_out,           // (B, H, T, D)
    int B, int H, int T, int D,
    int block_size, int topk, int n_sel
) {
    int pid_b = blockIdx.x;
    int pid_h = blockIdx.y;

    if (pid_b >= B || pid_h >= H) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    extern __shared__ char shared_mem[];
    half* s_q = (half*)shared_mem;                              // D * sizeof(half)
    float* s_red = (float*)(s_q + D);                            // nthreads * sizeof(float)
    long* s_topk = (long*)(s_red + nthreads);                    // topk * sizeof(long)

    int topk_actual = topk < n_sel ? topk : n_sel;

    // Load topk indices into shared memory
    long topk_base = (long)pid_b * H * topk + (long)pid_h * topk;
    for (int i = tid; i < topk_actual; i += nthreads) {
        s_topk[i] = __ldg(&top_idx[topk_base + i]);
    }
    __syncthreads();

    long q_base = (long)pid_b * H * T * D + (long)pid_h * T * D;
    long kv_base = (long)pid_b * H * T * D + (long)pid_h * T * D;
    float inv_sqrt_d = rsqrtf((float)D);

    // Iterate over all query positions
    for (int q_pos = 0; q_pos < T; q_pos++) {
        // Load current query vector into shared memory
        if (tid < D) {
            s_q[tid] = __ldg(&q_ptr[q_base + (long)q_pos * D + tid]);
        }
        __syncthreads();

        // Online softmax state (per-thread registers)
        float m = -1e38f;   // running maximum
        float d = 0.0f;     // denominator sum
        float o_acc = 0.0f; // output accumulator (one element of D per thread)

        // Iterate over selected blocks
        for (int sel = 0; sel < topk_actual; sel++) {
            int block_start = (int)s_topk[sel] * block_size;

            // Iterate over each token inside the selected block
            for (int off = 0; off < block_size; off++) {
                int kv_pos = block_start + off;
                if (kv_pos >= T) continue;

                // --- Dot product: q @ k (one element per thread) ---
                float partial = 0.0f;
                if (tid < D) {
                    float qv = __half2float(s_q[tid]);
                    float kv = __half2float(__ldg(&k_ptr[kv_base + (long)kv_pos * D + tid]));
                    partial = qv * kv;
                }
                s_red[tid] = partial;
                __syncthreads();

                // Tree reduction to sum the dot product across threads
                for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
                    if (tid < stride) {
                        s_red[tid] += s_red[tid + stride];
                    }
                    __syncthreads();
                }

                float score = s_red[0] * inv_sqrt_d;

                // Causal mask: prevent attending to future positions
                if (kv_pos > q_pos) {
                    score = -1e38f;
                }

                // Online softmax update
                float m_new = fmaxf(m, score);
                float exp_scale = expf(m - m_new);
                float exp_score = expf(score - m_new);
                d = d * exp_scale + exp_score;

                if (tid < D) {
                    float vv = __half2float(__ldg(&v_ptr[kv_base + (long)kv_pos * D + tid]));
                    o_acc = o_acc * exp_scale + vv * exp_score;
                }
                m = m_new;
                __syncthreads();
            }
        }

        // Write output for this query position
        if (tid < D) {
            attn_out[q_base + (long)q_pos * D + tid] = __float2half(o_acc / (d + 1e-8f));
        }
        __syncthreads();
    }
}


// ────────────────────────────────────────────────────────────────────────────
// Host launch functions
// ────────────────────────────────────────────────────────────────────────────

void launch_selective_phase1(
    const float* scores, long* top_idx,
    int B, int H, int n_sel, int topk,
    cudaStream_t stream) {

    int threads = n_sel < 256 ? 256 : (n_sel < 1024 ? 1024 : 1024);
    if (threads > 1024) threads = 1024;

    // Shared memory: s_scores[n_sel] + s_indices[n_sel]
    size_t shared_mem = (size_t)n_sel * (sizeof(float) + sizeof(int));

    dim3 grid(B, H);
    fused_selective_attn_phase1_kernel<<<grid, threads, shared_mem, stream>>>(
        scores, top_idx, B, H, n_sel, topk);
}


void launch_selective_phase2(
    const float* q_f, const float* k_f, const float* v_f,
    const long* top_idx, float* attn_out_f,
    int B, int H, int T, int D,
    int block_size, int topk, int n_sel,
    cudaStream_t stream) {
    const half* q = reinterpret_cast<const half*>(q_f);
    const half* k = reinterpret_cast<const half*>(k_f);
    const half* v = reinterpret_cast<const half*>(v_f);
    half* attn_out = reinterpret_cast<half*>(attn_out_f);

    // Use enough threads to cover D (the head dimension) for dot products
    int threads = 256;
    if (threads < D) threads = 256;  // 256 >= typical D values (64, 128)

    // Shared memory: s_q[D] + s_red[nthreads] + s_topk[topk]
    size_t shared_mem = (size_t)D * sizeof(half)
                      + (size_t)threads * sizeof(float)
                      + (size_t)topk * sizeof(long);

    dim3 grid(B, H);
    fused_selective_attn_phase2_kernel<<<grid, threads, shared_mem, stream>>>(
        q, k, v, top_idx, attn_out,
        B, H, T, D, block_size, topk, n_sel);
}
