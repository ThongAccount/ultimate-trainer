/*
 * gemm_update_cublas.cu — update re-hosted on cuBLAS (ranked item #1, 2026-10-20).
 *
 * Two pieces, matching gemm_update_tc_v2.cu's semantics exactly:
 *
 *   1. packed_ternary_dw_gemm_f32: dW = dY^T @ X as a cublasGemmEx with
 *      FP16 inputs and FP32 OUTPUT (compute type 32F). fp32-out is required:
 *      torch.matmul on fp16 inputs returns fp16, which rounds dW and can flip
 *      near-zero signs — the exact parity risk Researcher flagged.
 *        dW[n,k] = SUM_b dY[b,n] * X[b,k]
 *      Layout derivation (all buffers row-major torch tensors):
 *        X  [B,K] row-major == col-major [K,B] (lda=K)  -> A, OP_N
 *        dY [B,N] row-major == col-major [N,B] (ldb=N)  -> B, OP_T
 *        dW [N,K] row-major == col-major [K,N] (ldc=K)  -> C, FP32
 *      so m=K, n=N, k=B.
 *
 *   2. packed_ternary_update_epilogue: counter/flip over the fp32 dW, ONE
 *      thread per (n,k), replicating gemm_update_tc_v2.cu lines 186-201:
 *        cnt += (g > 0) ? -1 : (g < 0) ? +1 : 0;      // int16 arithmetic
 *        cnt >  threshold -> increment_weight_atomic, cnt = 0;
 *        cnt < -threshold -> decrement_weight_atomic, cnt = 0;
 *      increment/decrement (ternary ladder -1->0->+1 / +1->0->-1 via
 *      atomicCAS on the packed word) come from packed_ternary.cuh unchanged.
 *
 * This file is additive: nothing in the existing kernel set is modified.
 * Env PT_UPDATE_BACKEND=cublas routes custom_ops.update_tc_v2 here.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cublas_v2.h>
#include "packed_ternary.cuh"

extern "C" {

__global__ void update_epilogue_kernel(
    uint32_t* __restrict__ W,          // [N, stride_words] packed 2-bit
    int16_t*  __restrict__ counter,    // [N, K] int16
    const float* __restrict__ dW,      // [N, K] fp32
    int N, int K, int stride_words, int16_t threshold)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)N * K) return;
    int n = idx / K;
    int k = idx % K;

    float g = dW[idx];
    int16_t cnt = counter[idx];
    cnt += (g > 0.0f) ? -1 : (g < 0.0f) ? 1 : 0;

    uint32_t* w_row = W + (long)n * stride_words;
    if (cnt > threshold) { increment_weight_atomic(w_row, k); cnt = 0; }
    else if (cnt < -threshold) { decrement_weight_atomic(w_row, k); cnt = 0; }

    counter[idx] = cnt;
}

}  // extern "C"

// dW[n,k] = SUM_b dY[b,n] * X[b,k]  — cublasGemmEx, fp16 in, fp32 out.
// See layout derivation in the header comment.
extern "C" void packed_ternary_dw_gemm_f32(
    const half* dY, const half* X, float* dW,
    int B, int N, int K, cudaStream_t stream)
{
    if (B == 0) return;
    // Self-managed handle (at::cuda::getCurrentCUDALibraryHandles is not
    // available under nvcc's include set here); stream set per call.
    static cublasHandle_t s_handle = nullptr;
    if (!s_handle) { cublasCreate(&s_handle); }
    cublasSetStream(s_handle, stream);
    cublasHandle_t handle = s_handle;
    const float alpha = 1.0f, beta = 0.0f;
    // Row-major torch buffers reinterpreted column-major:
    //   A = X  as [K,B] col-major (OP_N, lda=K)
    //   B = dY as [N,B] col-major (OP_T, ldb=N)
    //   C = dW as [K,N] col-major (ldc=K), fp32
    cublasStatus_t st = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        K, N, B,
        &alpha,
        X,  CUDA_R_16F, K,
        dY, CUDA_R_16F, N,
        &beta,
        dW, CUDA_R_32F, K,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
                "cublasGemmEx dw_gemm_f32 failed: ", (int)st);
}

extern "C" void launch_update_epilogue(
    uint32_t* W, int16_t* counter, const float* dW,
    int N, int K, int stride_words, int16_t threshold, cudaStream_t stream)
{
    long total = (long)N * K;
    int threads = 256;
    long blocks = (total + threads - 1) / threads;
    update_epilogue_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
        W, counter, dW, N, K, stride_words, threshold);
}
