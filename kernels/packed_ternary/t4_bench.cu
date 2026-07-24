/**
 * t4_bench.cu — WMMA packed ternary forward GEMM for T4 (NO stride padding).
 *
 * Standalone. Compile: nvcc -O3 -arch=sm_75 -o t4_bench t4_bench.cu -run
 *
 * NOTE: WMMA load/store_matrix_sync REQUIRE stride to be a multiple of 16.
 * The bank-conflict padding (stride 17) approach was INVALID on sm_75.
 * This version uses stride=16 (no padding) for correctness.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <mma.h>
#include <stdio.h>
#include <math.h>

// ── Packed ternary helpers ──────────────────────────────────────────
constexpr int kTernaryBits = 2;
constexpr int kWeightsPerWord = 16;

__device__ __host__ inline int8_t decode_ternary(uint32_t code_2bit) {
    constexpr int8_t LUT[4] = {0, 1, -1, 0};
    return LUT[code_2bit & 3];
}

#ifdef __CUDACC__
__device__ __forceinline__ void decode4(uint32_t word, int pos,
    int8_t* w0, int8_t* w1, int8_t* w2, int8_t* w3) {
    uint32_t s = word >> (kTernaryBits * pos);
    *w0 = decode_ternary(s); *w1 = decode_ternary(s >> 2);
    *w2 = decode_ternary(s >> 4); *w3 = decode_ternary(s >> 6);
}
#endif

uint32_t pack16(const int8_t vals[16]) {
    uint32_t word = 0;
    for (int i = 0; i < 16; ++i) {
        uint32_t c = vals[i] == 0 ? 0 : vals[i] == 1 ? 1 : vals[i] == -1 ? 2 : 3;
        word |= (c << (2 * i));
    }
    return word;
}

// ── Host reference ──────────────────────────────────────────────────
void ref_fwd(const uint32_t* W, const float* X, float* Y, int B, int K, int N, int stride) {
    for (int b = 0; b < B; b++)
        for (int r = 0; r < N; r++) {
            float acc = 0;
            for (int c = 0; c < K; c++) {
                int wi = c / 16;
                int8_t w = decode_ternary(W[r * stride + wi] >> (2 * (c % 16)));
                acc += (float)w * X[b * K + c];
            }
            Y[b * N + r] = acc;
        }
}

// ── WMMA TC forward kernel (stride = kK = 16 — multiple of 16 ✓) ────
namespace wmma = nvcuda::wmma;
constexpr int kM = 16, kN = 16, kK = 16;

__global__ __launch_bounds__(256) void tc_forward(
    const uint32_t* __restrict__ W, const half* __restrict__ X, half* __restrict__ Y,
    int B, int K, int N, int stride_words)
{
    int b0 = blockIdx.x * kM, r0 = blockIdx.y * kN;
    int tid = threadIdx.x;  // 0..255

    __shared__ half   W_smem[kN][kK];
    __shared__ half   X_smem[kM][kK];
    __shared__ float  Y_float[kN][kM];
    __shared__ half   Y_half[kN][kM];

    wmma::fragment<wmma::matrix_a, kM, kN, kK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kM, kN, kK, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, kM, kN, kK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += kK) {
        int tile_k = min(kK, K - k0);

        // W tile: 1 element per thread (256 threads, 16x16)
        int r = tid / kK, c = tid % kK;
        half wv = __float2half(0.0f);
        if (r < kN && c < tile_k) {
            int gr = r0 + r, gc = k0 + c;
            if (gr < N && gc < K) {
                int wi = gc / kWeightsPerWord;
                if (wi < stride_words) {
                    uint32_t word = W[gr * stride_words + wi];
                    int pos = gc % kWeightsPerWord;
                    int8_t t = decode_ternary(word >> (kTernaryBits * pos));
                    wv = __float2half((float)t);
                }
            }
        }
        W_smem[r][c] = wv;

        // X tile: 1 element per thread
        int xb = tid / kK, xk = tid % kK;
        half xv = __float2half(0.0f);
        if (xb < kM && xk < tile_k) {
            int gb = b0 + xb, gk = k0 + xk;
            if (gb < B && gk < K) xv = X[gb * K + gk];
        }
        X_smem[xb][xk] = xv;
        __syncthreads();

        wmma::load_matrix_sync(a_frag, &W_smem[0][0], kK);  // stride=16 ✓
        wmma::load_matrix_sync(b_frag, &X_smem[0][0], kM);  // stride=16 ✓
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }
    wmma::store_matrix_sync(&Y_float[0][0], c_frag, kM, wmma::mem_row_major);  // stride=16 ✓
    __syncthreads();

    // Convert float→half
    for (int i = tid; i < kM * kN; i += blockDim.x)
        ((half*)Y_half)[i] = __float2half(((float*)Y_float)[i]);
    __syncthreads();

    // Store to global Y
    int r_out = tid / kM;
    int b = tid % kM;
    int gr = r0 + r_out, gb = b0 + b;
    if (gr < N && gb < B) Y[gb * N + gr] = Y_half[r_out][b];
}

// ── Benchmark ────────────────────────────────────────────────────────
int main() {
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    printf("Device: %s (SM %d.%d)\n", p.name, p.major, p.minor);
    int cuda_ver; cudaRuntimeGetVersion(&cuda_ver);
    printf("CUDA runtime: %d.%d\n\n", cuda_ver / 1000, (cuda_ver % 1000) / 10);

    struct { int B, K, N; } shapes[] = {
        {1, 1024, 1024}, {4, 1024, 1024}, {8, 1024, 1024},
        {16, 1024, 1024}, {32, 1024, 1024},
        {1, 4096, 4096}, {4, 4096, 4096}, {8, 4096, 4096},
        {16, 4096, 4096}, {32, 4096, 4096},
    };
    int ns = sizeof(shapes) / sizeof(shapes[0]);

    printf("%5s %6s %6s %10s %8s %10s\n", "B", "K", "N", "med(ms)", "GFLOPS", "max_err");
    for (int s = 0; s < ns; s++) {
        int B = shapes[s].B, K = shapes[s].K, N = shapes[s].N;
        int stride = (K + 15) / 16;

        uint32_t *d_W; cudaMalloc(&d_W, N * stride * sizeof(uint32_t));
        half *d_X, *d_Y;
        cudaMalloc(&d_X, B * K * sizeof(half));
        cudaMalloc(&d_Y, B * N * sizeof(half));

        uint32_t *h_W = new uint32_t[N * stride];
        for (int r = 0; r < N; r++)
            for (int wi = 0; wi < stride; wi++) {
                int8_t vals[16];
                for (int i = 0; i < 16; i++)
                    vals[i] = (wi * 16 + i < K) ? (int8_t)((rand() % 3) - 1) : 0;
                h_W[r * stride + wi] = pack16(vals);
            }
        cudaMemcpy(d_W, h_W, N * stride * sizeof(uint32_t), cudaMemcpyHostToDevice);

        half *h_X = new half[B * K];
        for (int i = 0; i < B * K; i++)
            h_X[i] = __float2half(((float)(rand() % 1000)) / 500.0f - 1.0f);
        cudaMemcpy(d_X, h_X, B * K * sizeof(half), cudaMemcpyHostToDevice);

        dim3 grid((B + kM - 1) / kM, (N + kN - 1) / kN);
        dim3 block(256);

        // Warmup
        for (int i = 0; i < 5; i++)
            tc_forward<<<grid, block>>>(d_W, d_X, d_Y, B, K, N, stride);
        cudaDeviceSynchronize();

        cudaEvent_t e1, e2;
        cudaEventCreate(&e1); cudaEventCreate(&e2);
        float times[20];
        for (int i = 0; i < 20; i++) {
            cudaEventRecord(e1);
            tc_forward<<<grid, block>>>(d_W, d_X, d_Y, B, K, N, stride);
            cudaEventRecord(e2);
            cudaEventSynchronize(e2);
            cudaEventElapsedTime(&times[i], e1, e2);
        }

        // Median
        for (int i = 0; i < 20; i++)
            for (int j = i + 1; j < 20; j++)
                if (times[i] > times[j]) { float t = times[i]; times[i] = times[j]; times[j] = t; }
        float median = times[10];

        // Verify
        half *h_Y = new half[B * N];
        cudaMemcpy(h_Y, d_Y, B * N * sizeof(half), cudaMemcpyDeviceToHost);
        float *h_Xf = new float[B * K], *h_Y_ref = new float[B * N];
        for (int i = 0; i < B * K; i++) h_Xf[i] = __half2float(h_X[i]);
        ref_fwd(h_W, h_Xf, h_Y_ref, B, K, N, stride);
        float max_err = 0;
        for (int i = 0; i < B * N; i++)
            max_err = fmaxf(max_err, fabsf(__half2float(h_Y[i]) - h_Y_ref[i]));

        float gflops = (double)B * N * K / 1e9 / (median / 1e3);
        printf("%5d %6d %6d %9.3f  %6.1f  %.6f\n", B, K, N, median, gflops, max_err);

        cudaEventDestroy(e1); cudaEventDestroy(e2);
        cudaFree(d_W); cudaFree(d_X); cudaFree(d_Y);
        delete[] h_W; delete[] h_X; delete[] h_Y; delete[] h_Xf; delete[] h_Y_ref;
    }
    printf("\nDone.\n");
    return 0;
}
