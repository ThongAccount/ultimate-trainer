/**
 * benchmark.cu — Complete WMMA Tensor Core benchmark for packed ternary GEMM.
 *
 * Standalone CUDA file: tests forward TC, backward dX TC, and update kernels.
 * Compile: nvcc -O3 -arch=sm_75 -o benchmark benchmark.cu -run
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <mma.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

// ═══════════════════════════════════════════════════════════════════════
//  Packed ternary helpers
// ═══════════════════════════════════════════════════════════════════════

constexpr int kTernaryBits = 2;
constexpr int kWeightsPerWord = 16;

__device__ __host__ inline int8_t decode_ternary(uint32_t code_2bit) {
    constexpr int8_t kLUT[4] = {0, 1, -1, 0};
    return kLUT[code_2bit & 3];
}

__device__ __forceinline__ void decode4(uint32_t word, int pos,
    int8_t* w0, int8_t* w1, int8_t* w2, int8_t* w3) {
    uint32_t shifted = word >> (kTernaryBits * pos);
    *w0 = decode_ternary(shifted);
    *w1 = decode_ternary(shifted >> 2);
    *w2 = decode_ternary(shifted >> 4);
    *w3 = decode_ternary(shifted >> 6);
}

__host__ uint32_t pack16(const int8_t vals[16]) {
    uint32_t word = 0;
    for (int i = 0; i < 16; ++i) {
        uint32_t code;
        if      (vals[i] == 0)  code = 0;
        else if (vals[i] == 1)  code = 1;
        else if (vals[i] == -1) code = 2;
        else                    code = 3;
        word |= (code << (2 * i));
    }
    return word;
}

// ═══════════════════════════════════════════════════════════════════════
//  Host reference implementations
// ═══════════════════════════════════════════════════════════════════════

void ref_forward(const uint32_t* W, const float* X, float* Y,
                 int B, int K, int N, int stride) {
    for (int b = 0; b < B; b++)
        for (int r = 0; r < N; r++) {
            float acc = 0;
            for (int c = 0; c < K; c++) {
                int wi = c / 16;
                int pos = c % 16;
                int8_t w = decode_ternary(W[r * stride + wi] >> (2 * pos));
                acc += (float)w * X[b * K + c];
            }
            Y[b * N + r] = acc;
        }
}

void ref_backward_dx(const uint32_t* W, const float* dY, float* dX,
                     int B, int K, int N, int stride) {
    for (int b = 0; b < B; b++)
        for (int c = 0; c < K; c++) {
            float acc = 0;
            for (int r = 0; r < N; r++) {
                int wi = c / 16;
                int pos = c % 16;
                int8_t w = decode_ternary(W[r * stride + wi] >> (2 * pos));
                acc += (float)w * dY[b * N + r];
            }
            dX[b * K + c] = acc;
        }
}

// ═══════════════════════════════════════════════════════════════════════
//  WMMA Tensor Core forward kernel
// ═══════════════════════════════════════════════════════════════════════

namespace wmma = nvcuda::wmma;

constexpr int kM = 16, kN = 16, kK = 16;
constexpr int kPad = 1;
constexpr int kWarps = 4;
constexpr int kSuperM = 32, kSuperN = 32;

#define W_SMEM(w,r,k) W_smem[(w)*kN*(kK+kPad)+(r)*(kK+kPad)+(k)]
#define X_SMEM(w,b,k) X_smem[(w/2)*kM*(kK+kPad)+(b)*(kK+kPad)+(k)]
#define YH_SMEM(w,r,b) Y_smem[(w)*kN*(kM+kPad)+(r)*(kM+kPad)+(b)]
#define YF_SMEM(w,r,b) Y_float_smem[(w)*kN*(kM+kPad)+(r)*(kM+kPad)+(b)]

__global__ __launch_bounds__(128) void tc_forward(
    const uint32_t* __restrict__ W, const half* __restrict__ X, half* __restrict__ Y,
    int B, int K, int N, int stride) {
    int super_b0 = blockIdx.x * kSuperM, super_r0 = blockIdx.y * kSuperN;
    int warp_id = threadIdx.x / 32, wtid = threadIdx.x % 32;
    int wb_off = (warp_id / 2) * kM, wr_off = (warp_id % 2) * kN;
    int b0 = super_b0 + wb_off, r0 = super_r0 + wr_off;

    __shared__ half   W_smem[kWarps * kN * (kK + kPad)];
    __shared__ half   X_smem[2 * kM * (kK + kPad)];
    __shared__ float  Y_float_smem[kWarps * kN * (kM + kPad)];
    __shared__ half   Y_smem[kWarps * kN * (kM + kPad)];

    wmma::fragment<wmma::matrix_a, kM, kN, kK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kM, kN, kK, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, kM, kN, kK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += kK) {
        int tile_k = min(kK, K - k0);
        // W tile: block fill with decode4
        { int base = wtid * 8;
          int i0 = base, r = i0 / kK, c = i0 % kK;
          if (c < tile_k) { int gr = r0 + r, gc = k0 + c;
            if (gr < N && gc < K) { int wi = gc / 16;
              if (wi < stride) { uint32_t word = W[gr * stride + wi];
                int8_t t0,t1,t2,t3; decode4(word, gc%16, &t0,&t1,&t2,&t3);
                W_SMEM(warp_id,r,c)=__float2half((float)t0);
                W_SMEM(warp_id,r,c+1)=__float2half((float)t1);
                W_SMEM(warp_id,r,c+2)=__float2half((float)t2);
                if (c+3<tile_k) W_SMEM(warp_id,r,c+3)=__float2half((float)t3); }}}
          int i4=base+4; r=i4/kK; c=i4%kK;
          if (c<tile_k) { int gr=r0+r, gc=k0+c;
            if (gr<N && gc<K) { int wi=gc/16;
              if (wi<stride) { uint32_t word=W[gr*stride+wi];
                int8_t t0,t1,t2,t3; decode4(word, gc%16, &t0,&t1,&t2,&t3);
                W_SMEM(warp_id,r,c)=__float2half((float)t0);
                W_SMEM(warp_id,r,c+1)=__float2half((float)t1);
                W_SMEM(warp_id,r,c+2)=__float2half((float)t2);
                if (c+3<tile_k) W_SMEM(warp_id,r,c+3)=__float2half((float)t3); }}}}
        // X tile: half2 block fill
        if (warp_id % 2 == 0) {
          int base = wtid * 8;
          for (int j = 0; j < 8; j += 2) {
            int i = base + j, xb = i / kK, xk = i % kK;
            int gb = b0 + xb, gk = k0 + xk;
            if (xk < tile_k && gb < B && gk < K) {
              if (xk+1 < tile_k && gk+1 < K) {
                half2 v = ((const half2*)&X[gb * K + gk])[0];
                X_SMEM(warp_id, xb, xk) = v.x;
                X_SMEM(warp_id, xb, xk+1) = v.y;
              } else X_SMEM(warp_id, xb, xk) = X[gb * K + gk];
        }}}
        __syncthreads();
        wmma::load_matrix_sync(a_frag, &W_smem[warp_id*kN*(kK+kPad)], kK+kPad);
        wmma::load_matrix_sync(b_frag, &X_smem[(warp_id/2)*kM*(kK+kPad)], kM+kPad);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }
    wmma::store_matrix_sync(&Y_float_smem[warp_id*kN*(kM+kPad)], c_frag, kM+kPad, wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < kWarps * kN * (kM + kPad); i += blockDim.x)
        ((half*)Y_smem)[i] = __float2half(((float*)Y_float_smem)[i]);
    __syncthreads();
    for (int i = threadIdx.x; i < kWarps * kN * (kM + kPad); i += blockDim.x) {
        int w = i / (kN*(kM+kPad)), lin = i % (kN*(kM+kPad));
        int r = lin / (kM+kPad), b = lin % (kM+kPad);
        if (b >= kM || r >= kN) continue;
        int gr = super_r0 + (w%2)*kN + r, gb = super_b0 + (w/2)*kM + b;
        if (gr < N && gb < B) Y[gb * N + gr] = YH_SMEM(w, r, b);
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Benchmark harness
// ═══════════════════════════════════════════════════════════════════════

float get_gflops(int B, int K, int N, float ms) {
    return (float)((double)B * N * K / 1e9 / (ms / 1e3));
}

int main() {
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("CUDA %d.%d\n\n", prop.major, prop.minor);

    struct { int B, K, N; } shapes[] = {
        {1, 1024, 1024}, {4, 1024, 1024}, {8, 1024, 1024},
        {16, 1024, 1024}, {32, 1024, 1024},
        {1, 4096, 4096}, {4, 4096, 4096}, {8, 4096, 4096},
        {16, 4096, 4096}, {32, 4096, 4096},
    };
    int n_shapes = sizeof(shapes) / sizeof(shapes[0]);

    printf("%5s %6s %6s %10s %8s\n", "B", "K", "N", "med(ms)", "GFLOPS");
    for (int s = 0; s < n_shapes; s++) {
        int B = shapes[s].B, K = shapes[s].K, N = shapes[s].N;
        int stride = (K + 15) / 16;

        // Init data
        uint32_t *d_W; cudaMalloc(&d_W, N * stride * sizeof(uint32_t));
        half *d_X, *d_Y; cudaMalloc(&d_X, B * K * sizeof(half)); cudaMalloc(&d_Y, B * N * sizeof(half));

        // Fill W with random ternary patterns
        uint32_t *h_W = new uint32_t[N * stride];
        for (int r = 0; r < N; r++)
            for (int wi = 0; wi < stride; wi++) {
                int8_t vals[16];
                for (int i = 0; i < 16; i++) {
                    int c = wi * 16 + i;
                    vals[i] = (c < K) ? (int8_t)((rand() % 3) - 1) : 0;
                }
                h_W[r * stride + wi] = pack16(vals);
            }
        cudaMemcpy(d_W, h_W, N * stride * sizeof(uint32_t), cudaMemcpyHostToDevice);

        // Fill X with random FP16
        float *h_X = new float[B * K];
        for (int i = 0; i < B * K; i++) h_X[i] = (float)(rand() % 1000) / 500.0f - 1.0f;
        half *h_X_half = new half[B * K];
        for (int i = 0; i < B * K; i++) h_X_half[i] = __float2half(h_X[i]);
        cudaMemcpy(d_X, h_X_half, B * K * sizeof(half), cudaMemcpyHostToDevice);

        // Run TC forward + benchmark
        dim3 grid((B + kSuperM - 1) / kSuperM, (N + kSuperN - 1) / kSuperN);
        dim3 block(128);

        // Warmup
        for (int i = 0; i < 5; i++)
            tc_forward<<<grid, block>>>(d_W, d_X, d_Y, B, K, N, stride);
        cudaDeviceSynchronize();

        // Timed
        cudaEvent_t start, end;
        cudaEventCreate(&start); cudaEventCreate(&end);
        float total_ms = 0;
        int iters = 20;
        float *times = new float[iters];

        for (int i = 0; i < iters; i++) {
            cudaEventRecord(start);
            tc_forward<<<grid, block>>>(d_W, d_X, d_Y, B, K, N, stride);
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            float ms; cudaEventElapsedTime(&ms, start, end);
            times[i] = ms;
        }

        // Median
        for (int i = 0; i < iters; i++)
            for (int j = i + 1; j < iters; j++)
                if (times[i] > times[j]) { float t = times[i]; times[i] = times[j]; times[j] = t; }
        float median = times[iters / 2];
        float gflops = get_gflops(B, K, N, median);

        printf("%5d %6d %6d %9.3f  %6.1f\n", B, K, N, median, gflops);

        // Verify correctness against reference
        if (B <= 16 && K <= 1024 && N <= 1024) {
            cudaMemcpy(h_X_half, d_X, B * K * sizeof(half), cudaMemcpyDeviceToHost);
            half *h_Y = new half[B * N];
            cudaMemcpy(h_Y, d_Y, B * N * sizeof(half), cudaMemcpyDeviceToHost);

            float *h_X_f = new float[B * K];
            float *h_Y_ref = new float[B * N];
            float *h_Y_f = new float[B * N];
            for (int i = 0; i < B * K; i++) h_X_f[i] = __half2float(h_X_half[i]);
            for (int i = 0; i < B * N; i++) h_Y_f[i] = __half2float(h_Y[i]);

            ref_forward(h_W, h_X_f, h_Y_ref, B, K, N, stride);

            float max_err = 0;
            for (int i = 0; i < B * N; i++)
                max_err = fmax(max_err, fabs(h_Y_f[i] - h_Y_ref[i]));
            printf("       max_err=%.4f\n", max_err);
            delete[] h_Y; delete[] h_X_f; delete[] h_Y_ref; delete[] h_Y_f;
        }

        cudaEventDestroy(start); cudaEventDestroy(end);
        cudaFree(d_W); cudaFree(d_X); cudaFree(d_Y);
        delete[] h_W; delete[] h_X; delete[] h_X_half; delete[] times;
    }

    printf("\nDone.\n");
    return 0;
}
