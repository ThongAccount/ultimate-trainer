/**
 * addsub.cu — GPU-accelerated vector addition & subtraction kernels.
 *
 * Standard CPU vector addition (a[i] + b[i]) saturates at ~2–8 GB/s memory
 * bandwidth on modern CPUs, while a T4 GPU delivers ~300 GB/s HBM bandwidth.
 * These kernels exploit that bandwidth gap for arbitrary-sized float vectors.
 *
 * Kernels provided:
 *   vec_add(a, b, c, N)        — c[i] = a[i] + b[i]
 *   vec_sub(a, b, c, N)        — c[i] = a[i] - b[i]
 *   vec_add_sub(a, b, add, sub, N) — fused: add[i]=a[i]+b[i], sub[i]=a[i]-b[i]
 *
 * All use grid-stride loops so a single kernel launch handles arbitrarily
 * large vectors without exceeding the block-grid dimension limits.
 *
 * Compile (CUDA 12.x / 13.x):
 *   nvcc -O3 -arch=sm_75 -o libaddsub.so --shared -Xcompiler -fPIC addsub.cu
 *
 * Or via PyTorch's JIT:
 *   from torch.utils.cpp_extension import load_inline
 *
 * Benchmark (after compiling):
 *   nvcc -O3 -arch=sm_75 -o bench_addsub addsub.cu -lcudart
 *   ./bench_addsub  # runs 1B-element benchmark
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

// ── Error-checking macro ────────────────────────────────────────────

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                       \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

// ── Kernels ──────────────────────────────────────────────────────────

/**
 * vec_add_kernel — element-wise addition c[i] = a[i] + b[i].
 *
 * Grid-stride loop: each thread processes multiple consecutive elements,
 * stepping by gridDim.x * blockDim.x (the total number of active threads).
 * This eliminates the need to launch N/256 blocks and keeps the launch
 * footprint small regardless of vector size.
 */
__global__ void vec_add_kernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                const size_t N) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;

  for (size_t i = tid; i < N; i += stride) {
    c[i] = a[i] + b[i];
  }
}

/**
 * vec_sub_kernel — element-wise subtraction c[i] = a[i] - b[i].
 */
__global__ void vec_sub_kernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                const size_t N) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;

  for (size_t i = tid; i < N; i += stride) {
    c[i] = a[i] - b[i];
  }
}

/**
 * vec_add_sub_kernel — fused addition & subtraction.
 *
 * Computes both a+b and a-b in a single kernel launch, streaming `a` and
 * `b` from HBM only once instead of twice (once for add, once for sub).
 * This halves the memory traffic for common fused operations.
 */
__global__ void vec_add_sub_kernel(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ add,
                                    float* __restrict__ sub,
                                    const size_t N) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;

  for (size_t i = tid; i < N; i += stride) {
    float ai = a[i];
    float bi = b[i];
    add[i] = ai + bi;
    sub[i] = ai - bi;
  }
}

// ── Host API (extern "C" for ctypes / PyTorch loading) ───────────────

extern "C" {

/**
 * Launch vec_add kernel on the default stream.
 *
 * Args:
 *   a, b  — device pointers (float, N elements)
 *   c     — device output pointer
 *   N     — number of elements
 *   stream — CUDA stream (0 for default)
 */
void vec_add(const float* a, const float* b, float* c, size_t N,
             cudaStream_t stream = 0) {
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  blocks = min(blocks, 65535);  // stay within launch limits
  vec_add_kernel<<<blocks, threads, 0, stream>>>(a, b, c, N);
}

void vec_sub(const float* a, const float* b, float* c, size_t N,
             cudaStream_t stream = 0) {
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  blocks = min(blocks, 65535);
  vec_sub_kernel<<<blocks, threads, 0, stream>>>(a, b, c, N);
}

void vec_add_sub(const float* a, const float* b, float* add, float* sub,
                 size_t N, cudaStream_t stream = 0) {
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  blocks = min(blocks, 65535);
  vec_add_sub_kernel<<<blocks, threads, 0, stream>>>(a, b, add, sub, N);
}

}  // extern "C"

// ── Standalone benchmark ────────────────────────────────────────────

#ifndef BUILD_AS_SHARED

double bandwidth_gb_s(size_t bytes, double seconds) {
  return (double)bytes / (1e9 * seconds);
}

int main() {
  printf("╔══════════════════════════════════════════════════╗\n");
  printf("║  CUDA Vector Add/Sub Benchmark                  ║\n");
  printf("╚══════════════════════════════════════════════════╝\n\n");

  int dev;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("Device       : %s\n", prop.name);
  printf("Compute Cap  : %d.%d\n", prop.major, prop.minor);
  printf("HBM size     : %.1f GB\n\n", prop.totalGlobalMem / 1e9);

  // Benchmark parameters
  const size_t N = 1UL << 28;          // ~268M elements = 1 GB per vector
  const size_t bytes_per_vec = N * sizeof(float);
  const int warmup = 10;
  const int iters = 30;

  printf("Vector size  : %zu elements (%.2f GB each)\n", N,
         bytes_per_vec / 1e9);
  printf("Warmup       : %d iterations\n", warmup);
  printf("Measure      : %d iterations\n\n", iters);

  // Allocate
  float *d_a, *d_b, *d_add, *d_sub;
  CUDA_CHECK(cudaMalloc(&d_a, bytes_per_vec));
  CUDA_CHECK(cudaMalloc(&d_b, bytes_per_vec));
  CUDA_CHECK(cudaMalloc(&d_add, bytes_per_vec));
  CUDA_CHECK(cudaMalloc(&d_sub, bytes_per_vec));

  // Initialize with known values
  float* h_a = (float*)malloc(bytes_per_vec);
  float* h_b = (float*)malloc(bytes_per_vec);
  for (size_t i = 0; i < N; i++) {
    h_a[i] = (float)(i % 1000);
    h_b[i] = (float)((i * 7) % 1000);
  }
  CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes_per_vec, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes_per_vec, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // ── Benchmark: vec_add ──
  for (int i = 0; i < warmup; i++)
    vec_add(d_a, d_b, d_add, N, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++)
    vec_add(d_a, d_b, d_add, N, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto t1 = std::chrono::high_resolution_clock::now();
  double add_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  // Read: a + b = 2 reads, write: c = 1 write => 3 * bytes_per_vec traffic
  double add_bw = bandwidth_gb_s(3 * bytes_per_vec, add_ms / 1000.0);
  printf("vec_add       : %8.2f ms  (%7.1f GB/s)  c[i] = a[i] + b[i]\n",
         add_ms, add_bw);

  // ── Benchmark: vec_sub ──
  for (int i = 0; i < warmup; i++)
    vec_sub(d_a, d_b, d_sub, N, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++)
    vec_sub(d_a, d_b, d_sub, N, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  t1 = std::chrono::high_resolution_clock::now();
  double sub_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  double sub_bw = bandwidth_gb_s(3 * bytes_per_vec, sub_ms / 1000.0);
  printf("vec_sub       : %8.2f ms  (%7.1f GB/s)  c[i] = a[i] - b[i]\n",
         sub_ms, sub_bw);

  // ── Benchmark: vec_add_sub (fused) ──
  for (int i = 0; i < warmup; i++)
    vec_add_sub(d_a, d_b, d_add, d_sub, N, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++)
    vec_add_sub(d_a, d_b, d_add, d_sub, N, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  t1 = std::chrono::high_resolution_clock::now();
  double fused_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  // Read: a + b = 2 reads, write: add + sub = 2 writes => 4 * bytes_per_vec
  double fused_bw = bandwidth_gb_s(4 * bytes_per_vec, fused_ms / 1000.0);
  printf("vec_add_sub   : %8.2f ms  (%7.1f GB/s)  fused a±b\n",
         fused_ms, fused_bw);

  // ── Validation ──
  float* h_add = (float*)malloc(bytes_per_vec);
  float* h_sub = (float*)malloc(bytes_per_vec);
  CUDA_CHECK(cudaMemcpy(h_add, d_add, bytes_per_vec, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_sub, d_sub, bytes_per_vec, cudaMemcpyDeviceToHost));

  int errors = 0;
  for (size_t i = 0; i < N && errors < 5; i++) {
    float expected_add = h_a[i] + h_b[i];
    float expected_sub = h_a[i] - h_b[i];
    if (fabsf(h_add[i] - expected_add) > 1e-5f ||
        fabsf(h_sub[i] - expected_sub) > 1e-5f) {
      printf("Mismatch at [%zu]: add=%f (expected %f), sub=%f (expected %f)\n",
             i, h_add[i], expected_add, h_sub[i], expected_sub);
      errors++;
    }
  }

  printf("\n");
  if (errors == 0) {
    printf("✅ All results verified — zero errors\n");
  } else {
    printf("❌ %d errors found\n", errors);
  }

  // ── Cleanup ──
  free(h_a);
  free(h_b);
  free(h_add);
  free(h_sub);
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_add));
  CUDA_CHECK(cudaFree(d_sub));

  return errors ? EXIT_FAILURE : EXIT_SUCCESS;
}

#endif  // BUILD_AS_SHARED
