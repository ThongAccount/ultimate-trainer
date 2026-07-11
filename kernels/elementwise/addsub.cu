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
 * Run tests & benchmark (standalone):
 *   nvcc -O3 -arch=sm_75 -o test_addsub addsub.cu -lcudart
 *   ./test_addsub               # benchmark only (default)
 *   ./test_addsub --test        # comprehensive tests only
 *   ./test_addsub --all         # tests + benchmark
 *   ./test_addsub --bench       # benchmark only (same as default)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
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

// ═══════════════════════════════════════════════════════════════════
//  TEST SUITE & BENCHMARK
// ═══════════════════════════════════════════════════════════════════
//
// Compile & run:
//   nvcc -O3 -arch=sm_75 -o test_addsub addsub.cu -lcudart
//   ./test_addsub               # benchmark only (default)
//   ./test_addsub --test        # run all tests
//   ./test_addsub --all         # tests + benchmark
//
// ═══════════════════════════════════════════════════════════════════

#ifndef BUILD_AS_SHARED

double bandwidth_gb_s(size_t bytes, double seconds) {
  return (double)bytes / (1e9 * seconds);
}

// ── Test tracking ──────────────────────────────────────────────────

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, fmt, ...)                                           \
  do {                                                                        \
    if (!(cond)) {                                                            \
      printf("  ❌ FAIL: " fmt " [%s:%d]\n", ##__VA_ARGS__, __FILE__, __LINE__); \
      g_tests_failed++;                                                       \
      return;                                                                 \
    }                                                                         \
  } while (0)

#define TEST_SECTION(name)                                                    \
  printf("\n  ── %s ──\n", name);                                             \
  g_tests_passed = g_tests_failed = 0;

#define TEST_REPORT()                                                         \
  do {                                                                        \
    printf("  ✅ %d passed, %d failed\n", g_tests_passed, g_tests_failed);     \
    g_tests_passed = g_tests_failed = 0;                                      \
  } while (0)

// ── Test helpers ───────────────────────────────────────────────────

static float* run_vec_add(const float* d_a, const float* d_b, size_t N,
                          cudaStream_t stream) {
  float *d_c;
  CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));
  vec_add(d_a, d_b, d_c, N, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float* h_c = (float*)malloc(N * sizeof(float));
  CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_c));
  return h_c;
}

static float* run_vec_sub(const float* d_a, const float* d_b, size_t N,
                          cudaStream_t stream) {
  float *d_c;
  CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));
  vec_sub(d_a, d_b, d_c, N, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float* h_c = (float*)malloc(N * sizeof(float));
  CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_c));
  return h_c;
}

static void run_vec_add_sub(const float* d_a, const float* d_b, size_t N,
                            cudaStream_t stream,
                            float** h_add, float** h_sub) {
  float *d_add, *d_sub;
  CUDA_CHECK(cudaMalloc(&d_add, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sub, N * sizeof(float)));
  vec_add_sub(d_a, d_b, d_add, d_sub, N, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  *h_add = (float*)malloc(N * sizeof(float));
  *h_sub = (float*)malloc(N * sizeof(float));
  CUDA_CHECK(cudaMemcpy(*h_add, d_add, N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(*h_sub, d_sub, N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_add));
  CUDA_CHECK(cudaFree(d_sub));
}

static float* upload_to_gpu(const float* h_data, size_t N, cudaStream_t stream) {
  float* d_data;
  CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, N * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
  return d_data;
}

// ── Test: correctness on tiny vectors ──────────────────────────────

static void test_tiny_vectors(cudaStream_t stream) {
  TEST_SECTION("Tiny vectors (1..256 elements)");

  size_t sizes[] = {1, 2, 3, 7, 13, 31, 64, 128, 256};
  for (int s = 0; s < (int)(sizeof(sizes)/sizeof(sizes[0])); s++) {
    size_t N = sizes[s];
    float* h_a = (float*)malloc(N * sizeof(float));
    float* h_b = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
      h_a[i] = (float)(i * 3 + 1);
      h_b[i] = (float)(i * 2 + 5);
    }

    float* d_a = upload_to_gpu(h_a, N, stream);
    float* d_b = upload_to_gpu(h_b, N, stream);

    float* h_c = run_vec_add(d_a, d_b, N, stream);
    for (size_t i = 0; i < N; i++) {
      TEST_ASSERT(fabsf(h_c[i] - (h_a[i] + h_b[i])) < 1e-6f,
                  "add N=%zu [%zu]: got %f, expected %f", N, i, h_c[i], h_a[i] + h_b[i]);
    }
    free(h_c);

    h_c = run_vec_sub(d_a, d_b, N, stream);
    for (size_t i = 0; i < N; i++) {
      TEST_ASSERT(fabsf(h_c[i] - (h_a[i] - h_b[i])) < 1e-6f,
                  "sub N=%zu [%zu]: got %f, expected %f", N, i, h_c[i], h_a[i] - h_b[i]);
    }
    free(h_c);

    float *h_add, *h_sub;
    run_vec_add_sub(d_a, d_b, N, stream, &h_add, &h_sub);
    for (size_t i = 0; i < N; i++) {
      TEST_ASSERT(fabsf(h_add[i] - (h_a[i] + h_b[i])) < 1e-6f,
                  "fused-add N=%zu [%zu]: got %f, expected %f", N, i, h_add[i], h_a[i] + h_b[i]);
      TEST_ASSERT(fabsf(h_sub[i] - (h_a[i] - h_b[i])) < 1e-6f,
                  "fused-sub N=%zu [%zu]: got %f, expected %f", N, i, h_sub[i], h_a[i] - h_b[i]);
    }
    free(h_add); free(h_sub);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    free(h_a); free(h_b);
    g_tests_passed++;
  }
  TEST_REPORT();
}

// ── Test: edge values (zeros, negatives, large) ────────────────────

static void test_edge_values(cudaStream_t stream) {
  TEST_SECTION("Edge values (zeros, negatives, near limits)");

  struct { float a_val; float b_val; const char* desc; } cases[] = {
    {0.0f, 0.0f, "both zero"},
    {0.0f, 1.0f, "zero + positive"},
    {1.0f, 0.0f, "positive + zero"},
    {0.0f, -1.0f, "zero + negative"},
    {-1.0f, 0.0f, "negative + zero"},
    {1e-10f, 1e-10f, "very small"},
    {1e10f, 1e10f, "very large"},
    {-1e10f, 1e10f, "large negative + large positive"},
    {3.14159265f, 2.71828182f, "pi + e"},
  };

  const size_t N = 1024;
  for (int c = 0; c < (int)(sizeof(cases)/sizeof(cases[0])); c++) {
    float* h_a = (float*)malloc(N * sizeof(float));
    float* h_b = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
      h_a[i] = cases[c].a_val + (float)i * 1e-7f;
      h_b[i] = cases[c].b_val + (float)i * 1e-7f;
    }

    float* d_a = upload_to_gpu(h_a, N, stream);
    float* d_b = upload_to_gpu(h_b, N, stream);

    float* h_c = run_vec_add(d_a, d_b, N, stream);
    for (size_t i = 0; i < N; i++) {
      float expected_a = h_a[i] + h_b[i];
      TEST_ASSERT(fabsf(h_c[i] - expected_a) < 1e-4f * fmaxf(1.0f, fabsf(expected_a)),
                  "add %s [%zu]: got %f, expected %f", cases[c].desc, i, h_c[i], expected_a);
    }
    free(h_c);

    h_c = run_vec_sub(d_a, d_b, N, stream);
    for (size_t i = 0; i < N; i++) {
      float expected_s = h_a[i] - h_b[i];
      TEST_ASSERT(fabsf(h_c[i] - expected_s) < 1e-4f * fmaxf(1.0f, fabsf(expected_s)),
                  "sub %s [%zu]: got %f, expected %f", cases[c].desc, i, h_c[i], expected_s);
    }
    free(h_c);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    free(h_a); free(h_b);
    g_tests_passed++;
  }
  TEST_REPORT();
}

// ── Test: power-of-two sizes ───────────────────────────────────────

static void test_power_of_two(cudaStream_t stream) {
  TEST_SECTION("Power-of-two sizes (1K..16M)");

  size_t sizes[] = {1024, 4096, 65536, 1048576, 4194304, 16777216};
  for (int s = 0; s < (int)(sizeof(sizes)/sizeof(sizes[0])); s++) {
    size_t N = sizes[s];
    float* h_a = (float*)malloc(N * sizeof(float));
    float* h_b = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
      h_a[i] = (float)((i * 1234567) % 1000000) / 1000.0f;
      h_b[i] = (float)((i * 7654321) % 1000000) / 1000.0f;
    }

    float* d_a = upload_to_gpu(h_a, N, stream);
    float* d_b = upload_to_gpu(h_b, N, stream);

    float* h_c = run_vec_add(d_a, d_b, N, stream);
    for (size_t i = 0; i < N; i += N/8) {
      TEST_ASSERT(fabsf(h_c[i] - (h_a[i] + h_b[i])) < 1e-4f,
                  "add N=%zu [%zu]: got %f, expected %f", N, i, h_c[i], h_a[i] + h_b[i]);
    }
    free(h_c);

    h_c = run_vec_sub(d_a, d_b, N, stream);
    for (size_t i = 0; i < N; i += N/8) {
      TEST_ASSERT(fabsf(h_c[i] - (h_a[i] - h_b[i])) < 1e-4f,
                  "sub N=%zu [%zu]: got %f, expected %f", N, i, h_c[i], h_a[i] - h_b[i]);
    }
    free(h_c);

    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b));
    free(h_a); free(h_b);
    g_tests_passed++;
  }
  TEST_REPORT();
}

// ── Test: non-power-of-two sizes ───────────────────────────────────

static void test_non_power_of_two(cudaStream_t stream) {
  TEST_SECTION("Non-power-of-two sizes (100..10M)");

  size_t sizes[] = {100, 500, 1000, 777, 12345, 99999, 10000000};
  for (int s = 0; s < (int)(sizeof(sizes)/sizeof(sizes[0])); s++) {
    size_t N = sizes[s];
    float* h_a = (float*)malloc(N * sizeof(float));
    float* h_b = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
      h_a[i] = (float)(i % 1000);
      h_b[i] = (float)((i * 3) % 1000);
    }

    float* d_a = upload_to_gpu(h_a, N, stream);
    float* d_b = upload_to_gpu(h_b, N, stream);

    float* h_c = run_vec_add(d_a, d_b, N, stream);
    for (size_t i = 0; i < N; i++) {
      TEST_ASSERT(fabsf(h_c[i] - (h_a[i] + h_b[i])) < 1e-5f,
                  "add N=%zu [%zu]: got %f, expected %f", N, i, h_c[i], h_a[i] + h_b[i]);
    }
    free(h_c);

    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b));
    free(h_a); free(h_b);
    g_tests_passed++;
  }
  TEST_REPORT();
}

// ── Test: fused vs separate (cross-validate) ───────────────────────

static void test_fused_vs_separate(cudaStream_t stream) {
  TEST_SECTION("Fused add_sub vs separate add + sub");

  size_t sizes[] = {1, 31, 256, 999, 65536, 1048576};
  for (int s = 0; s < (int)(sizeof(sizes)/sizeof(sizes[0])); s++) {
    size_t N = sizes[s];
    float* h_a = (float*)malloc(N * sizeof(float));
    float* h_b = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
      h_a[i] = (float)((i * 12345) % 1000);
      h_b[i] = (float)((i * 67890) % 1000);
    }

    float* d_a = upload_to_gpu(h_a, N, stream);
    float* d_b = upload_to_gpu(h_b, N, stream);

    float* h_add_sep = run_vec_add(d_a, d_b, N, stream);
    float* h_sub_sep = run_vec_sub(d_a, d_b, N, stream);

    float *h_add_fus, *h_sub_fus;
    run_vec_add_sub(d_a, d_b, N, stream, &h_add_fus, &h_sub_fus);

    for (size_t i = 0; i < N; i++) {
      TEST_ASSERT(fabsf(h_add_fus[i] - h_add_sep[i]) < 1e-6f,
                  "add mismatch N=%zu [%zu]: fused=%f separate=%f",
                  N, i, h_add_fus[i], h_add_sep[i]);
      TEST_ASSERT(fabsf(h_sub_fus[i] - h_sub_sep[i]) < 1e-6f,
                  "sub mismatch N=%zu [%zu]: fused=%f separate=%f",
                  N, i, h_sub_fus[i], h_sub_sep[i]);
    }

    free(h_a); free(h_b);
    free(h_add_sep); free(h_sub_sep);
    free(h_add_fus); free(h_sub_fus);
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b));
    g_tests_passed++;
  }
  TEST_REPORT();
}

// ── Test: random data — large-scale full validation ────────────────

static void test_random_large(cudaStream_t stream) {
  TEST_SECTION("Random data — full validation (1M elements)");

  const size_t N = 1UL << 20;  // 1,048,576
  float* h_a = (float*)malloc(N * sizeof(float));
  float* h_b = (float*)malloc(N * sizeof(float));

  unsigned seed = 42;
  for (size_t i = 0; i < N; i++) {
    seed = seed * 1103515245 + 12345;
    h_a[i] = (float)(seed % 1000000) / 1000.0f - 500.0f;
    seed = seed * 1103515245 + 67890;
    h_b[i] = (float)(seed % 1000000) / 1000.0f - 500.0f;
  }

  float* d_a = upload_to_gpu(h_a, N, stream);
  float* d_b = upload_to_gpu(h_b, N, stream);

  float* h_c = run_vec_add(d_a, d_b, N, stream);
  for (size_t i = 0; i < N; i++) {
    TEST_ASSERT(fabsf(h_c[i] - (h_a[i] + h_b[i])) < 1e-3f,
                "add [%zu]: got %f, expected %f", i, h_c[i], h_a[i] + h_b[i]);
  }
  free(h_c);

  h_c = run_vec_sub(d_a, d_b, N, stream);
  for (size_t i = 0; i < N; i++) {
    TEST_ASSERT(fabsf(h_c[i] - (h_a[i] - h_b[i])) < 1e-3f,
                "sub [%zu]: got %f, expected %f", i, h_c[i], h_a[i] - h_b[i]);
  }
  free(h_c);

  float *h_add, *h_sub;
  run_vec_add_sub(d_a, d_b, N, stream, &h_add, &h_sub);
  for (size_t i = 0; i < N; i++) {
    TEST_ASSERT(fabsf(h_add[i] - (h_a[i] + h_b[i])) < 1e-3f,
                "fused-add [%zu]: got %f, expected %f", i, h_add[i], h_a[i] + h_b[i]);
    TEST_ASSERT(fabsf(h_sub[i] - (h_a[i] - h_b[i])) < 1e-3f,
                "fused-sub [%zu]: got %f, expected %f", i, h_sub[i], h_a[i] - h_b[i]);
  }
  free(h_add); free(h_sub);

  CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b));
  free(h_a); free(h_b);
  g_tests_passed++;
  TEST_REPORT();
}

// ── Test: zero-size vector (boundary) ──────────────────────────────

static void test_zero_size(cudaStream_t stream) {
  TEST_SECTION("Zero-size vector (N=0)");

  const size_t N = 0;
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_c, 1));
  vec_add(d_a, d_b, d_c, N, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  vec_sub(d_a, d_b, d_c, N, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  vec_add_sub(d_a, d_b, d_c, d_c, N, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFree(d_c));

  g_tests_passed++;
  TEST_REPORT();
}

// ── Test: negation (subtract from zero) ───────────────────────────

static void test_negation(cudaStream_t stream) {
  TEST_SECTION("Negation (0 - val) and identity (val + 0)");

  size_t sizes[] = {1, 7, 256, 1000};
  for (int s = 0; s < (int)(sizeof(sizes)/sizeof(sizes[0])); s++) {
    size_t N = sizes[s];
    float* h_a = (float*)malloc(N * sizeof(float));
    float* h_zero = (float*)calloc(N, sizeof(float));
    for (size_t i = 0; i < N; i++) h_a[i] = (float)(i * 3 + 1);

    float* d_a = upload_to_gpu(h_a, N, stream);
    float* d_zero = upload_to_gpu(h_zero, N, stream);

    // 0 - a should give -a
    float* h_neg = run_vec_sub(d_zero, d_a, N, stream);
    for (size_t i = 0; i < N; i++) {
      TEST_ASSERT(fabsf(h_neg[i] - (-h_a[i])) < 1e-6f,
                  "neg N=%zu [%zu]: got %f, expected %f", N, i, h_neg[i], -h_a[i]);
    }
    free(h_neg);

    // a + 0 should give a
    float* h_id = run_vec_add(d_a, d_zero, N, stream);
    for (size_t i = 0; i < N; i++) {
      TEST_ASSERT(fabsf(h_id[i] - h_a[i]) < 1e-6f,
                  "id N=%zu [%zu]: got %f, expected %f", N, i, h_id[i], h_a[i]);
    }
    free(h_id);

    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_zero));
    free(h_a); free(h_zero);
    g_tests_passed++;
  }
  TEST_REPORT();
}

// ── Run all tests ──────────────────────────────────────────────────

static int run_all_tests() {
  printf("╔══════════════════════════════════════════════════╗\n");
  printf("║  CUDA Add/Sub — Comprehensive Test Suite        ║\n");
  printf("╚══════════════════════════════════════════════════╝\n");

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  int total_passed = 0, total_failed = 0;

#define RUN_TEST(test_fn)                                                     \
  do {                                                                        \
    int prev_failed = g_tests_failed;                                         \
    test_fn(stream);                                                          \
    int f = g_tests_failed - prev_failed;                                     \
    if (f == 0) total_passed++; else total_failed++;                          \
  } while (0)

  RUN_TEST(test_tiny_vectors);
  RUN_TEST(test_edge_values);
  RUN_TEST(test_power_of_two);
  RUN_TEST(test_non_power_of_two);
  RUN_TEST(test_fused_vs_separate);
  RUN_TEST(test_random_large);
  RUN_TEST(test_zero_size);
  RUN_TEST(test_negation);

  CUDA_CHECK(cudaStreamDestroy(stream));

  printf("\n════════════════════════════════════════════════════\n");
  if (total_failed == 0) {
    printf("  ✅ ALL TESTS PASSED  (%d test groups)\n", total_passed);
  } else {
    printf("  ❌ %d FAILED, %d passed\n", total_failed, total_passed);
  }
  printf("════════════════════════════════════════════════════\n\n");

  return total_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}

// ── Standalone benchmark ───────────────────────────────────────────

static int run_benchmark() {
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

  #ifdef QUICK_TEST
  const size_t N = 1UL << 20;          // 1M elements (fast smoke-test)
  const int warmup = 1;
  const int iters = 3;
  #else
  const size_t N = 1UL << 28;          // ~268M elements = 1 GB per vector
  const int warmup = 10;
  const int iters = 30;
  #endif
  const size_t bytes_per_vec = N * sizeof(float);

  printf("Vector size  : %zu elements (%.2f GB each)\n", N,
         bytes_per_vec / 1e9);
  printf("Warmup       : %d iterations\n", warmup);
  printf("Measure      : %d iterations\n\n", iters);

  float *d_a, *d_b, *d_add, *d_sub;
  CUDA_CHECK(cudaMalloc(&d_a, bytes_per_vec));
  CUDA_CHECK(cudaMalloc(&d_b, bytes_per_vec));
  CUDA_CHECK(cudaMalloc(&d_add, bytes_per_vec));
  CUDA_CHECK(cudaMalloc(&d_sub, bytes_per_vec));

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

  free(h_a); free(h_b); free(h_add); free(h_sub);
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_add)); CUDA_CHECK(cudaFree(d_sub));

  return errors ? EXIT_FAILURE : EXIT_SUCCESS;
}

// ── Quick smoke test (finishes in < 5 seconds) ─────────────────────

static int run_benchmark_quick(cudaStream_t stream) {
  const size_t N = 1UL << 20;  // 1M elements (4 MB)
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

  float* h_a = (float*)malloc(N * sizeof(float));
  float* h_b = (float*)malloc(N * sizeof(float));
  for (size_t i = 0; i < N; i++) { h_a[i] = (float)i; h_b[i] = (float)(i * 2); }
  CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

  vec_add(d_a, d_b, d_c, N, stream);
  vec_sub(d_a, d_b, d_c, N, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float* h_c = (float*)malloc(N * sizeof(float));
  CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

  int errors = 0;
  for (size_t i = 0; i < N && errors < 5; i++) {
    float expected = h_a[i] - h_b[i];
    if (fabsf(h_c[i] - expected) > 1e-5f) errors++;
  }

  free(h_a); free(h_b); free(h_c);
  CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_c));

  if (errors == 0)
    printf("Kernel addsub.cu --quick: OK (vec_sub verified on 1M elements)\n");
  else
    printf("Kernel addsub.cu --quick: %d ERRORS\n", errors);
  return errors;
}

// ── Main entry point ───────────────────────────────────────────────

int main(int argc, char** argv) {
  bool run_tests_flag = false;
  bool run_bench_flag = true;  // default: bench only
  bool quick_mode = false;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--test") == 0) {
      run_tests_flag = true;
      run_bench_flag = false;
    } else if (strcmp(argv[i], "--all") == 0) {
      run_tests_flag = true;
      run_bench_flag = true;
    } else if (strcmp(argv[i], "--bench") == 0) {
      run_tests_flag = false;
      run_bench_flag = true;
    } else if (strcmp(argv[i], "--quick") == 0) {
      quick_mode = true;
    }
  }

  if (quick_mode) {
    // Just compile & run a tiny smoke test for validation
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    run_benchmark_quick(stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
  }

  if (run_tests_flag) {
    int test_result = run_all_tests();
    if (test_result != EXIT_SUCCESS) return test_result;
  }

  if (run_bench_flag) {
    int bench_result = run_benchmark();
    if (bench_result != EXIT_SUCCESS) return bench_result;
  }

  return EXIT_SUCCESS;
}

#endif  // BUILD_AS_SHARED
