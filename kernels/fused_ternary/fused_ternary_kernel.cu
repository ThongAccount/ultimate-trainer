// fused_ternary_kernel.cu
// FP16 quantization kernel for BitNet b1.58 ternary weights.
// Reads FP32 master weights, writes FP16 gamma-scaled ternary {-gamma, 0, +gamma}.
// Designed to pair with torch.matmul FP16 (cuBLAS TensorCore HMMA path).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Quantize FP32 master weight to FP16 ternary {-gamma, 0, +gamma}.
// Each thread handles one element.
__global__ void quantize_ternary_fp16_kernel(
    const float* __restrict__ w_in,   // (N, K) FP32 master weights
    half* __restrict__ w_out,         // (N, K) FP16 ternary weights
    float gamma,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float w = w_in[idx];
    float w_q = rintf(w / gamma);  // round to nearest integer
    half w_ternary;
    if (w_q > 0.5f) {
        w_ternary = __float2half(gamma);
    } else if (w_q < -0.5f) {
        w_ternary = __float2half(-gamma);
    } else {
        w_ternary = __float2half(0.0f);
    }
    w_out[idx] = w_ternary;
}

// Host launch wrapper
extern "C" void launch_quantize_ternary_fp16(
    const float* w_in, half* w_out, float gamma, int num_elements) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    quantize_ternary_fp16_kernel<<<blocks, threads>>>(w_in, w_out, gamma, num_elements);
}
