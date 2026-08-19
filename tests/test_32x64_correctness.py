"""Test 32×64 backward_dx kernel correctness against torch reference."""
import os
import sys
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.path.insert(0, os.getcwd())

import torch
from torch.utils.cpp_extension import load_inline

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
CUH_PATH = os.path.join(PARENT, "kernels/packed_ternary/packed_ternary.cuh")
CU_32X64_PATH = os.path.join(PARENT, "kernels/packed_ternary/gemm_backward_dx_32x64.cu")

# Load kernel
print("Loading 32×64 kernel...")
with open(CUH_PATH) as f:
    cuh = f.read()
with open(CU_32X64_PATH) as f:
    cu = f.read()
combined = cuh + "\n" + cu.replace('#include "packed_ternary.cuh"', "")

lib = load_inline(
    name="packed_ternary_dx_32x64_test",
    cpp_sources=r"""
    #include <cuda_runtime.h>
    #include <torch/extension.h>
    extern "C" {
        void launch_packed_ternary_backward_dx_32x64(
            const uint32_t* W, const void* dY, void* dX,
            int B, int K, int N, int stride, cudaStream_t s);
    }
    torch::Tensor dx_32x64_wrapper(torch::Tensor W, torch::Tensor dY, int K) {
        int B = dY.size(0);
        int N = dY.size(1);
        auto dX = torch::empty({B, K}, torch::dtype(torch::kFloat16).device(dY.device()));
        launch_packed_ternary_backward_dx_32x64(
            reinterpret_cast<const uint32_t*>(W.data_ptr<int32_t>()),
            dY.data_ptr<at::Half>(), dX.data_ptr<at::Half>(),
            B, K, N, W.size(1), nullptr);
        return dX;
    }
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("backward_dx_32x64", &dx_32x64_wrapper, "dX = W^T @ dY (32x64)");
    }
    """,
    cuda_sources=[combined],
    verbose=False,
    extra_cuda_cflags=["-O2", "-arch=sm_75"],
)

print("Kernel loaded successfully!")

# Setup test case
torch.manual_seed(0)
B, K, N = 128, 1024, 1024  # Production dimensions
kWeightsPerWord = 16

# Create packed weight tensor
stride_words = (K + kWeightsPerWord - 1) // kWeightsPerWord
W_packed = torch.randint(0, 4, (N, stride_words), dtype=torch.int32, device="cuda")

# Decode to ternary for reference
W_ter = torch.zeros(N, K, device="cuda", dtype=torch.float32)
for wi in range(stride_words):
    word = W_packed[:, wi]
    for b in range(kWeightsPerWord):
        code = (word >> (2 * b)) & 3
        val = torch.where(code == 1, 1.0, torch.where(code == 2, -1.0, 0.0))
        k = wi * kWeightsPerWord + b
        if k < K:
            W_ter[:, k] = val

# Input gradient
dY = torch.randn(B, N, device="cuda", dtype=torch.float16)

# Reference: dX = dY @ W (B×N @ N×K = B×K)
ref = (dY.float() @ W_ter).half()

# Test 32×64 kernel
dX_32x64 = lib.backward_dx_32x64(W_packed, dY, K)

# Compare
err_abs = (dX_32x64 - ref).abs().max().item()
err_rel = err_abs / ref.abs().max().item()
ref_mag = ref.abs().max().item()

print(f"\n{'='*60}")
print(f"32×64 BACKWARD_DX CORRECTNESS TEST")
print(f"{'='*60}")
print(f"Dimensions: B={B}, K={K}, N={N}")
print(f"Reference magnitude: {ref_mag:.6f}")
print(f"Absolute error:      {err_abs:.6f}")
print(f"Relative error:      {err_rel:.6f} ({err_rel*100:.3f}%)")
print(f"{'='*60}")

# Gate: error < 0.05
if err_abs < 0.05:
    print("✅ PASS — Correctness validated!")
    sys.exit(0)
else:
    print(f"❌ FAIL — Error {err_abs:.6f} exceeds threshold 0.05")
    sys.exit(1)
