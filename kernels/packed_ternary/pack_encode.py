"""Loader for the Phase 2 counter-word update kernel (no atomicCAS).

Pattern mirrors pack_update.py's TC loaders: cpp_sources carries the extern
declaration + PYBIND11 module, cuda_sources the kernel + launcher.
"""
import os

_CW_MOD = None
_HAS_CW = False

HERE = os.path.dirname(os.path.abspath(__file__))
_CW_CU = os.path.join(HERE, "update_counterword.cu")

_CPP = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

extern "C" {
    void launch_packed_ternary_update_counterword(
        const void* X, const void* dY, uint32_t* W, int16_t* counter,
        int B, int K, int N, int stride, int16_t threshold, cudaStream_t s);
}

void update_counterword_wrapper(
    torch::Tensor W, torch::Tensor counter,
    torch::Tensor X, torch::Tensor dY, int16_t threshold)
{
    launch_packed_ternary_update_counterword(
        X.data_ptr<at::Half>(), dY.data_ptr<at::Half>(),
        reinterpret_cast<uint32_t*>(W.data_ptr<int32_t>()),
        counter.data_ptr<int16_t>(),
        X.size(0), X.size(1), dY.size(1), W.size(1),
        threshold, c10::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("update_counterword", &update_counterword_wrapper,
          "dW->sign->counter->flip, one thread per W word (no atomics)");
}
"""


def counterword_module():
    global _CW_MOD, _HAS_CW
    if _HAS_CW:
        return _CW_MOD
    from torch.utils.cpp_extension import load_inline

    with open(_CW_CU) as f:
        cu = f.read()
    with open(os.path.join(HERE, "packed_ternary.cuh")) as f:
        cuh = f.read()
    combined = cuh + "\n" + cu.replace('#include "packed_ternary.cuh"', "")

    _CW_MOD = load_inline(
        name="pt_update_counterword",
        cpp_sources=_CPP,
        cuda_sources=[combined],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )
    _HAS_CW = True
    return _CW_MOD
