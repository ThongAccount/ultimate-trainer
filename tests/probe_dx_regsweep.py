"""Register sweep on dX 64x64 K32 kernel via -maxrregcount.

Same kernel, 4 compile variants (default/96, 88, 80, 72). Checks:
- parity (all variants bit-exact vs default output)
- regs + spills (from ptxas -v in build log)
- interleaved timing to cancel drift

Usage: python tests/probe_dx_regsweep.py   (from Modal runner)
"""
import os, sys, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CUH = os.path.join(REPO, "kernels/packed_ternary/packed_ternary.cuh")
CU = os.path.join(REPO, "kernels/packed_ternary/gemm_backward_dx_tc.cu")

with open(CUH) as f:
    CUH_SRC = f.read()
with open(CU) as f:
    CU_SRC = f.read()
combined = CUH_SRC + "\n" + CU_SRC.replace('#include "packed_ternary.cuh"', "")

CPP = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
extern "C" {
    void launch_packed_ternary_backward_dx_tc_64(
        const uint32_t* W, const void* dY, void* dX,
        int batch_size, int in_features, int out_features, int stride_words,
        cudaStream_t stream);
}
torch::Tensor dx(torch::Tensor W, torch::Tensor dY, int64_t K) {
    auto dX = torch::empty({dY.size(0), K}, torch::dtype(torch::kFloat16).device(dY.device()));
    launch_packed_ternary_backward_dx_tc_64(
        reinterpret_cast<const uint32_t*>(W.data_ptr<int32_t>()),
        dY.data_ptr<at::Half>(), dX.data_ptr<at::Half>(),
        dY.size(0), K, W.size(0), W.size(1), nullptr);
    return dX;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("dx", &dx); }
"""

def build(maxreg, name):
    from torch.utils.cpp_extension import load_inline
    cflags = ["-O2", "-Xptxas=-v"]
    if maxreg:
        cflags.append(f"-maxrregcount={maxreg}")
    lib = load_inline(name=name, cpp_sources=CPP, cuda_sources=[combined],
                      verbose=False, extra_cuda_cflags=cflags)
    return lib


def unpack_ref(W, K):
    N, stride = W.shape
    words = W.long().unsqueeze(-1)
    shifts = torch.arange(16, device=W.device).view(1, 1, 16) * 2
    codes = (words >> shifts) & 3
    vals = torch.where(codes == 1, 1.0, torch.where(codes == 2, -1.0, 0.0))
    return vals.reshape(N, stride * 16)[:, :K].to(torch.float16)


def main():
    dev = "cuda"
    torch.manual_seed(0)
    BATCH = 16384
    VARIANTS = [("default", None), ("r88", 88), ("r80", 80), ("r72", 72)]
    libs = {}
    print("building variants...", flush=True)
    for label, reg in VARIANTS:
        print(f"  {label} (maxrregcount={reg})...", flush=True)
        try:
            libs[label] = build(reg, "dx_reg_" + label.replace("default", "def"))
        except Exception as e:
            print(f"  {label} BUILD FAILED: {str(e)[-300:]}", flush=True)
    print("built:", list(libs), flush=True)

    # parity on fc1 shape (fast): all vs default
    K, N = 1024, 4096
    dY = torch.randn(BATCH, N, device=dev, dtype=torch.float16)
    W = torch.randint(0, 4, (N, (K + 15) // 16), device=dev, dtype=torch.int32)
    W = torch.where(W == 3, 0, W)
    base = libs["default"].dx(W, dY, K)
    parity = {}
    for label, lib in libs.items():
        if label == "default":
            continue
        out = lib.dx(W, dY, K)
        same = torch.equal(base, out)
        maxdiff = (base.float() - out.float()).abs().max().item()
        parity[label] = (same, maxdiff)
        print(f"parity {label}: {'EXACT' if same else f'DIFF maxdiff={maxdiff:.2e}'}", flush=True)

    def timed(fn, iters=8, warmup=2):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    print(f"{'shape':<6} ", end="", flush=True)
    for label in libs:
        print(f"{label:>10}", end="", flush=True)
    print("", flush=True)
    for name, inn, out in [("fc1", 1024, 4096), ("head", 1024, 50272)]:
        dY2 = torch.randn(BATCH, out, device=dev, dtype=torch.float16)
        W2 = torch.randint(0, 4, (out, (inn + 15) // 16), device=dev, dtype=torch.int32)
        W2 = torch.where(W2 == 3, 0, W2)
        times = {label: [] for label in libs}
        for _ in range(3):
            for label, lib in libs.items():
                times[label].append(timed(lambda: lib.dx(W2, dY2, inn)))
        base_t = min(times["default"])
        print(f"{name:<6} ", end="", flush=True)
        for label in libs:
            m = min(times[label])
            print(f"{m:8.2f}({100.0*(m-base_t)/base_t:+5.1f}%)", end="", flush=True)
        print("", flush=True)

if __name__ == "__main__":
    main()