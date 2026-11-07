"""In-process paired A/B: update TC32 (baseline vs kSub=2 batch sub-tiling).

Parity: both variants must produce bit-identical W and counter changes
(same inputs, both reference-correct). Times each on fc1/fc2/head shapes,
interleaved A/B to cancel drift.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CUH = os.path.join(REPO, "kernels/packed_ternary/packed_ternary.cuh")
OLD_CU = os.path.join(HERE, "old_gemm_update_tc_v2_32.cu")
NEW_CU = os.path.join(HERE, "update_syncwarp.cu")

import shutil
if not os.path.exists(OLD_CU):
    shutil.copy(os.path.join(REPO, "kernels/packed_ternary/gemm_update_tc_v2_32.cu"), OLD_CU)

with open(CUH) as f:
    CUH_SRC = f.read()

CPP = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
extern "C" {
    void launch_packed_ternary_update_tc_v2(
        const void* X_ptr, const void* dY_ptr, uint32_t* W, int16_t* counter,
        int batch_size, int in_features, int out_features, int stride_words,
        int16_t threshold, cudaStream_t stream);
}
void upd(torch::Tensor W, torch::Tensor counter, torch::Tensor X, torch::Tensor dY, int64_t threshold) {
    launch_packed_ternary_update_tc_v2(
        X.data_ptr(), dY.data_ptr(),
        reinterpret_cast<uint32_t*>(W.data_ptr<int32_t>()),
        reinterpret_cast<int16_t*>(counter.data_ptr()),
        X.size(0), X.size(1), W.size(0), W.size(1), (int16_t)threshold, nullptr);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("upd", &upd, "update tc v2");
}
"""

def build(cu_path, name):
    from torch.utils.cpp_extension import load_inline
    with open(cu_path) as f:
        cu = f.read()
    combined = CUH_SRC + "\n" + cu.replace('#include "packed_ternary.cuh"', "")
    return load_inline(name=name, cpp_sources=CPP, cuda_sources=[combined],
                       verbose=False, extra_cuda_cflags=["-O2"])

def main():
    dev = "cuda"
    print("building OLD update...", flush=True)
    old = build(OLD_CU, "upd_probe_old")
    print("building syncwarp update...", flush=True)
    new = build(NEW_CU, "upd_probe_sw")

    torch.manual_seed(0)
    BATCH = 16384
    shapes = [("fc1", 1024, 4096), ("fc2", 4096, 1024), ("head", 1024, 50272)]
    print(f"{'name':<6} {'in':>5} {'out':>6}   {'old_ms':>8} {'new_ms':>8} {'d%':>7}   parity", flush=True)
    for name, inn, out in shapes:
        X = torch.randn(BATCH, inn, device=dev, dtype=torch.float16)
        dY = torch.randn(BATCH, out, device=dev, dtype=torch.float16)
        def fresh():
            W = torch.randint(0, 4, (out, (inn + 15) // 16), device=dev, dtype=torch.int32)
            W = torch.where(W == 3, 0, W)
            counter = torch.randint(-40, 41, (out * inn,), device=dev, dtype=torch.int16)
            return W, counter
        def timed(fn, iters=6, warmup=2):
            for _ in range(warmup): fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters): fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1000
        # parity: run both on identical (W,counter) clones, compare results
        W0, C0 = fresh()
        W1, C1 = W0.clone(), C0.clone()
        W2, C2 = W0.clone(), C0.clone()
        old.upd(W1, C1, X.contiguous(), dY.contiguous(), 32)
        new.upd(W2, C2, X.contiguous(), dY.contiguous(), 32)
        same_w = torch.equal(W1, W2)
        same_c = torch.equal(C1, C2)
        # timing
        to, tn = [], []
        for _ in range(3):
            Wa, Ca = fresh()
            to.append(timed(lambda: old.upd(Wa, Ca, X.contiguous(), dY.contiguous(), 32)))
            Wb, Cb = fresh()
            tn.append(timed(lambda: new.upd(Wb, Cb, X.contiguous(), dY.contiguous(), 32)))
        mo, mn = min(to), min(tn)
        d = 100.0 * (mn - mo) / mo
        print(f"{name:<6} {inn:>5} {out:>6}   {mo:8.2f} {mn:8.2f} {d:7.2f}   W{'OK' if same_w else 'DIFF'} C{'OK' if same_c else 'DIFF'}", flush=True)

if __name__ == "__main__":
    main()