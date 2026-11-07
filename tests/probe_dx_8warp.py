"""In-process paired A/B: dX 64x64 current (4-warp K32) vs 8-warp variant.

Builds OLD (committed HEAD dX) and NEW (tests/dx_8warp.cu) from source, checks
bit-exact parity of dX output, times interleaved A/B to cancel drift.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CUH = os.path.join(REPO, "kernels/packed_ternary/packed_ternary.cuh")
NEW_CU = os.path.join(HERE, "dx_8warp.cu")

# OLD = committed HEAD dX kernel, vendored (Modal container has no .git)
OLD_CU = os.path.join(HERE, "old_gemm_backward_dx_tc.cu")
import shutil
if not os.path.exists(OLD_CU):
    shutil.copy(os.path.join(REPO, "kernels/packed_ternary/gemm_backward_dx_tc.cu"), OLD_CU)

with open(CUH) as f:
    CUH_SRC = f.read()

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

def build(cu_path, name):
    from torch.utils.cpp_extension import load_inline
    with open(cu_path) as f:
        cu = f.read()
    combined = CUH_SRC + "\n" + cu.replace('#include "packed_ternary.cuh"', "")
    return load_inline(name=name, cpp_sources=CPP, cuda_sources=[combined],
                       verbose=False, extra_cuda_cflags=["-O2"])

def unpack_ref(W, K):
    N, stride = W.shape
    words = W.long().unsqueeze(-1)
    shifts = torch.arange(16, device=W.device).view(1, 1, 16) * 2
    codes = (words >> shifts) & 3
    vals = torch.where(codes == 1, 1.0, torch.where(codes == 2, -1.0, 0.0))
    return vals.reshape(N, stride * 16)[:, :K].to(torch.float16)

def main():
    dev = "cuda"
    print("building OLD dX...", flush=True)
    old = build(OLD_CU, "dx_probe_old")
    print("building NEW dX (8-warp)...", flush=True)
    new = build(NEW_CU, "dx_probe_8w")
    print("built both", flush=True)
    torch.manual_seed(0)
    BATCH = 16384
    shapes = [("fc1", 1024, 4096), ("fc2", 4096, 1024), ("head", 1024, 50272)]
    print(f"{'name':<6} {'in':>5} {'out':>6}   {'old_ms':>8} {'new_ms':>8} {'d%':>7}   parity", flush=True)
    for name, inn, out in shapes:
        dY = torch.randn(BATCH, out, device=dev, dtype=torch.float16)
        W = torch.randint(0, 4, (out, (inn + 15) // 16), device=dev, dtype=torch.int32)
        W = torch.where(W == 3, 0, W)
        Wf = unpack_ref(W, inn)
        ref = (dY.half().float() @ Wf.float()).half()  # dX = dY @ W

        def timed(fn, iters=8, warmup=2):
            for _ in range(warmup): fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters): fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1000

        yo = old.dx(W, dY, inn)
        yn = new.dx(W, dY, inn)
        eo = (yo.float() - ref.float()).abs().max().item()
        en = (yn.float() - ref.float()).abs().max().item()
        exact = torch.equal(yo, yn)
        to, tn = [], []
        for _ in range(3):
            to.append(timed(lambda: old.dx(W, dY, inn)))
            tn.append(timed(lambda: new.dx(W, dY, inn)))
        mo, mn = min(to), min(tn)
        d = 100.0 * (mn - mo) / mo
        print(f"{name:<6} {inn:>5} {out:>6}   {mo:8.2f} {mn:8.2f} {d:7.2f}   {'EXACT' if exact else 'DIFF'} err={eo:.1e}/{en:.1e}", flush=True)

if __name__ == "__main__":
    main()