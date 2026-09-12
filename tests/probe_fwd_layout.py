"""In-process paired A/B: forward TC64 W_smem layout (old transposed vs new row-major).

Compiles BOTH kernels from source in one process (old = committed HEAD version via
git show, new = current working tree), checks both against an FP32 reference, then
times both in interleaved A/B/A/B order to cancel Modal T4 drift. No production
dispatch touched.

Usage: python tests/probe_fwd_layout.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CUH = os.path.join(REPO, "kernels/packed_ternary/packed_ternary.cuh")
NEW_CU = os.path.join(REPO, "kernels/packed_ternary/gemm_forward_tc.cu")

# Old (baseline) source: committed HEAD version of the forward kernel, vendored
# into tests/old_gemm_forward_tc.cu (Modal container has no .git).
OLD_CU = os.path.join(HERE, "old_gemm_forward_tc.cu")

with open(CUH) as f:
    CUH_SRC = f.read()


def build(cu_path, name):
    from torch.utils.cpp_extension import load_inline
    with open(cu_path) as f:
        cu = f.read()
    combined = CUH_SRC + "\n" + cu.replace('#include "packed_ternary.cuh"', "")
    lib = load_inline(
        name=name,
        cpp_sources=r"""
        #include <cuda_runtime.h>
        #include <torch/extension.h>
        extern "C" {
            void launch_packed_ternary_forward_tc_64(
                const uint32_t* W, const void* X, void* Y,
                int batch_size, int in_features, int out_features,
                int stride_words, cudaStream_t stream);
        }
        torch::Tensor fwd(torch::Tensor W, torch::Tensor X) {
            auto Y = torch::empty({X.size(0), W.size(0)}, torch::dtype(torch::kFloat16).device(X.device()));
            launch_packed_ternary_forward_tc_64(
                reinterpret_cast<const uint32_t*>(W.data_ptr<int32_t>()),
                X.data_ptr<at::Half>(), Y.data_ptr<at::Half>(),
                X.size(0), X.size(1), W.size(0), W.size(1), nullptr);
            return Y;
        }
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fwd", &fwd); }
        """,
        cuda_sources=[combined],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
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
    print("building OLD (transposed) ...", flush=True)
    old = build(OLD_CU, "fwd_probe_old")
    print("built OLD ok", flush=True)
    print("building NEW (row-major) ...", flush=True)
    new = build(NEW_CU, "fwd_probe_new")
    print("built NEW ok", flush=True)

    torch.manual_seed(0)
    BATCH = 16384
    shapes = [("fc1", 1024, 4096), ("fc2", 4096, 1024), ("head", 1024, 50272)]

    print(f"{'name':<6} {'in':>5} {'out':>6}   {'old_ms':>8} {'new_ms':>8} {'d%':>7}   {'old_err':>8} {'new_err':>8}", flush=True)

    for name, inn, out in shapes:
        X = torch.randn(BATCH, inn, device=dev, dtype=torch.float16)
        W = torch.randint(0, 4, (out, (inn + 15) // 16), device=dev, dtype=torch.int32)
        W = torch.where(W == 3, 0, W)
        Wf = unpack_ref(W, inn)
        ref = (X.half().float() @ Wf.float().t()).half()

        def timed(fn, iters=8, warmup=2):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1000

        yo = old.fwd(W, X)
        yn = new.fwd(W, X)
        eo = (yo.float() - ref.float()).abs().max().item()
        en = (yn.float() - ref.float()).abs().max().item()

        to, tn = [], []
        for _ in range(3):
            to.append(timed(lambda: old.fwd(W, X)))
            tn.append(timed(lambda: new.fwd(W, X)))
        mo, mn = min(to), min(tn)
        d = 100.0 * (mn - mo) / mo
        print(f"{name:<6} {inn:>5} {out:>6}   {mo:8.2f} {mn:8.2f} {d:7.2f}   {eo:.3e} {en:.3e}", flush=True)


if __name__ == "__main__":
    main()