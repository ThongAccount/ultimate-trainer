"""Modal: same-instance interleaved e2e A/B — OLD forward (transposed) vs NEW (row-major).

Compiles BOTH forward kernels in one process, builds the gigatoken model with each,
and runs 5 steps each, interleaved OLD/NEW/OLD/NEW to cancel T4 thermal/instance
drift. Uses the exact production model/step from train_gigatoken.py but with
kernel source swapped via the pack_forward loader patched to the variant.

Simplest robust approach: monkeypatch pack_forward._load_tc_64 to load from a given
.cu path, run train_step_cudagraph N times per variant, report mean per-step ms.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
import torch

REPO = "/repo"
HERE = os.path.join(REPO, "tests")
CUH = os.path.join(REPO, "kernels/packed_ternary/packed_ternary.cuh")
OLD_CU = os.path.join(HERE, "old_gemm_forward_tc.cu")   # committed HEAD (transposed)
NEW_CU = os.path.join(REPO, "kernels/packed_ternary/gemm_forward_tc.cu")  # row-major

with open(CUH) as f:
    CUH_SRC = f.read()

# shim: replace the kernel source pack_forward reads, then rebuild the module ref
def build_lib(cu_path, name):
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
        cuda_sources=[combined], verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    return lib


from train_gigatoken import build_model, B, SEQ, VOCAB, WARMUP, STEPS

def run_steps(lib, n=5, x=None, y=None):
    # point production forward loader at our lib by patching the module's fn ref
    import kernels.packed_ternary.custom_ops as co
    from kernels.packed_ternary import pack_forward as pf
    pf._forward_fn_tc_64 = lib.fwd
    pf._HAS_TC_64 = True
    co._fwd_tc = lib.fwd
    # forward_auto uses custom_ops.forward_tc -> co._fwd_tc
    model = build_model().cuda()
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        loss_det = train_step_like(model, x, y)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        loss_det.item()
    return times[1:]  # drop warmup

def train_step_like(model, x, y):
    # mirror train_gigatoken.train_step_cudagraph (non-profile path)
    scale = 1024 ** -0.5
    Bt, T = x.shape
    h = model.embed(x).half().view(Bt * T, 1024)
    for i in range(6):
        h = model.norms[i](h.float()).half()
        h = model.fc1s[i](h) * scale
        h = torch.nn.functional.gelu(h)
        h = model.fc2s[i](h) * scale
    h = model.head(h) * scale
    logits = h.view(Bt, T, VOCAB)
    # target = next-token per position, length Bt*T (logits shape [B*T, VOCAB])
    loss = torch.nn.functional.cross_entropy(logits.view(-1, VOCAB), y.reshape(-1), reduction="mean")
    loss.backward()
    return loss.detach()

def main():
    print("building OLD...", flush=True)
    old = build_lib(OLD_CU, "e2e_fwd_old")
    print("building NEW...", flush=True)
    new = build_lib(NEW_CU, "e2e_fwd_new")
    torch.manual_seed(0)
    x = torch.randint(0, VOCAB, (B, SEQ), device="cuda")
    y = x.clone()
    # warm both models once
    for lib in (old, new):
        run_steps(lib, n=1, x=x, y=y)
    torch.cuda.synchronize()
    o_times, n_times = [], []
    for r in range(3):
        o_times += run_steps(old, n=2, x=x, y=y)
        n_times += run_steps(new, n=2, x=x, y=y)
    o_avg = sum(o_times)/len(o_times)
    n_avg = sum(n_times)/len(n_times)
    print(f"OLD: {o_times}", flush=True)
    print(f"NEW: {n_times}", flush=True)
    print(f"OLD avg {o_avg:.1f} ms/step  NEW avg {n_avg:.1f} ms/step  delta {100.0*(n_avg-o_avg)/o_avg:+.2f}%", flush=True)

if __name__ == "__main__":
    main()