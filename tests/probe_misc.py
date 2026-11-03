"""Probe batch 2 misc items — no kernel changes, measurements + parity only:

  #12  layer-0 clone cost: X.detach().clone() at the shipped layer-0 shape
       [16352, 1024] fp16 (and [16352, 4096] for scale), solo_median ms and
       as % of the 593.98 ms shipped step (results from measure_p0).
  #15  small-dims fallback parity: production TC64 fwd kernel JIT-compiled
       here, run at small/edge dims (64^3, 48^3, 16x1024x4096, odd 50) vs an
       fp32 reference; reports max rel err per shape. Catches silent wrong
       results on the fallback path, not speed.
  #6   fused-CE feasibility probe: eager F.cross_entropy vs torch.compile'd
       CE vs (logits*scale -> CE) at the shipped logits shape
       [16352, 50272] fp16. solo_median ms. Informs whether a fused CE+scale
       kernel is worth a branch (it breaks the bit-stable gate, so it needs
       a loss-trajectory gate + re-baseline first).

Usage: python tests/probe_misc.py   (from the Colab VM)
"""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CUH = os.path.join(REPO, "kernels/packed_ternary/packed_ternary.cuh")
PROD_CU = os.path.join(REPO, "kernels/packed_ternary/gemm_forward_tc.cu")

with open(CUH) as f:
    CUH_SRC = f.read()

CPP = r"""
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
"""


def unpack_ref(W, K):
    N, stride = W.shape
    words = W.long().unsqueeze(-1)
    shifts = torch.arange(16, device=W.device).view(1, 1, 16) * 2
    codes = (words >> shifts) & 3
    vals = torch.where(codes == 1, 1.0, torch.where(codes == 2, -1.0, 0.0))
    return vals.reshape(N, stride * 16)[:, :K]


def make_W(N, K, dev):
    W = torch.randint(0, 4, (N, (K + 15) // 16), device=dev, dtype=torch.int32)
    return torch.where(W == 3, 0, W)


def probe_clone():
    from bench_protocol import solo_median
    dev = "cuda"
    print("== #12 clone cost ==", flush=True)
    STEP_MS = 593.98  # shipped arm, measure_p0 @ ab4dbeb0
    for shape in [(16352, 1024), (16352, 4096)]:
        x = torch.randn(*shape, device=dev, dtype=torch.float16)
        res = solo_median(lambda: x.detach().clone(), null_arm=True)
        print(f"clone {shape}: median={res.a_ms:.4f} ms "
              f"({100.0 * res.a_ms / STEP_MS:.3f}% of {STEP_MS} ms step, "
              f"sigma_null={res.sigma_null_pct:.2f}%)", flush=True)


def probe_small_dims():
    from torch.utils.cpp_extension import load_inline
    dev = "cuda"
    print("== #15 small-dims fwd parity (TC64 kernel vs fp32 ref) ==", flush=True)
    with open(PROD_CU) as f:
        cu = f.read()
    combined = CUH_SRC + "\n" + cu.replace('#include "packed_ternary.cuh"', "")
    lib = load_inline(name="probe_misc_fwd_ext", cpp_sources=CPP,
                      cuda_sources=[combined], verbose=False,
                      extra_cuda_cflags=["-O3", "--use_fast_math"])
    torch.manual_seed(1)
    ok = True
    for B, K, N in [(64, 64, 64), (48, 48, 48), (16, 1024, 4096), (80, 96, 112), (16352, 1024, 4096)]:
        X = torch.randn(B, K, device=dev, dtype=torch.float16)
        W = make_W(N, K, dev)
        Y = lib.fwd(W, X)
        ref = (X.float() @ unpack_ref(W, K).float().t())
        denom = ref.abs().clamp_min(1e-3)
        rel = ((Y.float() - ref).abs() / denom).max().item()
        good = rel < 5e-3
        ok = ok and good
        print(f"shape B={B} K={K} N={N}: max rel err {rel:.2e} "
              f"{'OK' if good else 'FAIL'}", flush=True)
    print("SMALL_DIMS:", "PASS" if ok else "FAIL", flush=True)


def probe_fused_ce():
    from bench_protocol import solo_median
    dev = "cuda"
    print("== #6 fused CE probe ==", flush=True)
    B, V = 16352, 50272
    logits = torch.randn(B, V, device=dev, dtype=torch.float16)
    tgt = torch.randint(0, V, (B,), device=dev)
    scale = 2.0 ** -5

    r_eager = solo_median(lambda: F.cross_entropy(logits, tgt), null_arm=True)
    print(f"CE eager:            median={r_eager.a_ms:8.2f} ms (sigma_null={r_eager.sigma_null_pct:.2f}%)", flush=True)

    r_scaled = solo_median(lambda: F.cross_entropy(logits * scale, tgt))
    print(f"CE (logits*scale):   median={r_scaled.a_ms:8.2f} ms", flush=True)

    try:
        ce_c = torch.compile(lambda l, t: F.cross_entropy(l, t), mode="reduce-overhead")
        ce_c(logits, tgt)  # warm compile
        torch.cuda.synchronize()
        r_comp = solo_median(lambda: ce_c(logits, tgt))
        print(f"CE torch.compile:    median={r_comp.a_ms:8.2f} ms", flush=True)
    except Exception as e:
        print(f"CE torch.compile FAILED: {str(e)[-200:]}", flush=True)
    torch.cuda.empty_cache()


def main():
    if not torch.cuda.is_available():
        print("no cuda", flush=True)
        return 1
    probe_clone()
    probe_small_dims()
    probe_fused_ce()
    print("PROBE_MISC: DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
