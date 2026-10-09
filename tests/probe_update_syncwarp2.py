"""In-process paired A/B: update TC32 batch-loop CTA barriers (prod vs warp-scope).

C12 hypothesis: the batch loop in gemm_update_tc_v2_32.cu calls __syncthreads()
twice per trip, but dY_smem and X_smem are warp-private slices (DYS/XS are
indexed by warp_id, and both load loops write only warp_id's slice). On T4 the
head grid is 1571 CTAs x 1024 batch trips = 103M CTA-barrier pairs per step.
Replacing the hot-path barrier with __syncwarp() removes CTA-wide convoy
without touching smem or registers.

This is NOT the previously falsified C4 (which swapped one barrier for __syncwarp
without checking whether the tiles were private). Here the whole hot path is
warp-private, so both barriers go.

Parity gate: bit-exact W and counter changes vs the production kernel. A wrong
answer here silently corrupts training, so the gate is mandatory, not advisory.

Usage: python tests/probe_update_syncwarp2.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CUH = os.path.join(REPO, "kernels/packed_ternary/packed_ternary.cuh")
OLD_CU = os.path.join(REPO, "kernels/packed_ternary/gemm_update_tc_v2_32.cu")
NEW_CU = os.path.join(HERE, "update_syncwarp2.cu")

with open(CUH) as f:
    CUH_SRC = f.read()


def build(cu_path, name):
    from torch.utils.cpp_extension import load_inline

    with open(cu_path) as f:
        cu = f.read()
    combined = CUH_SRC + "\n" + cu.replace('#include "packed_ternary.cuh"', "")
    return load_inline(
        name=name,
        cpp_sources=r"""
        #include <cuda_runtime.h>
        #include <torch/extension.h>
        extern "C" {
            void launch_packed_ternary_update_tc_v2(
                const void* X, const void* dY, uint32_t* W, int16_t* counter,
                int batch_size, int in_features, int out_features,
                int stride_words, int16_t threshold, cudaStream_t stream);
        }
        void upd(torch::Tensor X, torch::Tensor dY, torch::Tensor W,
                 torch::Tensor counter, int64_t in_features, int64_t threshold) {
            launch_packed_ternary_update_tc_v2(
                X.data_ptr<at::Half>(), dY.data_ptr<at::Half>(),
                reinterpret_cast<uint32_t*>(W.data_ptr<int32_t>()),
                reinterpret_cast<int16_t*>(counter.data_ptr<int16_t>()),
                X.size(0), (int)in_features, (int)W.size(0),
                (int)W.size(1), (int16_t)threshold, nullptr);
        }
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("upd", &upd); }
        """,
        cuda_sources=[combined],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def main():
    dev = "cuda"
    torch.manual_seed(0)
    BATCH = 16384
    THRESHOLD = 8  # small enough that flips actually happen during parity

    print("building OLD (production, CTA barriers) ...", flush=True)
    old = build(OLD_CU, "upd_sw_old")
    print("built OLD ok", flush=True)
    print("building NEW (warp-scope barriers) ...", flush=True)
    new = build(NEW_CU, "upd_sw_new")
    print("built NEW ok", flush=True)

    print(
        f"\n{'name':<6} {'in':>5} {'out':>6}   {'old_ms':>8} {'new_ms':>8} {'d%':>7}"
        f"   {'W':>6} {'counter':>8}",
        flush=True,
    )

    for name, inn, out in [
        ("fc1", 1024, 4096),
        ("fc2", 4096, 1024),
        ("head", 1024, 50272),
    ]:
        X = torch.randn(BATCH, inn, device=dev, dtype=torch.float16) * 0.1
        dY = torch.randn(BATCH, out, device=dev, dtype=torch.float16) * 0.1
        W0 = torch.randint(0, 4, (out, (inn + 15) // 16), device=dev, dtype=torch.int32)
        W0 = torch.where(W0 == 3, 0, W0)
        C0 = torch.zeros(out * inn, device=dev, dtype=torch.int16)

        # parity: both kernels mutate W and counter in place, so run each on a
        # private copy and compare bit-for-bit.
        Wo, Co = W0.clone(), C0.clone()
        Wn, Cn = W0.clone(), C0.clone()
        old.upd(X, dY, Wo, Co, inn, THRESHOLD)
        new.upd(X, dY, Wn, Cn, inn, THRESHOLD)
        torch.cuda.synchronize()
        w_same = torch.equal(Wo, Wn)
        c_same = torch.equal(Co, Cn)
        n_flip = int((Wo != W0).sum().item())
        print(
            f"  parity {name}: W {'EXACT' if w_same else 'DIFF'} "
            f"counter {'EXACT' if c_same else 'DIFF'} flips={n_flip}",
            flush=True,
        )
        if not (w_same and c_same):
            raise SystemExit(f"PARITY FAIL on {name} — refusing to report timings")

        # timings on fresh state each iteration (update is destructive)
        def timed(fn, iters=5, warmup=2):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1000

        def run_old():
            old.upd(X, dY, W0.clone(), C0.clone(), inn, THRESHOLD)

        def run_new():
            new.upd(X, dY, W0.clone(), C0.clone(), inn, THRESHOLD)

        to, tn = [], []
        for _ in range(3):
            to.append(timed(run_old))
            tn.append(timed(run_new))
        mo, mn = min(to), min(tn)
        d = 100.0 * (mn - mo) / mo
        print(
            f"{name:<6} {inn:>5} {out:>6}   {mo:8.2f} {mn:8.2f} {d:7.2f}"
            f"   {'ok' if w_same else 'BAD':>6} {'ok' if c_same else 'BAD':>8}",
            flush=True,
        )


if __name__ == "__main__":
    main()
