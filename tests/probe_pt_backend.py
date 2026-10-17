"""Phase 2b probe: PT_BACKEND=cublas equivalence + acceptance.

Checks, on T4:
  1. fwd bit-parity: cublas-backend Y vs custom-kernel Y on the same
     packed W / X (fp16 matmul vs WMMA — tolerance, not bit-exact; report
     max abs diff and cosine similarity).
  2. dX same.
  3. fwd GEMM TFLOPS at B=16384, K=1024, N=4096 (brief acceptance:
     >= 10 TFLOPS).
  4. whole-step time vs the custom baseline (Phase 0: dispatch fwd 86.04,
     dX 42.02, update 46.53 ms at fc1).

Run with: PT_BACKEND=cublas python3 tests/probe_pt_backend.py
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.getcwd())

import torch


def timeit(fn, warmup=10, trials=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16384)
    args, _ = ap.parse_known_args()   # colab exec appends -f <json>

    B = args.batch
    backend = os.environ.get("PT_BACKEND", "custom")
    print(f"PT_BACKEND={backend} B={B}")

    from kernels.packed_ternary import pack_tensor
    from kernels.packed_ternary.custom_ops import forward_tc, backward_dx_tc

    results = {}
    for name, K, N in [("fc1", 1024, 4096), ("fc2", 4096, 1024), ("head", 1024, 50272)]:
        torch.manual_seed(0)
        W = pack_tensor((torch.randn(N, K, device="cuda") * 0.05)).cuda()
        X = torch.randn(B, K, device="cuda", dtype=torch.float16) * 0.5
        dY = torch.randn(B, N, device="cuda", dtype=torch.float16) * 0.1

        # force BOTH backends for the same inputs: the env picks one, so run
        # the other by direct module call
        from kernels.packed_ternary import cublas_backend as cb
        from kernels.packed_ternary.pack_forward import packed_ternary_forward_tc as fwd_custom
        from kernels.packed_ternary.pack_update import backward_dx as dx_custom
        import importlib

        Y_custom = fwd_custom(W, X) if fwd_custom is not None else None
        dX_custom = dx_custom(W, dY, K)
        Y_cublas = cb.forward(W, X, K)
        dX_cublas = cb.backward_dx(W, dY, K)

        row = {"shape": name, "K": K, "N": N}
        if Y_custom is not None:
            row["fwd_max_abs"] = (Y_custom.float() - Y_cublas.float()).abs().max().item()
            row["fwd_cos"] = torch.nn.functional.cosine_similarity(
                Y_custom.float().flatten(), Y_cublas.float().flatten(), dim=0).item()
        row["dx_max_abs"] = (dX_custom.float() - dX_cublas.float()).abs().max().item()
        row["dx_cos"] = torch.nn.functional.cosine_similarity(
            dX_custom.float().flatten(), dX_cublas.float().flatten(), dim=0).item()

        # timing on the ACTIVE backend
        row["fwd_ms"] = timeit(lambda: forward_tc(W, X, K))
        row["dx_ms"] = timeit(lambda: backward_dx_tc(W, dY, K))
        flops = 2 * B * N * K
        row["fwd_tflops"] = flops / (row["fwd_ms"] * 1e-3) / 1e12
        results[name] = row
        print(f"{name}: fwd {row['fwd_ms']:8.3f} ms ({row['fwd_tflops']:6.2f} TFLOPS) "
              f"dx {row['dx_ms']:8.3f} ms | fwd_max_abs={row.get('fwd_max_abs')} "
              f"cos={row.get('fwd_cos')} dx_max_abs={row['dx_max_abs']}")

    ok = results["fc1"]["fwd_tflops"] >= 10.0
    print(f"\nACCEPTANCE fwd >= 10 TFLOPS @16384x4096x4096: "
          f"{'PASS' if ok else 'FAIL'} ({results['fc1']['fwd_tflops']:.2f})")


if __name__ == "__main__":
    main()
