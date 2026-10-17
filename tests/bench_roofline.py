"""Phase 0 roofline: dispatch path vs pre-decoded fp16 cuBLAS at B=16384.

Per (K,N) shape from the production model, times:
  (a) the current packed-ternary dispatch path (fwd / dX / update)
  (b) pre-decoded fp16 W with torch.matmul (cuBLAS) — decode cost reported
      separately, not amortized into the GEMM time
Also logs SM clock + torch/CUDA versions + git commit into a JSON block.

Usage (Colab T4):
    TORCH_CUDA_ARCH_LIST=7.5 python3 tests/bench_roofline.py \
        --batch 16384 --out /content/roofline.json

Protocol: warmup 10, median of >= 20, torch.cuda.Event timing.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.getcwd())

# Production shapes from train_gigatoken.py: 6 layers fc1(1024->4096) +
# fc2(4096->1024) + head(1024->50272).  fc layers share K/N pairing.
SHAPES = [
    ("fc1", 1024, 4096),
    ("fc2", 4096, 1024),
    ("head", 1024, 50272),
]


def gpu_info() -> dict:
    import torch

    info = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "branch": subprocess.run(
            ["git", "branch", "--show-current"], capture_output=True, text=True
        ).stdout.strip(),
    }
    # SM clock (best effort; often permission-gated in hosted VMs)
    try:
        clk = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip()
        info["sm_clock_during_run"] = clk
    except Exception:
        info["sm_clock_during_run"] = None
    return info


def timeit(fn, warmup=10, trials=20):
    """Median-of-trials ms via cuda events. fn() must sync internally or not."""
    import torch

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


def bench_shape(name, K, N, B, dev):
    import torch
    from kernels.packed_ternary import pack_tensor
    from kernels.packed_ternary.pack_decode import decode_packed as _decode
    from kernels.packed_ternary.pack_update import backward_dx, update
    from kernels.packed_ternary.custom_ops import forward_tc

    torch.manual_seed(0)
    W_fp = torch.randn(N, K, device=dev) * 0.05
    W_packed = pack_tensor(W_fp.cuda().half())           # int32 words [N, stride]
    W16 = _decode(W_packed, N, K)  # [N, K] fp16 (GPU decode kernel)
    X = torch.randn(B, K, device=dev, dtype=torch.float16) * 0.5
    dY = torch.randn(B, N, device=dev, dtype=torch.float16) * 0.1

    row = {"shape": name, "K": K, "N": N, "B": B}

    # ---- (a) dispatch path ----
    try:
        row["dispatch_fwd_ms"] = timeit(lambda: forward_tc(W_packed, X, K))
    except Exception as e:
        row["dispatch_fwd_ms"] = f"ERR {type(e).__name__}: {e}"
    try:
        row["dispatch_dx_ms"] = timeit(lambda: backward_dx(W_packed, dY, K))
    except Exception as e:
        row["dispatch_dx_ms"] = f"ERR {type(e).__name__}: {e}"
    # update mutates in place -> fresh clone per trial OUTSIDE the timed region
    try:
        counter = torch.zeros(N, K, dtype=torch.int16, device=dev)
        W0, C0 = W_packed.clone(), counter.clone()

        def _one_update():
            e0 = torch.cuda.Event(True)
            e1 = torch.cuda.Event(True)
            W_packed.copy_(W0)
            counter.copy_(C0)
            torch.cuda.synchronize()
            e0.record()
            update(W_packed, counter, X, dY, 32)
            e1.record()
            torch.cuda.synchronize()
            return e0.elapsed_time(e1)

        for _ in range(10):           # warmup
            _one_update()
        row["dispatch_update_ms"] = statistics.median(
            [_one_update() for _ in range(20)]
        )
    except Exception as e:
        row["dispatch_update_ms"] = f"ERR {type(e).__name__}: {e}"

    # ---- (b) decode (reported separately) + cuBLAS GEMMs ----
    try:
        row["decode_ms"] = timeit(lambda: _decode(W_packed, N, K))
    except Exception:
        row["decode_ms"] = None

    if isinstance(row.get("decode_ms"), float):
        row["cublas_fwd_ms"] = timeit(lambda: torch.matmul(X, W16.t()))
        row["cublas_dx_ms"] = timeit(lambda: torch.matmul(dY, W16))
        row["cublas_dw_ms"] = timeit(lambda: torch.matmul(dY.t(), X))

    # TFLOPS: fwd GEMM = 2*B*N*K
    flops = 2 * B * N * K
    for key in ("dispatch_fwd_ms", "cublas_fwd_ms"):
        v = row.get(key)
        row[key.replace("_ms", "_tflops")] = (
            flops / (v * 1e-3) / 1e12 if isinstance(v, float) and v > 0 else None
        )

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--out", default="/content/roofline.json")
    ap.add_argument("--shapes", default="all",
                    help="'all' or comma list of fc1,fc2,head")
    args = ap.parse_args()

    import torch
    assert torch.cuda.is_available(), "no CUDA"
    cap = torch.cuda.get_device_capability(0)
    assert cap == (7, 5), f"NOT a T4: {torch.cuda.get_device_name(0)} cap={cap} — numbers invalid"
    dev = "cuda"

    wanted = (SHAPES if args.shapes == "all"
              else [s for s in SHAPES if s[0] in args.shapes.split(",")])

    rows = []
    for name, K, N in wanted:
        print(f"--- {name} B={args.batch} K={K} N={N} ---", flush=True)
        row = bench_shape(name, K, N, args.batch, dev)
        rows.append(row)
        for k, v in row.items():
            if isinstance(v, float):
                print(f"  {k:28s} {v:10.3f}")
            elif v is not None and not isinstance(v, dict):
                print(f"  {k:28s} {v}")

    info = gpu_info()
    info["batch"] = args.batch
    payload = {"roofline": rows, "env": info}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nRESULT JSON: {json.dumps(payload, indent=2)}")


if __name__ == "__main__":
    main()
