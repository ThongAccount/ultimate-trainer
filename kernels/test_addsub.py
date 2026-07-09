"""
test_addsub.py — Test and benchmark the CUDA add/sub kernels.

Usage:
    # Unit test (requires CUDA)
    uv run python kernels/test_addsub.py

    # Benchmark on GPU
    uv run python kernels/test_addsub.py --bench

    # CPU-only correctness (no CUDA needed, uses PyTorch fallback)
    uv run python kernels/test_addsub.py --cpu
"""

import sys
import os
import math
import time
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _vec_add_sub_cpu(a, b):
    """Reference PyTorch implementation for verification."""
    return a + b, a - b


def test_correctness():
    """Verify add/sub outputs match PyTorch reference across sizes & shapes."""
    print("Testing correctness...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try loading CUDA extension
    use_cuda = device.type == "cuda"
    if use_cuda:
        from kernels.addsub import vec_add, vec_sub, vec_add_sub
        print(f"  Using CUDA kernels on {torch.cuda.get_device_name(0)}")
    else:
        print("  Using PyTorch fallback (CPU)")

    cases = [
        (1,),          # single element
        (128,),        # one warp
        (1024,),       # one block
        (1_000_000,),  # moderate
        (2, 3, 5),     # multi-dim
        (16, 64, 128), # 3D
    ]

    for shape in cases:
        a = torch.randn(shape, device=device)
        b = torch.randn(shape, device=device)

        if use_cuda:
            c_add = vec_add(a, b)
            c_sub = vec_sub(a, b)
            add, sub = vec_add_sub(a, b)
        else:
            c_add = a + b
            c_sub = a - b
            add, sub = a + b, a - b

        ref_add, ref_sub = _vec_add_sub_cpu(a.cpu(), b.cpu())

        # Move results to CPU for comparison
        if use_cuda:
            c_add = c_add.cpu()
            c_sub = c_sub.cpu()
            add = add.cpu()
            sub = sub.cpu()

        # Verify
        assert torch.allclose(c_add, ref_add, atol=1e-5), \
            f"vec_add failed for shape {shape}: max diff={torch.max(torch.abs(c_add - ref_add)).item()}"
        assert torch.allclose(c_sub, ref_sub, atol=1e-5), \
            f"vec_sub failed for shape {shape}"
        assert torch.allclose(add, ref_add, atol=1e-5), \
            f"vec_add_sub (add) failed for shape {shape}"
        assert torch.allclose(sub, ref_sub, atol=1e-5), \
            f"vec_add_sub (sub) failed for shape {shape}"

        print(f"  ✅ vec_add, vec_sub, vec_add_sub OK for shape {shape}")

    print("  ✅ All correctness tests passed!\n")


def benchmark(batch_size: int = 4, seq_len: int = 4096, hidden_dim: int = 2560,
              warmup: int = 10, iters: int = 50):
    """Benchmark CUDA add/sub against PyTorch reference.

    Uses sizes typical of transformer training: (B, T, D) tensors flattened
    to (B*T*D,) for element-wise operations.
    """
    print("Running benchmarks...")
    print(f"  Tensor shape: ({batch_size}, {seq_len}, {hidden_dim})")
    print(f"  Elements:     {batch_size * seq_len * hidden_dim:,}")
    print(f"  Size:         {batch_size * seq_len * hidden_dim * 4 / 1e9:.2f} GB per tensor\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("  ⚠️  No CUDA available — benchmarks limited to CPU\n")

    N = batch_size * seq_len * hidden_dim
    a = torch.randn(N, device=device)
    b = torch.randn(N, device=device)

    use_cuda = device.type == "cuda"

    def bench_fn(fn, name, reads=2, writes=1):
        """Benchmark a function, measuring throughput in GB/s."""
        # Warmup
        for _ in range(warmup):
            fn(a, b)

        if use_cuda:
            torch.cuda.synchronize()

        # Measure
        t0 = time.perf_counter()
        for _ in range(iters):
            c = fn(a, b)
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        ms = (t1 - t0) / iters * 1000
        bytes_moved = (reads + writes) * N * 4  # 4 bytes per float
        bw = bytes_moved / (ms / 1000) / 1e9  # GB/s

        # Verification
        c_ref = (a + b) if '+' in name else (a - b) if '-' in name else None
        if c_ref is not None and use_cuda:
            max_err = torch.max(torch.abs(c.cpu() - c_ref.cpu())).item()
        else:
            max_err = 0.0

        print(f"  {name:30s}  {ms:8.2f} ms  {bw:8.1f} GB/s  (max err: {max_err:.2e})")
        return ms, bw

    print(f"  {'Operation':30s}  {'Time':>8s}  {'BW':>8s}")
    print(f"  {'─' * 52}")

    if use_cuda:
        from kernels.addsub import vec_add, vec_sub, vec_add_sub

        bench_fn(lambda a, b: torch.add(a, b), "PyTorch a + b", reads=2, writes=1)
        bench_fn(lambda a, b: torch.sub(a, b), "PyTorch a - b", reads=2, writes=1)
        print()

        bench_fn(lambda a, b: vec_add(a, b), "CUDA vec_add", reads=2, writes=1)
        bench_fn(lambda a, b: vec_sub(a, b), "CUDA vec_sub", reads=2, writes=1)
        print()

        # Fused add_sub: 2 reads + 2 writes per call
        bench_fn(
            lambda a, b: vec_add_sub(a, b),
            "CUDA vec_add_sub (fused)",
            reads=2, writes=2
        )

        print()
        # Compare separate vs fused: separate add + sub = 2 launches
        t0 = time.perf_counter()
        for _ in range(iters):
            add = vec_add(a, b)
            sub = vec_sub(a, b)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        separate_ms = (t1 - t0) / iters * 1000

        t0 = time.perf_counter()
        for _ in range(iters):
            add, sub = vec_add_sub(a, b)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fused_ms = (t1 - t0) / iters * 1000

        speedup = separate_ms / fused_ms
        print(f"  Separate add+sub:        {separate_ms:8.2f} ms")
        print(f"  Fused add_sub:           {fused_ms:8.2f} ms")
        print(f"  Speedup (fused vs separate): {speedup:.2f}×")

    else:
        # CPU benchmarks
        bench_fn(lambda a, b: torch.add(a, b), "PyTorch a + b (CPU)")
        bench_fn(lambda a, b: torch.sub(a, b), "PyTorch a - b (CPU)")
        a_np = a.numpy()
        b_np = b.numpy()

    print()


def main():
    parser = argparse.ArgumentParser(description="Test and benchmark CUDA add/sub kernels")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only test")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=2560)
    args = parser.parse_args()

    if args.cpu:
        # Force CPU by not importing CUDA and just testing correctness
        torch.set_num_threads(os.cpu_count())
        print(f"Running on CPU with {os.cpu_count()} threads")
        test_correctness()
        return

    if not torch.cuda.is_available():
        print("⚠️  No CUDA GPU detected — running CPU-only tests")
        test_correctness()
        return

    test_correctness()

    if args.bench:
        benchmark(
            batch_size=args.batch,
            seq_len=args.seq_len,
            hidden_dim=args.hidden,
        )

    print("All done!")


if __name__ == "__main__":
    main()
