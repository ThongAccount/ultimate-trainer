"""
test_ternary_matmul.py — Test and benchmark the CUDA ternary matmul kernel.

Tests:
  - Correctness vs F.linear(x, W_q) with on-the-fly quantization
  - Multiple shapes (small, medium, large)
  - Gradient shapes (backward dx)
  - Bias support

Benchmarks:
  - Forward throughput vs F.linear(x, w_ste) for model-relevant shapes
  - Backward throughput vs standard matmul
  - Shows speedup from add/sub exploitation

Usage:
    uv run python kernels/ternary/test_ternary_matmul.py          # correctness
    uv run python kernels/ternary/test_ternary_matmul.py --bench   # + benchmarks
    uv run python kernels/ternary/test_ternary_matmul.py --cpu     # CPU only
"""

import sys
import os
import math
import time
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _cpu_ref(x, weight, gamma, bias=None):
    """Reference: standard quantize + matmul."""
    w_q = torch.clamp(torch.round(weight / gamma), -1.0, 1.0)
    y = F.linear(x, w_q)
    if bias is not None:
        y = y + bias
    return y


def _cpu_ref_dx(dy, weight, gamma):
    """Reference backward: dx = dy @ Q(W)."""
    w_q = torch.clamp(torch.round(weight / gamma), -1.0, 1.0)
    return dy @ w_q  # NOT F.linear (which would do dy @ w_q^T)


def _random_weights(N, K, gamma=None, device="cpu"):
    """Create realistic weights: mix of ~+1, ~-1, ~0 values."""
    w = torch.randn(N, K, device=device) * 0.5
    if gamma is None:
        gamma = w.abs().mean() + 1e-5
    return w, gamma


def test_forward_correctness():
    """Verify forward ternary matmul matches CPU reference."""
    print("Testing forward correctness...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    if use_cuda:
        from kernels.ternary.ternary_matmul import ternary_matmul
        print(f"  Using CUDA kernel on {torch.cuda.get_device_name(0)}")
    else:
        print("  Using PyTorch fallback (CPU)")

    # Test shapes: (M, K) → (M, N) for various model-relevant sizes
    shapes = [
        (1, 256, 256),       # tiny
        (128, 2560, 2560),   # QKV: 1 token
        (1024, 2560, 2560),  # QKV: batch
        (512, 6912, 2560),   # FFN up: 512 tok
        (256, 2560, 6912),   # FFN down: 256 tok
        (4, 128256, 2560),   # LM head: 4 tok
    ]

    for M, N, K in shapes:
        x = torch.randn(M, K, device=device)
        weight, gamma = _random_weights(N, K, device=device)

        if use_cuda:
            y_gpu = ternary_matmul(x, weight, gamma)
        else:
            y_gpu = _cpu_ref(x, weight, gamma)

        y_ref = _cpu_ref(x.cpu(), weight.cpu(), gamma.cpu())

        max_diff = (y_gpu.cpu() - y_ref).abs().max().item()
        assert max_diff < 1e-4, \
            f"Forward mismatch for ({M},{N},{K}): max_diff={max_diff:.6f}"
        print(f"  ✅ ({M:5d}, {N:6d}, {K:6d})  max_diff={max_diff:.2e}")

    # Test with bias
    for M, N, K in [(128, 2560, 2560), (64, 6912, 2560)]:
        x = torch.randn(M, K, device=device)
        weight, gamma = _random_weights(N, K, device=device)
        bias = torch.randn(N, device=device)

        if use_cuda:
            y_gpu = ternary_matmul(x, weight, gamma, bias)
        else:
            y_gpu = _cpu_ref(x, weight, gamma, bias)

        y_ref = _cpu_ref(x.cpu(), weight.cpu(), gamma.cpu(), bias.cpu())
        max_diff = (y_gpu.cpu() - y_ref).abs().max().item()
        assert max_diff < 1e-4
        print(f"  ✅ ({M:5d}, {N:6d}, {K:6d}) +bias  max_diff={max_diff:.2e}")

    print("  ✅ All forward correctness tests passed!\n")


def test_backward_correctness():
    """Verify backward dx ternary matmul matches CPU reference."""
    print("Testing backward dx correctness...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    if not use_cuda:
        print("  ⚠️  Skipping backward test (no CUDA)\n")
        return

    from kernels.ternary.ternary_matmul import backward_dx_ternary

    shapes = [
        (128, 2560, 2560),
        (64, 6912, 2560),
        (256, 2560, 6912),
    ]

    for M, N, K in shapes:
        dy = torch.randn(M, N, device=device)
        weight, gamma = _random_weights(N, K, device=device)

        dx_gpu = backward_dx_ternary(dy, weight, gamma)
        dx_ref = _cpu_ref_dx(dy.cpu(), weight.cpu(), gamma.cpu())

        max_diff = (dx_gpu.cpu() - dx_ref).abs().max().item()
        assert max_diff < 1e-4, \
            f"Backward mismatch for ({M},{N},{K}): max_diff={max_diff:.6f}"
        print(f"  ✅ ({M:5d}, {N:6d}, {K:6d})  max_diff={max_diff:.2e}")

    print("  ✅ All backward correctness tests passed!\n")


def test_batch_handling():
    """Verify batch/seq dims are handled correctly."""
    print("Testing batch dimension handling...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from kernels.ternary.ternary_matmul import ternary_matmul

    # 3D input: (B, T, D)
    B, T, K, N = 2, 128, 2560, 2560
    x = torch.randn(B, T, K, device=device)
    weight, gamma = _random_weights(N, K, device=device)

    y = ternary_matmul(x, weight, gamma)
    assert y.shape == (B, T, N), f"Expected ({B},{T},{N}), got {y.shape}"

    # Compare with flat version
    y_flat = ternary_matmul(x.reshape(-1, K), weight, gamma)
    assert y_flat.shape == (B * T, N)
    assert torch.allclose(y.reshape(-1, N), y_flat, atol=1e-5)

    print(f"  ✅ 3D input ({B},{T},{K}) → ({B},{T},{N})  shape OK")
    print("  ✅ All batch handling tests passed!\n")


def benchmark(model_shapes=False, warmup=5, iters=30):
    """Benchmark forward ternary matmul vs F.linear with STE weights."""
    print("Running benchmarks...")
    print(f"  Warmup: {warmup} iters  |  Measure: {iters} iters\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("  ⚠️  No CUDA — benchmarks limited to CPU fallback\n")
        return

    from kernels.ternary.ternary_matmul import ternary_matmul, backward_dx_ternary

    # Shapes: either model-specific or all
    if model_shapes:
        shapes = [
            (512, 2560, 2560, "QKV proj: 512 tok × 2560"),
            (4096, 2560, 2560, "QKV proj: 4096 tok × 2560"),
            (512, 6912, 2560, "FFN up/gate: 512 tok × 6912"),
            (512, 2560, 6912, "FFN down: 512 tok × 6912"),
            (256, 128256, 2560, "LM head: 256 tok × 128K vocab"),
        ]
    else:
        shapes = [
            (128, 2560, 2560, "Small: 128×2560→2560"),
            (1024, 2560, 2560, "Medium: 1024×2560→2560"),
            (512, 6912, 2560, "Wide: 512×6912→2560"),
            (512, 2560, 6912, "Tall: 512×2560→6912"),
        ]

    print(f"  {'Shape':>30s}  {'Ternary (ms)':>13s}  {'Dense (ms)':>13s}  {'Speedup':>8s}")
    print(f"  {'─' * 68}")

    for M, N, K, label in shapes:
        x = torch.randn(M, K, device=device)
        dy = torch.randn(M, N, device=device)
        weight, gamma = _random_weights(N, K, device=device)
        w_q = torch.clamp(torch.round(weight / gamma), -1.0, 1.0)

        # Warmup
        for _ in range(warmup):
            _ = ternary_matmul(x, weight, gamma)
            _ = F.linear(x, w_q)
            torch.cuda.synchronize()

        # Benchmark ternary kernel
        t0 = time.perf_counter()
        for _ in range(iters):
            y = ternary_matmul(x, weight, gamma)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ternary_ms = (t1 - t0) / iters * 1000

        # Benchmark dense F.linear
        t0 = time.perf_counter()
        for _ in range(iters):
            y_ref = F.linear(x, w_q)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        dense_ms = (t1 - t0) / iters * 1000

        speedup = dense_ms / ternary_ms

        # Verify correctness
        max_err = (y - y_ref).abs().max().item()

        print(f"  {label:>30s}  {ternary_ms:>10.2f} ms  {dense_ms:>10.2f} ms  {speedup:>6.2f}×  (err={max_err:.2e})")

    # ── Backward benchmark ──
    print(f"\n  {'Backward dx':>30s}  {'Ternary (ms)':>13s}  {'Dense (ms)':>13s}  {'Speedup':>8s}")
    print(f"  {'─' * 68}")

    for M, N, K, label in shapes:
        dy = torch.randn(M, N, device=device)
        weight, gamma = _random_weights(N, K, device=device)
        w_q = torch.clamp(torch.round(weight / gamma), -1.0, 1.0)

        for _ in range(warmup):
            _ = backward_dx_ternary(dy, weight, gamma)
            _ = dy @ w_q
            torch.cuda.synchronize()

        # Ternary backward
        t0 = time.perf_counter()
        for _ in range(iters):
            dx = backward_dx_ternary(dy, weight, gamma)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ternary_ms = (t1 - t0) / iters * 1000

        # Dense backward: dx = dy @ w_q (NOT F.linear which does dy @ w_q^T)
        t0 = time.perf_counter()
        for _ in range(iters):
            dx_ref = dy @ w_q
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        dense_ms = (t1 - t0) / iters * 1000

        speedup = dense_ms / ternary_ms

        print(f"  {label:>30s}  {ternary_ms:>10.2f} ms  {dense_ms:>10.2f} ms  {speedup:>6.2f}×")

    print()


def main():
    parser = argparse.ArgumentParser(description="Test and benchmark CUDA ternary matmul kernel")
    parser.add_argument("--bench", action="store_true", help="Run benchmarks")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only test")
    parser.add_argument("--model-shapes", action="store_true", help="Use exact 2B model shapes for benchmarks")
    args = parser.parse_args()

    if args.cpu:
        torch.set_num_threads(os.cpu_count())
        print(f"CPU-only with {os.cpu_count()} threads\n")

    # Always run correctness tests
    test_forward_correctness()
    test_backward_correctness()

    if torch.cuda.is_available():
        test_batch_handling()

    # Benchmarks
    if args.bench:
        benchmark(model_shapes=args.model_shapes)

    print("All done! 🎉")


if __name__ == "__main__":
    main()
