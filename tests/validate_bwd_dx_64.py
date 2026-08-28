"""Validate 64x64 backward dX kernel vs torch reference + benchmark vs 32x32.

Runs on Modal T4. Tests the exact gigatoken GEMM shapes and the dispatch path
(custom_ops.backward_dx_tc should route 64x64 multipled dims to the 64x64 kernel).
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
import torch
torch.manual_seed(0)

def decode_ternary(Wp, N, K):
    """Unpack packed uint32 words -> ternary {-1,0,1} float32 [N,K]."""
    nw = (K + 15) // 16
    tern = torch.zeros(N, K, dtype=torch.float32, device="cuda")
    for j in range(nw):
        for i in range(16):
            k = j * 16 + i
            if k < K:
                v = (Wp[:, j] >> (2 * i)) & 3
                tern[:, k] = torch.where(v == 1, torch.tensor(1.0, device="cuda"),
                              torch.where(v == 2, torch.tensor(-1.0, device="cuda"),
                                           torch.tensor(0.0, device="cuda")))
    return tern

def make_packed(N, K):
    nw = (K + 15) // 16
    codes = torch.randint(0, 3, (N, K), device="cuda")
    Wp = torch.zeros(N, nw, dtype=torch.int32, device="cuda")
    for k in range(K):
        Wp[:, k // 16] |= (codes[:, k] << (2 * (k % 16)))
    return Wp.contiguous()

import kernels.packed_ternary.pack_update as pu
from kernels.packed_ternary.custom_ops import backward_dx_tc as co_bwd

# ── Correctness ──
print("=== CORRECTNESS: 64x64 backward dX vs torch reference ===")
# gigatoken shapes: (B, N_out, K_in) for fc1/fc2/head backward
shapes = [
    (16384, 4096, 1024),   # fc1 backward
    (16384, 1024, 4096),   # fc2 backward
    (16384, 1024, 1024),
    (512, 4096, 1024),
    (256, 1024, 4096),
    (128, 64, 64),
    (64, 64, 64),
]
pu._load_dx_tc()
pu._load_dx_tc_32()
print(f"dx_tc (64x64) loaded: {pu._HAS_DX_TC},  dx_tc_32 loaded: {pu._HAS_DX_TC_32}")

all_ok = True
for (B, N_out, K) in shapes:
    Wp = make_packed(N_out, K)
    W_ter = decode_ternary(Wp, N_out, K)
    dY = (torch.randn(B, N_out, device="cuda") * 0.5).half()
    ref = dY.float() @ W_ter  # [B, K]

    d64 = pu._dx_tc_fn(Wp, dY, K).float() if pu._HAS_DX_TC else None
    d32 = pu._dx_tc_32_fn(Wp, dY, K).float() if pu._HAS_DX_TC_32 else None

    e64 = (d64 - ref).abs().max().item() if d64 is not None else float('nan')
    e32 = (d32 - ref).abs().max().item() if d32 is not None else float('nan')
    ok = e64 < 0.05 * ref.abs().max().item()
    all_ok &= ok
    print(f"  B={B:6d} N={N_out:5d} K={K:5d}  refmax={ref.abs().max().item():8.2f} "
          f"err64={e64:.4f} err32={e32:.4f}  {'OK' if ok else 'FAIL 64x64'}")

# ── Dispatch check: co_bwd should route 64-mult dims to 64x64 ──
print("\n=== DISPATCH: custom_ops.backward_dx_tc routing ===")
for (B, N_out, K) in [(16384, 4096, 1024), (64, 64, 64)]:
    Wp = make_packed(N_out, K)
    W_ter = decode_ternary(Wp, N_out, K)
    dY = (torch.randn(B, N_out, device="cuda") * 0.5).half()
    ref = dY.float() @ W_ter
    out = co_bwd(Wp, dY, K).float()
    err = (out - ref).abs().max().item()
    print(f"  co_bwd B={B} N={N_out} K={K}: err={err:.4f} (refmax={ref.abs().max().item():.2f})")

# ── Benchmark ──
print("\n=== BENCHMARK: 64x64 vs 32x32 backward dX (ms, 20 iters, gigatoken fc1) ===")
B, N_out, K = 16384, 4096, 1024
Wp = make_packed(N_out, K)
dY = (torch.randn(B, N_out, device="cuda") * 0.5).half()

def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

if pu._HAS_DX_TC:
    t64 = bench(lambda: pu._dx_tc_fn(Wp, dY, K))
    print(f"  64x64 backward dX: {t64:.2f} ms")
if pu._HAS_DX_TC_32:
    t32 = bench(lambda: pu._dx_tc_32_fn(Wp, dY, K))
    print(f"  32x32 backward dX: {t32:.2f} ms")
    if pu._HAS_DX_TC:
        print(f"  speedup 64 vs 32: {t32/t64:.2f}x")

print(f"\n{'ALL PASS' if all_ok else 'FAILURES PRESENT'}")