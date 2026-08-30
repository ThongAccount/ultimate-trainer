"""Probe: does the 64x64 backward-dX kernel correctly handle the HEAD shape?

Head backward dX: dX = dY @ W, dY[B=16384, N_out=50272], K(in)=1024.
The 64x64 kernel tiles B x K at 64 and reduces over N (out_features) in
16-steps (kWMMA_K=16) with tail zero-pad. It should NOT require N % 64 == 0.

The dispatch (custom_ops.backward_dx_tc) currently gates on N_out % 64 == 0,
excluding the head (50272 % 64 = 32). This probe calls the 64x64 kernel
DIRECTLY on the head shape (bypassing dispatch) to test correctness + speed.
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
import torch
torch.manual_seed(0)

import kernels.packed_ternary.pack_update as pu


def decode_ternary(Wp, N, K):
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


pu._load_dx_tc()       # 64x64
pu._load_dx_tc_32()    # 32x32
print(f"dx_tc(64) loaded={pu._HAS_DX_TC}  dx_tc_32 loaded={pu._HAS_DX_TC_32}", flush=True)

print("\n=== CORRECTNESS: 64x64 vs torch ref on non-64-multiple out_features ===", flush=True)
# (B, N_out, K): head shape + a few odd out_features to stress tail handling
shapes = [
    (16384, 50272, 1024),  # head (VOCAB)
    (16384, 4096, 1024),   # fc1 control (64-multiple N)
    (256, 50272, 1024),    # small-B head
    (128, 50000, 256),     # non-16-multiple N tail stress
    (64, 50272, 1024),     # tiny-B head
]

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
    refmax = ref.abs().max().item()
    ok = e64 < 0.05 * refmax
    all_ok &= ok
    print(f"  B={B:6d} N={N_out:6d} K={K:5d}  refmax={refmax:8.2f} "
          f"err64={e64:.4f} err32={e32:.4f}  {'OK' if ok else 'FAIL 64x64'}", flush=True)

print("\n=== BENCHMARK: head shape (16384, 50272, 1024) ===", flush=True)
B, N_out, K = 16384, 50272, 1024
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


t64 = bench(lambda: pu._dx_tc_fn(Wp, dY, K)) if pu._HAS_DX_TC else float('nan')
t32 = bench(lambda: pu._dx_tc_32_fn(Wp, dY, K)) if pu._HAS_DX_TC_32 else float('nan')
print(f"  64x64 head dX: {t64:.1f} ms", flush=True)
print(f"  32x32 head dX: {t32:.1f} ms", flush=True)
if pu._HAS_DX_TC and pu._HAS_DX_TC_32:
    print(f"  speedup 64 vs 32: {t32/t64:.2f}x", flush=True)

print(f"\n{'ALL PASS' if all_ok else 'FAILURES PRESENT'}", flush=True)