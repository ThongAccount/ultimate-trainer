"""Bench: fixed 64×64 update kernel vs production 32×32, per layer shape."""
import torch, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.packed_ternary import pack_update as pu
from kernels.packed_ternary import custom_ops as co

BATCH = 16384

def bench_fn(fn, iters=20, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

import kernels.packed_ternary.pack_update as m

# Load the real 64×64 kernel FIRST — _ensure_loaded/_load_tc_if_needed alias
# _up_tc_v2_fn to the 32×32 fn and set _HAS_UP_TC_V2, which would make
# _load_up_tc_v2() a silent no-op.
pu._load_up_tc_v2()
assert m._HAS_UP_TC_V2 and m._up_tc_v2_64_fn is not None
up64 = m._up_tc_v2_64_fn

co._ensure_loaded()          # now alias 32×32 over the v2 slot; keep 32 handle
up32 = m._up_tc_v2_32_fn
assert up64 is not up32, "loader aliasing bug: 64×64 fn must differ from 32×32"

print(f"{'shape':<28} {'32x32 ms':>10} {'64x64 ms':>10} {'speedup':>8}", flush=True)
for name, inn, out in [("fc1", 1024, 4096), ("fc2", 4096, 1024), ("head", 1024, 50272)]:
    X = torch.randn(BATCH, inn, device="cuda", dtype=torch.float16)
    dY = torch.randn(BATCH, out, device="cuda", dtype=torch.float16) * 0.5
    W = torch.zeros(out, (inn + 15) // 16, device="cuda", dtype=torch.int32)
    counter = torch.zeros(out, inn, device="cuda", dtype=torch.int16)
    t32 = bench_fn(lambda: up32(W, counter, X, dY, 32))
    # 64x64 production gate would only fire on %64 dims; all model dims qualify
    # except head out=50272 (%16 but not %64) — bench it anyway for information.
    try:
        t64 = bench_fn(lambda: up64(W, counter, X, dY, 32))
        print(f"{name:>5} in={inn:5d} out={out:6d} {t32:>10.2f} {t64:>10.2f} {t32/t64:>7.2f}x", flush=True)
    except Exception as e:
        print(f"{name}: 64x64 failed: {e}", flush=True)
