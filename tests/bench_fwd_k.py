"""Isolated bench of forward TC64 via production 64x64 path."""
import torch, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.packed_ternary import pack_forward as pf

BATCH = 16384
VOCAB = 50272

def bench(fn, iters=20, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

pf._load_tc_64()

print(f"{'shape':<28} {'ms':>9}  {'GFLOP/ms':>9}")
for name, inn, out in [("fc1", 1024, 4096), ("fc2", 4096, 1024), ("head", 1024, VOCAB)]:
    X = torch.randn(BATCH, inn, device="cuda", dtype=torch.float16)
    W = torch.zeros(out, (inn + 15) // 16, device="cuda", dtype=torch.int32)
    t = bench(lambda: pf.packed_ternary_forward_tc_64(W, X))
    gf = 2 * BATCH * out * inn / 1e9
    print(f"  {name:>5} in={inn:5d} out={out:6d}  {t:8.2f}  {gf/t:9.1f}", flush=True)