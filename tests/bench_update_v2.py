"""Isolated update_tc_v2 kernel baseline + breakdown (in-process).

Times the kernel directly (bypassing autograd) for each layer shape, so we can
attribute where the 1383ms goes and separate:
  (a) WMMA accumulation (GEMM) time
  (b) counter-update + atomic flip time

Also runs a full PDP (forward/backward/update) so we have end-to-end numbers.
"""
import torch, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.packed_ternary import custom_ops as co
from kernels.packed_ternary import pack_update as pu

DEV = "cuda"
B, SEQ = 32, 512
BATCH = B * SEQ  # 16384 reduction dim

def bench_fn(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

def make_tensors(inn, out):
    X = torch.randn(BATCH, inn, device=DEV, dtype=torch.float16)
    dY = torch.randn(BATCH, out, device=DEV, dtype=torch.float16) * 0.5
    W = torch.zeros(out, (inn + 15) // 16, device=DEV, dtype=torch.int32)
    counter = torch.zeros(out, inn, device=DEV, dtype=torch.int16)
    return X, dY, W, counter

co._ensure_loaded()

print("=" * 70)
print("update_tc_v2 isolated kernel timing (bypass autograd)")
print("=" * 70)
shapes = [("fc1", 1024, 4096), ("fc2", 4096, 1024), ("head", 1024, 50272)]
for name, inn, out in shapes:
    X, dY, W, counter = make_tensors(inn, out)
    t = bench_fn(lambda: co.update_tc_v2(W, counter, X, dY, 32))
    flop = 2 * BATCH * out * inn
    print(f"  {name:5s} in={inn:5d} out={out:6d}: {t:7.2f} ms  "
          f"({flop / 1e9:.1f} GFLOP, {flop / t / 1e9:.1f} GFLOP/ms)")
print()
print("=" * 70)
print("full backward (dX + update) via autograd, per-step")
print("=" * 70)

from kernels.packed_ternary import PackedTernaryLinear
import torch.nn.functional as F

class Mini(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(1024)
        self.fc1 = PackedTernaryLinear(1024, 4096, threshold=32)
        self.fc2 = PackedTernaryLinear(4096, 1024, threshold=32)
        self.head = PackedTernaryLinear(1024, 50272, threshold=32)
    def forward(self, x):
        Bt, T = x.shape
        h = self.norm(self.fc1(x.view(Bt*T, 1024)).float()).half() * 0.03
        h = self.norm(self.fc2(F.gelu(h)).float()).half() * 0.03
        return self.head(h)

torch.manual_seed(0)
m = Mini().cuda().train()
xx = torch.randint(0, 50272, (B, SEQ), device=DEV)
yy = xx.clone()
lf = torch.nn.CrossEntropyLoss()

def one_step():
    m.zero_grad(set_to_none=True)
    logits = m(xx)
    loss = lf(logits.view(-1, 50272), yy.view(-1))
    loss.backward()

t = bench_fn(one_step, iters=10)
print(f"  full step (fwd+bwd, 3 layers): {t:.1f} ms/step")