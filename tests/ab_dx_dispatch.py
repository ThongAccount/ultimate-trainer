"""Decisive A/B: full model backward with 64x64 vs forced-32x32 dX dispatch.

Same process, no file swapping, no cache ambiguity. Monkeypatches
custom_ops._dx_tc_64=None to force the 32x32 path, so we get a clean
apples-to-apples of the dispatch change on FULL step + per-kernel attribution.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import torch

from train_gigatoken import N_LAYERS, K, VOCAB
from kernels.packed_ternary import PackedTernaryLinear
import kernels.packed_ternary.custom_ops as co

B, SEQ = 32, 512
THRESHOLD = 32

class TernaryTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, K)
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(K) for _ in range(N_LAYERS)])
        self.fc1s = torch.nn.ModuleList([PackedTernaryLinear(K, 4*K, threshold=THRESHOLD) for _ in range(N_LAYERS)])
        self.fc2s = torch.nn.ModuleList([PackedTernaryLinear(4*K, K, threshold=THRESHOLD) for _ in range(N_LAYERS)])
        self.head = PackedTernaryLinear(K, VOCAB, threshold=THRESHOLD)

    def forward(self, x):
        Bt, T = x.shape
        scale = K ** -0.5
        h = self.embed(x).half().view(Bt * T, K)
        for i in range(N_LAYERS):
            h = self.norms[i](h.float()).half()
            h = self.fc1s[i](h) * scale
            h = torch.nn.functional.gelu(h)
            h = self.fc2s[i](h) * scale
        h = self.head(h) * scale
        return h.view(Bt, T, VOCAB)

torch.manual_seed(0)
model = TernaryTransformer().cuda().train()
x = torch.randint(0, VOCAB, (B, SEQ), device="cuda")
y = x.clone()
loss_fn = torch.nn.CrossEntropyLoss()

# warmup (dispatch both ways so both kernels are compiled/cached)
co._ensure_loaded()
for _ in range(3):
    model.zero_grad(set_to_none=True)
    logits = model(x); loss = loss_fn(logits.view(-1, VOCAB), y.view(-1)); loss.backward()
torch.cuda.synchronize()

def bench_bwd(label, force32):
    saved = co._dx_tc_64
    if force32:
        co._dx_tc_64 = None
    # warmup this config
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        logits = model(x); loss = loss_fn(logits.view(-1, VOCAB), y.view(-1)); loss.backward()
    torch.cuda.synchronize()
    ts = []
    for _ in range(10):
        model.zero_grad(set_to_none=True)
        logits = model(x); loss = loss_fn(logits.view(-1, VOCAB), y.view(-1))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1000)
    co._dx_tc_64 = saved
    avg = sum(ts) / len(ts)
    print(f"  {label}: bwd {avg:.1f} ms/step  (min {min(ts):.1f}, max {max(ts):.1f})", flush=True)
    return avg

print("=== A/B: 64x64 vs forced-32x32 backward dX dispatch (full backward) ===", flush=True)
a64 = bench_bwd("64x64 dispatch  ", force32=False)
a32 = bench_bwd("32x32 forced    ", force32=True)
print(f"\n  speedup: {a32/a64:.3f}x  (saving {a32-a64:.1f} ms/step)", flush=True)

# per-layer attribution on the 64x64 path
co._dx_tc_64 = None  # will be restored after
co._ensure_loaded()
print("\n=== per-shape backward_dx timing (64x64 path) ===", flush=True)
import kernels.packed_ternary.custom_ops as _co
# restore 64
_ensure = _co._ensure_loaded
for (B_, N_out_, K_) in [(16384, 4096, 1024), (16384, 1024, 4096), (16384, 50272, 1024)]:
    nw = (K_ + 15) // 16
    Wp = torch.zeros(N_out_, nw, dtype=torch.int32, device="cuda")
    dY = (torch.randn(B_, N_out_, device="cuda") * 0.5).half()
    def call_64(): return _co.backward_dx_tc(Wp, dY, K_)
    def call_32():
        _0 = _co._dx_tc_64; _co._dx_tc_64 = None
        o = _co.backward_dx_tc(Wp, dY, K_)
        _co._dx_tc_64 = _0
        return o
    for _ in range(3): call_64(); call_32()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20): call_64()
    torch.cuda.synchronize(); t64 = (time.perf_counter()-t0)/20*1000
    t0 = time.perf_counter()
    for _ in range(20): call_32()
    torch.cuda.synchronize(); t32 = (time.perf_counter()-t0)/20*1000
    print(f"  B={B_} N_out={N_out_} K={K_}: 64x64={t64:.1f}ms  32x32={t32:.1f}ms  speedup={t32/t64:.2f}x", flush=True)