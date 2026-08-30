"""Clean head-only A/B: isolate incremental gain of the N_out gate fix.

old = MLP 64x64 + head 32x32   (pre-fix)
new = MLP 64x64 + head 64x64   (post-fix)

Routes ONLY head shape (N_out == VOCAB) to 32x32; MLP layers keep normal dispatch.
"""
import sys, os, time
sys.path.insert(0, os.getcwd())
import torch
torch.manual_seed(0)

import kernels.packed_ternary.custom_ops as co
import kernels.packed_ternary.packed_linear as pl
from train_gigatoken import N_LAYERS, K, VOCAB
from kernels.packed_ternary import PackedTernaryLinear

B, SEQ = 32, 512
THRESHOLD = 32

co._ensure_loaded()
_dx32 = co._dx_tc
_orig = co.backward_dx_tc


class TernaryTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, K)
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(K) for _ in range(N_LAYERS)])
        self.fc1s = torch.nn.ModuleList([PackedTernaryLinear(K, 4 * K, threshold=THRESHOLD) for _ in range(N_LAYERS)])
        self.fc2s = torch.nn.ModuleList([PackedTernaryLinear(4 * K, K, threshold=THRESHOLD) for _ in range(N_LAYERS)])
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


def run_bwd():
    model.zero_grad(set_to_none=True)
    logits = model(x); loss = loss_fn(logits.view(-1, VOCAB), y.view(-1)); loss.backward()


def bench_bwd(label):
    for _ in range(3):
        run_bwd()
    torch.cuda.synchronize()
    ts = []
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_bwd()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1000)
    avg = sum(ts) / len(ts)
    print(f"  {label}: bwd {avg:.1f} ms/step (min {min(ts):.1f})", flush=True)
    return avg


def _head32_wrapper(W, dY, Kk):
    if dY.size(1) == VOCAB:            # head layer only -> 32x32
        return _dx32(W.contiguous(), dY.contiguous(), Kk)
    return _orig(W, dY, Kk)


old_binding = pl.co_backward_dx_tc
pl.co_backward_dx_tc = _head32_wrapper
a_old = bench_bwd("head->32 (old)   ")
pl.co_backward_dx_tc = old_binding
a_new = bench_bwd("head->64 (new)   ")

print(f"\n  incremental head saving: {a_old - a_new:.1f} ms/step", flush=True)
print(f"  backward speedup: {a_old/a_new:.3f}x", flush=True)