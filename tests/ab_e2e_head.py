"""Single-process interleaved e2e A/B: head->32 (pre-fix) vs head->64 (post-fix).

Measures FULL train step (fwd+bwd) wall time + tok/s, alternating configs
within one process to cancel Modal T4 thermal/clock drift.
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
STEPS_PER = 12

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


def run_step():
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = loss_fn(logits.view(-1, VOCAB), y.view(-1))
    loss.backward()
    return loss.detach()


def time_steps(head64):
    import kernels.packed_ternary.packed_linear as _pl
    if not head64:
        _pl.co_backward_dx_tc = _head32_wrapper
    else:
        _pl.co_backward_dx_tc = _orig_binding
    # warmup
    for _ in range(2):
        run_step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(STEPS_PER):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1000)
    return ts


_orig_binding = pl.co_backward_dx_tc


def _head32_wrapper(W, dY, Kk):
    if dY.size(1) == VOCAB:
        return _dx32(W.contiguous(), dY.contiguous(), Kk)
    return _orig(W, dY, Kk)


# interleaved: alternate to cancel drift (A,B,A,B,...)
labels, ts32, ts64 = [], [], []
for rnd in range(3):
    l = time_steps(False); ts32 += l; labels += ["32"] * len(l)
    l = time_steps(True);  ts64 += l; labels += ["64"] * len(l)

tok_per_step = B * SEQ


def report(name, ts):
    avg = sum(ts) / len(ts)
    mn = min(ts)
    tps = tok_per_step / (avg / 1000.0)
    print(f"  head->{name}: {avg:.1f} ms/step  tok/s={tps:.0f}  (min {mn:.1f}ms)", flush=True)
    return avg


a32 = report("32", ts32)
a64 = report("64", ts64)
print(f"\n  e2e step speedup: {a32/a64:.3f}x  saving {a32-a64:.1f} ms/step", flush=True)
print(f"  e2e tok/s gain: {(a32-a64)/a32*100:.1f}%", flush=True)