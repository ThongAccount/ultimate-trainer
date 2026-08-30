"""Decisive validation of relaxed N_out gate: head dX to 64x64.

1. Confirm dispatch routes head (16384,50272,1024) to 64x64 kernel.
2. Bit-exact check: 64x64 vs 32x32 vs torch ref on head + edge shapes.
3. Full-backward A/B: new dispatch (head->64) vs forced-32 (head->32).
"""
import sys, os, time
sys.path.insert(0, os.getcwd())
import torch
torch.manual_seed(0)

import kernels.packed_ternary.pack_update as pu
import kernels.packed_ternary.custom_ops as co

from train_gigatoken import N_LAYERS, K, VOCAB
from kernels.packed_ternary import PackedTernaryLinear

B, SEQ = 32, 512
THRESHOLD = 32


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


print("=== 1. Dispatch routing ===", flush=True)
co._ensure_loaded()
head_shape = (16384, 50272, 1024)
Wp = make_packed(*head_shape[1::])
dY = (torch.randn(*head_shape[:2], device="cuda") * 0.5).half()
out = co.backward_dx_tc(Wp, dY, head_shape[2])
# confirm which kernel the dispatch selected by comparing to direct 64x64 call
d64 = pu._dx_tc_fn(Wp, dY, head_shape[2]).float()
routed_64 = torch.equal(out.float(), d64)
print(f"  head dispatch -> {'64x64' if routed_64 else '32x32'} "
      f"(max|out-d64| = {(out.float()-d64).abs().max().item():.2e})", flush=True)

print("\n=== 2. Bit-exact: 64 vs 32 vs torch ref ===", flush=True)
shapes = [(16384, 50272, 1024), (16384, 4096, 1024), (128, 50000, 256)]
all_bit_exact = True
for (Bb, Nr, Kk) in shapes:
    Wp = make_packed(Nr, Kk)
    Wt = decode_ternary(Wp, Nr, Kk)
    dY = (torch.randn(Bb, Nr, device="cuda") * 0.5).half()
    ref = dY.float() @ Wt
    d64 = pu._dx_tc_fn(Wp, dY, Kk).float()
    d32 = pu._dx_tc_32_fn(Wp, dY, Kk).float()
    same = torch.equal(d64.float(), d32.float())
    err64 = (d64 - ref).abs().max().item()
    all_bit_exact &= same
    tag = "BIT-EXACT" if same else f"DIFF max={(d64-d32).abs().max().item():.2e}"
    print(f"  B={Bb:6d} N={Nr:6d} K={Kk:4d}: err64={err64:.4f}  64vs32={tag}", flush=True)

print("\n=== 3. Full-backward A/B: head->64 vs head->32 ===", flush=True)


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

for _ in range(3):
    model.zero_grad(set_to_none=True)
    logits = model(x); loss = loss_fn(logits.view(-1, VOCAB), y.view(-1)); loss.backward()
torch.cuda.synchronize()


def bench_bwd(label, head64):
    # head64=False -> force head to 32x32 by dropping _dx_tc_64 entirely
    saved = co._dx_tc_64
    if not head64:
        co._dx_tc_64 = None
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
    print(f"  {label}: bwd {avg:.1f} ms/step (min {min(ts):.1f})", flush=True)
    return avg


a64 = bench_bwd("head->64 (new)  ", head64=True)
a32 = bench_bwd("head->32 (old)  ", head64=False)
print(f"\n  full-backward speedup: {a32/a64:.3f}x  saving {a32-a64:.1f} ms/step", flush=True)
print(f"  {'ALL BIT-EXACT' if all_bit_exact else 'NON-EXACT PRESENT'}", flush=True)