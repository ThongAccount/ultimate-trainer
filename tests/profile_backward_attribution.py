"""Backward-pass time attribution: WHERE does the 3.3s backward go?

Measures, for the full backward() and for each category:
  - GPU kernel self time (self_device_time)
  - CPU time per op (self_cpu_time) — catches Python/dispatch/custom_op overhead
  - op count

Goal: find the REAL cause of the backward slowness — is it GPU-bound
(kernels) or CPU-bound (autograd dispatch / custom_op Python overhead)?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import torch

from train_gigatoken import N_LAYERS, K, VOCAB
from kernels.packed_ternary import PackedTernaryLinear

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

for _ in range(3):
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = loss_fn(logits.view(-1, VOCAB), y.view(-1))
    loss.backward()
torch.cuda.synchronize()

# ── time forward vs backward wall separately ──
import time
torch.cuda.synchronize()
t0 = time.perf_counter()
logits = model(x); loss = loss_fn(logits.view(-1, VOCAB), y.view(-1))
torch.cuda.synchronize()
t1 = time.perf_counter()
loss.backward()
torch.cuda.synchronize()
t2 = time.perf_counter()
print(f"fwd wall = {(t1-t0)*1000:.0f} ms,  bwd wall = {(t2-t1)*1000:.0f} ms", flush=True)

# ── full profile of ONE step (fwd+bwd) with CPU + CUDA ──
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
    record_shapes=False, with_stack=False,
) as prof:
    logits = model(x)
    loss = loss_fn(logits.view(-1, VOCAB), y.view(-1))
    loss.backward()
    torch.cuda.synchronize()

evts = prof.key_averages()
evts.sort(key=lambda e: e.self_device_time_total, reverse=True)

# Split fwd vs bwd by op name heuristic
def is_bwd(name):
    return ("backward" in name or "Backward" in name or "Grad" in name or
            "autograd" in name or "engine" in name or "update" in name or
            "accumulate" in name or "_bwd" in name)

gpu_fwd = gpu_bwd = cpu_fwd = cpu_bwd = 0.0
cnt_fwd = cnt_bwd = 0
for e in evts:
    dt = e.self_device_time_total / 1000.0
    ct = e.self_cpu_time_total / 1000.0
    if is_bwd(e.key):
        gpu_bwd += dt; cpu_bwd += ct; cnt_bwd += e.count
    else:
        gpu_fwd += dt; cpu_fwd += ct; cnt_fwd += e.count

print(f"\nGPU total: fwd={gpu_fwd:.0f}ms  bwd={gpu_bwd:.0f}ms")
print(f"CPU total: fwd={cpu_fwd:.0f}ms  bwd={cpu_bwd:.0f}ms")
print(f"op count:  fwd={cnt_fwd}  bwd={cnt_bwd}")

# ── Top CPU-time ops (dispatch overhead candidates) ──
print("\n=== TOP 25 CPU-TIME OPS (Python/dispatch overhead) ===", flush=True)
evts_cpu = sorted(evts, key=lambda e: e.self_cpu_time_total, reverse=True)
for e in evts_cpu[:25]:
    if e.self_cpu_time_total <= 0: continue
    print(f"  {e.key[:75]:<75} cpu={e.self_cpu_time_total/1000:8.1f}ms  dev={e.self_device_time_total/1000:8.1f}ms  x{e.count}", flush=True)

# ── Top GPU-time ops ──
print("\n=== TOP 25 GPU-TIME OPS ===", flush=True)
for e in evts[:25]:
    if e.self_device_time_total <= 0: continue
    print(f"  {e.key[:75]:<75} dev={e.self_device_time_total/1000:8.1f}ms  cpu={e.self_cpu_time_total/1000:8.1f}ms  x{e.count}", flush=True)