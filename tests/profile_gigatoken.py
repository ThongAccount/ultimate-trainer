"""Profile the REAL gigatoken model backward — per-layer, per-kernel breakdown.

Runs 3 steps with torch.profiler, reports the top CUDA kernels by time so we
can see exactly where the ~5.2s backward goes (bwd_dx_tc vs update_tc_v2 vs
the fused backward_update kernel, plus any copies/dtype conversions).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import torch

from train_gigatoken import N_LAYERS, K, VOCAB
from kernels.packed_ternary import PackedTernaryLinear

B = 32
SEQ = 512
THRESHOLD = 32

class TernaryTransformer(torch.nn.Module):
    """Mirror of train_gigatoken's model (defined inside main() there)."""
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
model = TernaryTransformer().cuda()
model.train()

x = torch.randint(0, VOCAB, (B, SEQ), device="cuda")
loss_fn = torch.nn.CrossEntropyLoss()

# warmup + JIT compile kernels
for _ in range(2):
    logits = model(x)
    loss = loss_fn(logits.view(-1, VOCAB), x.view(-1))
    loss.backward()
torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=False,
) as prof:
    logits = model(x)
    loss = loss_fn(logits.view(-1, VOCAB), x.view(-1))
    loss.backward()
    torch.cuda.synchronize()

print("\n=== TOP CUDA KERNELS (by self time) ===", flush=True)
evts = prof.key_averages()
evts.sort(key=lambda e: e.self_device_time_total, reverse=True)
for e in evts[:20]:
    ms = e.self_device_time_total / 1000.0
    n = e.count
    print(f"  {e.key[:70]:<72} {ms:9.1f}ms  x{n}", flush=True)

# Group by prefix
print("\n=== GROUPED BY KERNEL PREFIX ===", flush=True)
groups = {}
for e in evts:
    if e.self_device_time_total <= 0:
        continue
    # group by first token after namespace
    key = e.key
    parts = key.split("::")
    g = parts[0][:60] if len(parts) > 1 else key[:60]
    groups.setdefault(g, [0, 0])
    groups[g][0] += e.self_device_time_total / 1000.0
    groups[g][1] += e.count
for g, (ms, n) in sorted(groups.items(), key=lambda kv: -kv[1][0])[:20]:
    print(f"  {g:<72} {ms:9.1f}ms  x{n}", flush=True)
