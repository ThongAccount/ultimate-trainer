"""COMPREHENSIVE profile of the gigatoken training step.

Unlike profile_gigatoken.py (top-20 CUDA only), this captures:
  - FULL kernel list grouped by name (every kernel, not just top-20)
  - CPU-side op timing alongside CUDA
  - dtype conversion / copy / LayerNorm / GELU / softmax / cross-entropy
  - per-phase wall clock (layers / head / loss / backward)

Goal: resolve the ~2s/step gap between wall time (~6.5s) and summed
GEMM kernel time (~4.5s). Find where the unaccounted time actually goes.
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
y = x.clone()  # cross-entropy target is x[:, 1:]
loss_fn = torch.nn.CrossEntropyLoss()

# warmup + JIT compile kernels
for _ in range(2):
    logits = model(x)
    loss = loss_fn(logits.view(-1, VOCAB), x.view(-1))
    loss.backward()
    model.zero_grad(set_to_none=True)
torch.cuda.synchronize()

# ── Full profile: CPU + CUDA activities, everything ──
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.CPU],
    record_shapes=False,
    with_stack=False,
) as prof:
    logits = model(x)
    loss = loss_fn(logits.view(-1, VOCAB), x.view(-1))
    loss.backward()
    torch.cuda.synchronize()

print("\n=== ALL CUDA KERNELS (by self time, grouped) ===", flush=True)
evts = prof.key_averages()
evts.sort(key=lambda e: e.self_device_time_total, reverse=True)

# Group by a coarse category
CATEGORIES = {
    "fwd_gemm": ["forward_tc_64", "packed_ternary_tc_kernel", "forward_tc", "forward_v"],
    "bwd_dx":   ["backward_dx_tc", "backward_dx"],
    "update":   ["update_tc_v2", "update_tc", "update_v"],
    "softmax_xent": ["softmax", "cross_entropy", "nll_loss", "log_softmax", "cudnn"],
    "layernorm": ["layer_norm", "native_layer_norm", "vectorized_layer_norm"],
    "gelu":     ["gelu"],
    "embedding":["embedding", "embed"],
    "cast_copy": ["cast", "copy", "contiguous", "as_strided", "to", "dtype"],
    "elementwise": ["mul", "add", "fill", "zero", "clamp", "relu"],
}

cat_ms = {}
cat_cnt = {}
def categorize(name):
    for c, keys in CATEGORIES.items():
        for k in keys:
            if k in name:
                return c
    return "other"

# Enrich: name every kernel not caught, instead of lumping into "other"
CATEGORIES["elementwise_scale"] = ["vectorized_elementwise", "elementwise", "mul_kernel", "Mul"]
CATEGORIES["reduction"] = ["reduce", "sum", "mean", "argmax", "max_pool", "min_pool"]
CATEGORIES["embedding_bwd"] = ["embedding_backward", "embedding_dense_backward"]

for e in evts:
    if e.self_device_time_total <= 0:
        continue
    c = categorize(e.key)
    cat_ms[c] = cat_ms.get(c, 0.0) + e.self_device_time_total / 1000.0
    cat_cnt[c] = cat_cnt.get(c, 0) + e.count
    if c == "other":
        print(f"  [UNMATCHED] {e.key[:90]}  {e.self_device_time_total/1000.0:.1f}ms x{e.count}", flush=True)

total_gpu = sum(cat_ms.values())
print(f"{'CATEGORY':<16} {'ms':>10} {'calls':>7}  {'%':>6}")
print("-" * 46)
for c, ms in sorted(cat_ms.items(), key=lambda kv: -kv[1]):
    pct = 100.0 * ms / total_gpu if total_gpu else 0
    print(f"{c:<16} {ms:10.1f} {cat_cnt[c]:7d}  {pct:5.1f}%")
print("-" * 46)
print(f"{'TOTAL':<16} {total_gpu:10.1f}")

print("\n=== TOP 25 INDIVIDUAL KERNELS ===", flush=True)
for e in evts[:25]:
    if e.self_device_time_total <= 0:
        continue
    ms = e.self_device_time_total / 1000.0
    n = e.count
    print(f"  {e.key[:80]:<82} {ms:9.1f}ms  x{n}", flush=True)

# ── CPU-side breakdown ──
print("\n=== CPU-side (dispatch) top ops ===", flush=True)
cpu_evts = [e for e in evts if e.self_cpu_time_total > 0]
cpu_evts.sort(key=lambda e: e.self_cpu_time_total, reverse=True)
for e in cpu_evts[:15]:
    ms = e.self_cpu_time_total / 1000.0
    print(f"  {e.key[:80]:<82} {ms:9.1f}ms  x{e.count}", flush=True)

# ── Wall-clock per phase (mirrors train_gigatoken [TIME]) ──
print("\n=== WALL-CLOCK PHASES (5 steps) ===", flush=True)
phases = {"layers": 0.0, "head": 0.0, "loss": 0.0, "bwd": 0.0}
for it in range(5):
    e = [torch.cuda.Event(enable_timing=True) for _ in range(8)]
    e[0].record()
    h = model.embed(x).half().view(B * SEQ, K)
    for i in range(N_LAYERS):
        h = model.norms[i](h.float()).half()
        h = model.fc1s[i](h) * (K ** -0.5)
        h = torch.nn.functional.gelu(h)
        h = model.fc2s[i](h) * (K ** -0.5)
    e[1].record()
    h = model.head(h) * (K ** -0.5)
    e[2].record()
    logits = h.view(B, SEQ, VOCAB)
    loss = loss_fn(logits.view(-1, VOCAB), x.view(-1))
    e[3].record()
    loss.backward()
    model.zero_grad(set_to_none=True)
    e[4].record()
    torch.cuda.synchronize()
    phases["layers"] += e[0].elapsed_time(e[1])
    phases["head"] += e[1].elapsed_time(e[2])
    phases["loss"] += e[2].elapsed_time(e[3])
    phases["bwd"] += e[3].elapsed_time(e[4])

for k in phases:
    phases[k] /= 5.0
total_wall = sum(phases.values())
for k, ms in phases.items():
    print(f"  {k:<10} {ms:8.1f}ms  {100*ms/total_wall:5.1f}%")
print(f"  {'TOTAL':<10} {total_wall:8.1f}ms")

print("\nDONE", flush=True)