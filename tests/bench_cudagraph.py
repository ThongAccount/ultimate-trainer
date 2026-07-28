"""Benchmark: autograd vs manual vs CUDAGraph for discrete train step."""
import sys, os, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from kernels.packed_ternary import PackedTernaryLinear

B, K, N = 32, 4096, 4096
WARMUP, ITERS = 10, 100

layer = PackedTernaryLinear(K, N, threshold=8).cuda()
layer.train()

X = torch.randn(B, K, dtype=torch.float16, device="cuda")

# ── 0. Verify tensors exist ──
print(f"W_packed: {layer.W_packed.shape} {layer.W_packed.dtype}")
print(f"counter:  {layer.counter.shape} {layer.counter.dtype}")

# ── 1. Full autograd baseline ──
def autograd_step():
    y = layer(X)
    loss = y.mean()
    loss.backward()

for _ in range(WARMUP):
    autograd_step()
torch.cuda.synchronize()

e0 = torch.cuda.Event(enable_timing=True)
e1 = torch.cuda.Event(enable_timing=True)
times_ag = []
for _ in range(ITERS):
    e0.record()
    autograd_step()
    e1.record()
    torch.cuda.synchronize()
    times_ag.append(e0.elapsed_time(e1))
ag_ms = sorted(times_ag)[len(times_ag)//2]

# ── 2. Manual (direct kernel calls, no autograd) ──
from kernels.packed_ternary.custom_ops import forward_tc, backward_dx_tc, update_tc_v2

def manual_step():
    Y = forward_tc(layer.W_packed, X, K)
    dY = 2.0 * (Y - Y.detach()) / (B * N)  # MSE grad for mean()
    dX = backward_dx_tc(layer.W_packed, dY, K)
    update_tc_v2(layer.W_packed, layer.counter, X, dY, 8)
    return Y

for _ in range(WARMUP):
    manual_step()
torch.cuda.synchronize()

times_manual = []
for _ in range(ITERS):
    e0.record()
    manual_step()
    e1.record()
    torch.cuda.synchronize()
    times_manual.append(e0.elapsed_time(e1))
man_ms = sorted(times_manual)[len(times_manual)//2]

# ── 3. CUDAGraph (forward+backward captured, update outside) ──
from train_step_graph_v2 import TrainStepGraphCUDAGraph

layer2 = PackedTernaryLinear(K, N, threshold=8).cuda()
layer2.train()
layer2.W_packed.data.copy_(layer.W_packed)
layer2.counter.data.copy_(layer.counter)

graph = TrainStepGraphCUDAGraph(layer2, B, K, N, use_graph=True, threshold=8)

for _ in range(WARMUP):
    graph.step(X, X)  # dummy target
torch.cuda.synchronize()

times_graph = []
for _ in range(ITERS):
    e0.record()
    graph.step(X, X)
    e1.record()
    torch.cuda.synchronize()
    times_graph.append(e0.elapsed_time(e1))
gr_ms = sorted(times_graph)[len(times_graph)//2]

# ── Report ──
print(f"\n{'='*60}")
print(f"  Benchmark: {B}×{K}×{N}, {ITERS} iters")
print(f"{'='*60}")
print(f"  {'Method':<20} {'ms/step':>10} {'tok/s':>10} {'vs autograd':>12}")
print(f"  {'─'*54}")
ag_tok = B * K / (ag_ms / 1000)
man_tok = B * K / (man_ms / 1000)
gr_tok = B * K / (gr_ms / 1000)
print(f"  {'Autograd':<20} {ag_ms:>10.3f} {ag_tok:>10.0f} {'1.00×':>12}")
print(f"  {'Manual (direct)':<20} {man_ms:>10.3f} {man_tok:>10.0f} {ag_ms/man_ms:>11.2f}×")
print(f"  {'CUDAGraph':<20} {gr_ms:>10.3f} {gr_tok:>10.0f} {ag_ms/gr_ms:>11.2f}×")
print(f"{'='*60}")
