"""Isolate: custom op overhead vs raw CUDA call."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from kernels.packed_ternary.custom_ops import _ensure_loaded, forward_tc
_ensure_loaded()

# Raw C extension function (bypass custom_op)
from kernels.packed_ternary.pack_forward import _load_tc
_load_tc()
from kernels.packed_ternary.pack_forward import _forward_fn_tc

B, K, N = 16384, 1024, 4096
# Head dimensions: 16384, 1024, 50272
# But let's use 4096 for head-like test first

X = torch.randn(B, K, dtype=torch.float16, device='cuda')
W = torch.randint(0, 1<<30, (N, (K+15)//16), dtype=torch.int32, device='cuda')

# Warmup
for _ in range(5):
    _forward_fn_tc(W, X)
    forward_tc(W, X, K)
torch.cuda.synchronize()

# Benchmark raw C extension
e0 = torch.cuda.Event(enable_timing=True)
e1 = torch.cuda.Event(enable_timing=True)
e0.record()
for _ in range(50):
    _forward_fn_tc(W, X)
e1.record()
torch.cuda.synchronize()
raw_ms = e0.elapsed_time(e1) / 50

# Benchmark custom op
e0.record()
for _ in range(50):
    forward_tc(W, X, K)
e1.record()
torch.cuda.synchronize()
op_ms = e0.elapsed_time(e1) / 50

print(f"  [TEST] Raw ext:  {raw_ms:.3f}ms/call")
print(f"  [TEST] Custom op: {op_ms:.3f}ms/call")
print(f"  [TEST] Ratio: {op_ms/raw_ms:.1f}×")

# Now test with head size
N_head = 50272
W_head = torch.randint(0, 1<<30, (N_head, (K+15)//16), dtype=torch.int32, device='cuda')
for _ in range(3):
    _forward_fn_tc(W_head, X)
    forward_tc(W_head, X, K)
torch.cuda.synchronize()

e0.record()
for _ in range(10):
    _forward_fn_tc(W_head, X)
e1.record()
torch.cuda.synchronize()
raw_h_ms = e0.elapsed_time(e1) / 10

e0.record()
for _ in range(10):
    forward_tc(W_head, X, K)
e1.record()
torch.cuda.synchronize()
op_h_ms = e0.elapsed_time(e1) / 10
print(f"  [HEAD] Raw ext:  {raw_h_ms:.3f}ms/call")
print(f"  [HEAD] Custom op: {op_h_ms:.3f}ms/call")
print(f"  [HEAD] Ratio: {op_h_ms/raw_h_ms:.1f}×")
