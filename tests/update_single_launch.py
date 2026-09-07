"""Single-launch head update for ncu profiling."""
import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernels.packed_ternary import custom_ops as co

BATCH = 16384
inn, out = 1024, 50272
X = torch.randn(BATCH, inn, device="cuda", dtype=torch.float16)
dY = torch.randn(BATCH, out, device="cuda", dtype=torch.float16) * 0.5
W = torch.zeros(out, (inn + 15) // 16, device="cuda", dtype=torch.int32)
counter = torch.zeros(out, inn, device="cuda", dtype=torch.int16)

co._ensure_loaded()
# single launch for ncu
co.update_tc_v2(W, counter, X, dY, 32)
torch.cuda.synchronize()
print("done")