"""Sparse kernel debug — print exactly where kernel diverges from ref."""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from kernels.block_sparse_ternary.block_sparse_ternary import (
    block_sparse_ternary_matmul, compute_block_mask, _HAS_CUDA,
)

print(f"_HAS_CUDA = {_HAS_CUDA}")

torch.manual_seed(0)
DO = 64
opw = torch.randn(DO, DO, device="cuda") * 0.05
gamma = 0.1

BN = 16
num_n = (DO + BN - 1) // BN
num_k = (DO + 15) // 16
mask = compute_block_mask(torch.tensor([[0, 1]], device="cuda"), 2, BN, num_n, num_k)
print(f"mask = {mask.item():#018x}")
print(f"num_n={num_n} num_k={num_k}")

xm = torch.randn(128, DO, device="cuda")
y = block_sparse_ternary_matmul(xm, opw, gamma, mask)
torch.cuda.synchronize()

wq = torch.clamp(torch.round(opw / gamma), -1, 1) * gamma
ref = xm @ wq.t()
for tn in range(num_n):
    for tk in range(num_k):
        bit = tn * num_k + tk
        if not (mask[bit // 64] & (1 << (bit % 64))):
            ref[:, tn * BN:(tn + 1) * BN] = 0.0

diff = (y - ref).abs()
print(f"max err = {diff.max().item():.6f}")
print(f"y[0,:8]   = {y[0,:8].tolist()}")
print(f"ref[0,:8] = {ref[0,:8].tolist()}")
print(f"y[0,32:40]   = {y[0,32:40].tolist()}")
print(f"ref[0,32:40] = {ref[0,32:40].tolist()}")

# find which element has max diff
flat_idx = diff.argmax()
r, c = flat_idx // DO, flat_idx % DO
print(f"max diff at row={r}, col={c}")
print(f"  y[{r},{c}]   = {y[r,c].item():.6f}")
print(f"  ref[{r},{c}] = {ref[r,c].item():.6f}")

# check a non-masked column
print(f"\nColumn 0: y={y[:3,0].tolist()} ref={ref[:3,0].tolist()}")
print(f"Column 1: y={y[:3,1].tolist()} ref={ref[:3,1].tolist()}")
print(f"Column 16: y={y[:3,16].tolist()} ref={ref[:3,16].tolist()}")
print(f"Column 32 (should be 0): y={y[:3,32].tolist()} ref={ref[:3,32].tolist()}")

# check wq for col 0
print(f"\nQ(opw[0,:5]/gamma) = {torch.clamp(torch.round(opw[0,:5]/gamma),-1,1).tolist()}")
print(f"opw[0,:5] = {opw[0,:5].tolist()}")
print(f"opw[0,:5]/gamma = {(opw[0,:5]/gamma).tolist()}")
