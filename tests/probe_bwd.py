"""Check bwd_dx_tc against a TRUE torch reference (not tc32)."""
import os, sys
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.path.insert(0, os.getcwd())
import torch
from kernels.packed_ternary import PackedTernaryLinear
from kernels.packed_ternary.pack_update import backward_dx, backward_update

torch.manual_seed(0)

N, K = 128, 128
layer = PackedTernaryLinear(K, N, threshold=32).cuda()
W_packed = layer.W_packed  # (N, stride_words) uint32, 2-bit codes

# Decode W_packed to ternary {-1,0,1}
stride = W_packed.shape[1]
n_words = stride * 16  # 16 weights per word? check kWeightsPerWord
kWeightsPerWord = 16  # from packed_ternary.cuh
W_ter = torch.zeros(N, K, device="cuda", dtype=torch.float32)
for wi in range(stride):
    word = W_packed[:, wi]  # (N,) uint32
    for b in range(kWeightsPerWord):
        code = (word >> (2 * b)) & 3
        val = torch.where(code == 1, 1.0, torch.where(code == 2, -1.0, 0.0))
        k = wi * kWeightsPerWord + b
        if k < K:
            W_ter[:, k] = val
# note: code 1 = +1, 2 = -1 (verify with decode_ternary semantics)

B = 64
dY = torch.randn(B, N, device="cuda", dtype=torch.float16)
X = torch.randn(B, K, device="cuda", dtype=torch.float16)

ref = dY.float() @ W_ter  # (B, K)

dx = backward_dx(W_packed, dY, K)
err = (dx.float() - ref).abs().max().item()
print(f"bwd_dx  err vs torch: {err:.6f}  (ref mag {ref.abs().max().item():.3f})")
