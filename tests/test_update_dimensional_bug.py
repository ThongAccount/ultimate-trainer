"""Test update_tc_v2 correctness against torch reference.

This test specifically checks if the dimensional bug in update_tc_v2 causes
numerical errors by comparing against a ground-truth implementation.
"""
import os
import sys
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.path.insert(0, os.getcwd())

import torch
torch.manual_seed(0)

# Test dimensions: Must trigger the bug (N, K >= 64)
B, N, K = 32, 128, 128
threshold = 32

print(f"Testing update_tc_v2 correctness: B={B}, N={N}, K={K}")

# Create test inputs
X = torch.randn(B, K, device="cuda", dtype=torch.float16)
dY = torch.randn(B, N, device="cuda", dtype=torch.float16)

# Compute reference gradient: dW = dY^T @ X
dW_ref = (dY.T.float() @ X.float()).half()  # [N, K]

# Initialize packed weight and counter
kWeightsPerWord = 16
stride_words = (K + kWeightsPerWord - 1) // kWeightsPerWord
W_packed = torch.zeros(N, stride_words, dtype=torch.int32, device="cuda")
counter_ref = torch.zeros(N, K, dtype=torch.int16, device="cuda")
counter_kernel = torch.zeros(N, K, dtype=torch.int16, device="cuda")

# Reference update: counter -= sign(dW)
for n in range(N):
    for k in range(K):
        g = dW_ref[n, k].item()
        if g > 0:
            counter_ref[n, k] = -1
        elif g < 0:
            counter_ref[n, k] = 1

# Kernel update
from kernels.packed_ternary.pack_update import _load_up_tc_v2_32
_load_up_tc_v2_32()

from kernels.packed_ternary import pack_update as pu
if not pu._HAS_UP_TC_V2_32:
    print("❌ TC32 update kernel not available")
    sys.exit(1)

W_packed_test = W_packed.clone()
pu._up_tc_v2_32_fn(W_packed_test, counter_kernel, X, dY, threshold)

# Compare
diff = (counter_kernel - counter_ref).abs()
max_diff = diff.max().item()
num_errors = (diff > 0).sum().item()
error_rate = num_errors / (N * K) * 100

print(f"\n{'='*60}")
print(f"UPDATE_TC_V2 CORRECTNESS TEST")
print(f"{'='*60}")
print(f"Dimensions: B={B}, N={N}, K={K}")
print(f"Reference counter changes: {(counter_ref != 0).sum().item()} / {N*K}")
print(f"Kernel counter changes:    {(counter_kernel != 0).sum().item()} / {N*K}")
print(f"Max difference: {max_diff}")
print(f"Error count: {num_errors} / {N*K} ({error_rate:.2f}%)")
print(f"{'='*60}")

if max_diff == 0:
    print("✅ PASS — Kernel matches reference exactly")
    sys.exit(0)
else:
    print(f"❌ FAIL — Kernel has {error_rate:.2f}% errors (dimensional bug confirmed)")
    
    # Show sample errors
    errors = torch.nonzero(diff > 0)[:10]
    print(f"\nSample errors (first 10):")
    for i in range(min(10, len(errors))):
        n, k = errors[i]
        print(f"  [{n:3d}, {k:3d}]: ref={counter_ref[n,k]:3d}, kernel={counter_kernel[n,k]:3d}, dW_ref={dW_ref[n,k]:.3f}")
    
    sys.exit(1)
