#!/usr/bin/env python
"""Diag: which packed-ternary forward kernel is producing NaNs on Shakespeare dims."""
import torch

from kernels.packed_ternary.pack_forward import (
    has_tc, has_tc_64, has_forward_kernel, has_forward_kernel_v2, has_packed,
    packed_ternary_forward_tc, packed_ternary_forward_tc_64,
    packed_ternary_forward_v2, packed_ternary_forward,
)
from kernels.packed_ternary.pack_update import _tc_ok

print("flags: tc32=%s tc64=%s v2=%s v1=%s packed=%s" % (
    has_tc(), has_tc_64(), has_forward_kernel_v2(), has_forward_kernel(), has_packed()))

torch.manual_seed(0)
# Shakespeare qkv style dims: B*T=4096, N=384, K=128, fp16
for (B, N, K) in [(4096, 384, 128), (4096, 256, 128), (64, 128, 128), (32, 384, 128), (4096, 128, 384)]:
    W = torch.randint(-2, 3, (N, K), dtype=torch.int32, device="cuda")
    X = torch.randn(B, K, dtype=torch.float16, device="cuda")
    ref = torch.zeros_like(X)
    # reference: scalar unpack in torch (W bits -> {-1,0,1})
    # pack layout: 2 ternary per int32? just use v1 as fallback ref below.
    print(f"\nB={B} N={N} K={K}  tc_ok={_tc_ok(B) and _tc_ok(N) and _tc_ok(K)}")
    if has_tc_64() and B >= 64 and N >= 64 and K >= 64:
        y = packed_ternary_forward_tc_64(W, X)
        print("  tc64:", "nan" if torch.isnan(y).any().item() else f"max|y|={y.abs().max().item():.2f}")
    if has_tc():
        y = packed_ternary_forward_tc(W, X)
        print("  tc32:", "nan" if torch.isnan(y).any().item() else f"max|y|={y.abs().max().item():.2f}")
    if has_forward_kernel_v2() and N >= 4:
        y = packed_ternary_forward_v2(W, X)
        print("  v2  :", "nan" if torch.isnan(y).any().item() else f"max|y|={y.abs().max().item():.2f}")
    if has_forward_kernel():
        y = packed_ternary_forward(W, X)
        print("  v1  :", "nan" if torch.isnan(y).any().item() else f"max|y|={y.abs().max().item():.2f}")