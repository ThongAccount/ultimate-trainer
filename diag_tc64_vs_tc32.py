#!/usr/bin/env python
"""Elementwise tc64 vs tc32 comparison — localize the tc64 3x bug."""
import torch

from kernels.packed_ternary.pack_forward import (
    packed_ternary_forward_tc_64, packed_ternary_forward_tc, has_forward_kernel,
    packed_ternary_forward,
)

torch.manual_seed(0)

for (B, N, K) in [(64, 64, 64), (64, 64, 128), (64, 128, 64), (128, 64, 64),
                  (64, 64, 16), (64, 64, 32), (64, 96, 96), (64, 128, 256)]:
    W = torch.randint(-2, 3, (N, K), dtype=torch.int32, device="cuda")
    X = (torch.randn(B, K, device="cuda") * 0.5).half()
    y64 = packed_ternary_forward_tc_64(W, X).float()
    y32 = packed_ternary_forward_tc(W, X).float()
    d = (y32 - y64).abs()
    # Also compare against a torch reference using pack/unpack semantics
    # decode: {-1,0,1} per 2-bit; W int32 stores 16 ternary/word little-endian
    Wd = W.to(torch.int64)
    tern = torch.zeros(N, K, dtype=torch.int64, device="cuda")
    for i in range(16):
        val = (Wd >> (2 * i)) & 3
        tern[:, i::16] = torch.where(val == 1, torch.tensor(1),
                           torch.where(val == 2, torch.tensor(-1),
                                        torch.tensor(0)))
    ref = (X.float() @ tern.float().T)
    e32 = (ref - y32).abs().max().item()
    e64 = (ref - y64).abs().max().item()
    print(f"B={B:4d} N={N:4d} K={K:4d}  max|y32|={y32.abs().max().item():7.2f} "
          f"max|y64|={y64.abs().max().item():7.2f} max|ref|={ref.abs().max().item():7.2f} "
          f" err32={e32:.4f} err64={e64:.4f}")
    if e64 > 1e-1 and y32.abs().max().item() > 1e-6:
        bad = (y32 - y64).abs() / (y32.abs() + 1e-6)
        print("   worst rel-diff idx:", bad.argmax().item())
        nz = (bad > 0.05).sum().item()
        print(f"   elems >5% rel diff: {nz}/{bad.numel()}")
        # where
        b_idx = torch.nonzero(y32 != y64, as_tuple=False)[:8]
        for bi, nj, ki_ in b_idx:
            print(f"    y32[{bi},{nj}]={y32[bi,nj].item():.3f} y64={y64[bi,nj].item():.3f}")