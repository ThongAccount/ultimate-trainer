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
    # Build a REAL packed W: [N, ceil(K/16)] uint32 words, 16 ternary codes each.
    nw = (K + 15) // 16
    codes = torch.randint(0, 3, (N, K), device="cuda")  # 0,1,2 = 0,+1,-1
    Wp = torch.zeros(N, nw, dtype=torch.int32, device="cuda")
    for k in range(K):
        Wp[:, k // 16] |= (codes[:, k] << (2 * (k % 16)))
    Wp = Wp.contiguous()
    X = (torch.randn(B, K, device="cuda") * 0.5).half()
    y64 = packed_ternary_forward_tc_64(Wp, X).float()
    y32 = packed_ternary_forward_tc(Wp, X).float()
    d = (y32 - y64).abs()
    # torch reference: unpack words
    tern = torch.zeros(N, K, dtype=torch.int64, device="cuda")
    for j in range(nw):
        for i in range(16):
            if j * 16 + i < K:
                v = (Wp[:, j] >> (2 * i)) & 3
                tern[:, j * 16 + i] = torch.where(v == 1, torch.tensor(1, device="cuda"),
                               torch.where(v == 2, torch.tensor(-1, device="cuda"),
                                            torch.tensor(0, device="cuda")))
    ref = (X.float() @ tern.float().T)
    e32 = (ref - y32).abs().max().item()
    e64 = (ref - y64).abs().max().item()
    print(f"B={B:4d} N={N:4d} K={K:4d}  max|y32|={y32.abs().max().item():7.2f} "
          f"max|y64|={y64.abs().max().item():7.2f} max|ref|={ref.abs().max().item():7.2f} "
          f" err32={e32:.4f} err64={e64:.4f}")
    if torch.isnan(y64).any():
        nz = torch.nonzero(torch.isnan(y64), as_tuple=False)
        print(f"   NaN count={nz.size(0)}/{y64.numel()}  first 12: "
              + " ".join(f"({int(b)},{int(n)})" for b, n in nz[:12].tolist()))
        # tile-corner histogram: which 64x64 CTA / warp region
        bq = (nz[:, 0] // 16) * 4 + (nz[:, 1] // 16)  # 16x16 frag idx
        import collections
        hist = collections.Counter(bq.tolist())
        print("   frag-idx histogram:", dict(sorted(hist.items())))
    if e64 > 1e-1 and y32.abs().max().item() > 1e-6:
        bad = (y32 - y64).abs() / (y32.abs() + 1e-6)
        print("   worst rel-diff idx:", bad.argmax().item())
        nz = (bad > 0.05).sum().item()
        print(f"   elems >5% rel diff: {nz}/{bad.numel()}")
        # where
        b_idx = torch.nonzero(y32 != y64, as_tuple=False)[:8]
        for ri, ci_ in b_idx:
            print(f"    y32[{ri},{ci_}]={y32[ri,ci_].item():.3f} y64={y64[ri,ci_].item():.3f}")