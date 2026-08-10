#!/usr/bin/env python
"""Per-stage timing: forward / backward_dx / update — one layer, steady state.

Run on GPU host. Splits the train step into its three kernel stages and reports
median wall per stage, so we can see which stage dominates for a given shape.
"""
import sys, os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kernels.packed_ternary.custom_ops import (
    forward_tc as _fwd,
    backward_dx_tc as _bwd,
    update_tc_v2 as _upd,
)
from kernels.packed_ternary.packed_linear import PackedTernaryLinear

torch.manual_seed(0)

SHAPES = [
    (16, 1024, 1024),
    (32, 1024, 1024),
    (16, 4096, 4096),
    (32, 4096, 4096),
]
WARMUP, ITERS = 5, 20


def bench_stage(name, fn, n_iters=ITERS):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


for (B, K, N) in SHAPES:
    print(f"== shape B={B} K={K} N={N}")
    lin = PackedTernaryLinear(K, N, threshold=32).cuda()
    W = lin.W_packed
    ctr = lin.counter
    X = torch.randn(B, K, device="cuda", dtype=torch.float16)
    Y = torch.randn(B, N, device="cuda", dtype=torch.float16)

    t_f = bench_stage("fwd", lambda: _fwd(W, X, K))
    t_b = bench_stage("bwd_dx", lambda: _bwd(W, Y, K))
    t_u = bench_stage("update", lambda: _upd(W, ctr, X, Y, 32))
    tot = t_f + t_b + t_u
    print(f"  fwd={t_f:7.3f} bwd_dx={t_b:7.3f} update={t_u:7.3f} total={tot:7.3f} ms")
print("done")