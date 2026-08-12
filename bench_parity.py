"""Parity + perf: median ms for fwd/bwd/update on GPU.

Imports whichever impl exists in the current branch (torch_impl or triton_impl)
plus the CUDA ground-truth reference (ref_linear / unpack_tensor) for parity.
"""
import sys, os, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernels.packed_ternary import pack_tensor, unpack_tensor
from kernels.packed_ternary.pack_forward import ref_linear

torch.manual_seed(0)

SHAPES = [(64, 64, 64), (32, 128, 128), (128, 96, 96), (17, 33, 65),
          (64, 256, 256), (256, 512, 512)]
WARMUP, ITERS = 5, 20


def med(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def pick():
    try:
        from kernels.packed_ternary import torch_impl
        return torch_impl, "torch_impl"
    except Exception:
        pass
    from kernels.packed_ternary import triton_impl
    return triton_impl, "triton_impl"


impl, name = pick()
print(f"impl: {name}")
for (B, N, K) in SHAPES:
    Wf = torch.randn(N, K) * 0.1
    Wp = pack_tensor(Wf, gamma=1.0).cuda()
    X = torch.randn(B, K, device="cuda").half()
    dY = torch.randn(B, N, device="cuda").half()
    ctr = torch.zeros(N, K, dtype=torch.int16, device="cuda")

    # parity
    y_ref = ref_linear(Wp.cpu(), X.cpu()).cuda()
    y = impl.ternary_forward(Wp, X, K)
    e_f = (y.float() - y_ref.float()).abs().max().item()
    tern = unpack_tensor(Wp.cpu(), N, K).half().cuda()
    dx_ref = (dY.float() @ tern.float()).half()
    e_b = (impl.ternary_backward_dx(Wp, dY, K).float() - dx_ref.float()).abs().max().item()

    t_f = med(lambda: impl.ternary_forward(Wp, X, K))
    t_b = med(lambda: impl.ternary_backward_dx(Wp, dY, K))
    t_u = med(lambda: impl.ternary_update(Wp.clone(), ctr, X, dY, 32, K))

    print(f"B={B:3d} N={N:4d} K={K:4d}  err_f={e_f:.1e} err_b={e_b:.1e}  "
          f"fwd={t_f:7.3f} bwd={t_b:7.3f} upd={t_u:7.3f} ms")
print("done")
