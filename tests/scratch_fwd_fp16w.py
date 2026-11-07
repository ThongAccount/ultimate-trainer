"""Scratch: bench packed-W fwd vs pre-decoded fp16-W fwd (same process, Modal T4).

Tests the W-decode hypothesis directly: if fp16-W is much faster than packed-W,
decode cost dominates forward; the split-decode design is warranted. If not,
forward is bound elsewhere (loads/sync) and Exp2 has no headroom either.
"""
import torch, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.cpp_extension import load_inline
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
cuh = (ROOT / "kernels/packed_ternary/packed_ternary.cuh").read_text()
cu  = (ROOT / "tests/scratch_fwd_fp16w.cu").read_text()
lib = load_inline(
    name="scratch_fwd_fp16w_ext",
    cpp_sources="""#include <cuda_runtime.h>
#include <torch/extension.h>
extern "C" void launch_forward_tc_64_fp16w(const void*, const void*, void*, int, int, int, cudaStream_t);
torch::Tensor fwd_fp16w(torch::Tensor W, torch::Tensor X) {
    auto Y = torch::empty({X.size(0), W.size(0)}, X.options());
    launch_forward_tc_64_fp16w(W.data_ptr(), X.data_ptr(), Y.data_ptr(),
                               X.size(0), X.size(1), W.size(0), nullptr);
    return Y;
}
""",
    cuda_sources=[cuh + "\n" + cu.replace('#include "packed_ternary.cuh"', "")],
    verbose=False,
    extra_cuda_cflags=["-O2", "--use_fast_math", f"-I{ROOT / 'kernels/packed_ternary'}"],
    functions=["fwd_fp16w"],
)

from kernels.packed_ternary import pack_forward as pf
from kernels.packed_ternary import pack_update as pu

BATCH, VOCAB = 16384, 50272

def bench(fn, iters=20, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

pf._load_tc_64()

print(f"{'shape':<26} {'packed-W ms':>11} {'fp16-W ms':>10} {'speedup':>8}", flush=True)
for name, inn, out in [("fc1", 1024, 4096), ("fc2", 4096, 1024), ("head", 1024, VOCAB)]:
    X = torch.randn(BATCH, inn, device="cuda", dtype=torch.float16)
    Wp = torch.zeros(out, (inn + 15) // 16, device="cuda", dtype=torch.int32)
    Wh = torch.zeros(out, inn, device="cuda", dtype=torch.float16)
    tp = bench(lambda: pf.packed_ternary_forward_tc_64(Wp, X))
    th = bench(lambda: lib.fwd_fp16w(Wh, X))
    print(f"  {name:>5} ({inn}x{out})  {tp:10.2f}  {th:9.2f}  {tp/th:7.2f}x", flush=True)