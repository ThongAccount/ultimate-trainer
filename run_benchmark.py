import sys, os
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath('tests/test_gemm_perf.py')))
from kernels.packed_ternary.pack_forward import packed_ternary_forward_tc, packed_ternary_forward_packed, has_tc, has_packed
from tests.test_gemm_perf import benchmark_shape, SHAPES

if not has_packed() or not has_tc():
    print("Missing kernels")
    sys.exit(1)

print(f"{'ver':>6} {'batch':>5} {'in_f':>6} {'out_f':>6} {'median(ms)':>9} {'GFLOPS':>8}")
for batch, in_f, out_f in SHAPES:
    for ver, fn in [("tc", packed_ternary_forward_tc), ("packed", packed_ternary_forward_packed)]:
        s = benchmark_shape(fn, ver, batch, in_f, out_f)
        print(f"{ver:>6} {s['batch']:>5} {s['in_features']:>6} {s['out_features']:>6} {s['median_ms']:>9.3f} {s['gflops']:>8.1f}")
