"""In-process paired A/B: forward TC64 rasterization swizzle (C13 GO #2).

OLD = production kernels/packed_ternary/gemm_forward_tc.cu
     grid=(m_tiles, n_tiles), blockIdx.x=m fastest -> co-resident CTAs span
     different m of the SAME n-tile: X tiles re-streamed from DRAM (25.7GB
     at the head shape) while W gets the L2 reuse.
NEW = tests/fwd_swizzle.cu — same launch config, but the CTA->tile map is
     swizzled so n varies FASTEST (formulation A: swizzled linear id with
     num_n_tiles kernel arg) or the launch itself is transposed
     (formulation B: grid=(n_tiles, m_tiles)). Co-resident CTAs then share
     one 128KB X tile and touch distinct packed-W slices (~1.9MB < 4MB L2).

Pure compute + disjoint output tiles -> parity must be BIT-EXACT
(torch.equal on the fp16 Y).

CPU-side coverage proof: both CTA->tile maps are asserted to be bijections
on every (m,n) tile for (M,N) in {(16384,50272),(16384,4096),(512,64)}.

Timing: tests.bench_protocol.ab_median (10-trial median, interleaved A/B,
null arm on the first shape) — the mandatory C13 protocol.

Usage: python tests/probe_fwd_swizzle.py
"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CUH = os.path.join(REPO, "kernels/packed_ternary/packed_ternary.cuh")
OLD_CU = os.path.join(REPO, "kernels/packed_ternary/gemm_forward_tc.cu")
NEW_CU = os.path.join(HERE, "fwd_swizzle.cu")

with open(CUH) as f:
    CUH_SRC = f.read()

# Production flags for tc_64 (kernels/packed_ternary/pack_forward.py:505)
CUDA_CFLAGS = ["-O3", "--use_fast_math"]


def build(name, cu_path, funcs):
    """Compile one load_inline module exposing the given launchers."""
    decls = "\n".join(
        f"""
        extern "C" void {fn}(
            const uint32_t* W, const void* X, void* Y,
            int batch_size, int in_features, int out_features,
            int stride_words, cudaStream_t stream);"""
        for fn in funcs
    )
    wraps = "\n".join(
        f"""
        torch::Tensor {fn}(torch::Tensor W, torch::Tensor X) {{
            auto Y = torch::empty({{X.size(0), W.size(0)}}, torch::dtype(torch::kFloat16).device(X.device()));
            {fn}(
                reinterpret_cast<const uint32_t*>(W.data_ptr<int32_t>()),
                X.data_ptr<at::Half>(), Y.data_ptr<at::Half>(),
                X.size(0), X.size(1), W.size(0), W.size(1), nullptr);
            return Y;
        }}"""
        for fn in funcs
    )
    binds = "\n".join(f'        m.def("{fn}", &{fn});' for fn in funcs)

    from torch.utils.cpp_extension import load_inline

    with open(cu_path) as f:
        cu = f.read()
    combined = CUH_SRC + "\n" + cu.replace('#include "packed_ternary.cuh"', "")

    lib = load_inline(
        name=name,
        cpp_sources=f"""
        #include <cuda_runtime.h>
        #include <torch/extension.h>
        extern "C" {{
        {decls}
        }}
        {wraps}
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
        {binds}
        }}
        """,
        cuda_sources=[combined],
        verbose=False,
        extra_cuda_cflags=CUDA_CFLAGS,
    )
    return lib


# ── Fixture (duplicated ~15 lines; probe_fwd_spill.py did not exist when
#    this probe was written — do not block on the sibling). ──────────────────

def make_fixture(dev, B, K, N, seed=0):
    """Packed W (valid ternary codes), fp16 X. Returns (W, X)."""
    torch.manual_seed(seed)
    X = torch.randn(B, K, device=dev, dtype=torch.float16)
    W = torch.randint(0, 4, (N, (K + 15) // 16), device=dev, dtype=torch.int32)
    W = torch.where(W == 3, 0, W)  # code 3 unused -> renormalize to 0
    return W, X


# ── CPU-side coverage proof ──────────────────────────────────────────────────

def check_coverage():
    """Assert both CTA->tile maps are bijections for the required shapes.

    Formulation A: linear = by*gridDim.x + bx over the UNCHANGED launch
                   grid=(m_tiles, n_tiles); the kernel then derives
                   m = linear // num_n_tiles, n = linear % num_n_tiles.
    Formulation B: transposed launch grid=(n_tiles, m_tiles); the kernel
                   reads n = blockIdx.x, m = blockIdx.y directly.
    """
    TM = TN = 64
    for M, N in [(16384, 50272), (16384, 4096), (512, 64)]:
        m_tiles = (M + TM - 1) // TM
        n_tiles = (N + TN - 1) // TN

        # Production map (bx=m fastest): every (bx,by) is a distinct CTA
        # computing a distinct tile — trivially bijective; kept as the anchor
        # that all three maps enumerate the same tile set.
        prod = {(bx, by) for bx in range(m_tiles) for by in range(n_tiles)}
        assert len(prod) == m_tiles * n_tiles

        # Formulation A (swizzled linear id, launch config unchanged).
        # Hardware linear raster order: linear = by*gridDim.x + bx.
        swz = set()
        for bx in range(m_tiles):          # grid.x = m_tiles (unchanged launch)
            for by in range(n_tiles):      # grid.y = n_tiles
                lin = by * m_tiles + bx    # hardware linear raster order
                m, n = divmod(lin, n_tiles)
                assert 0 <= m < m_tiles and 0 <= n < n_tiles, (M, N, lin, m, n)
                swz.add((m, n))
        assert len(swz) == m_tiles * n_tiles, (
            f"A coverage hole at (M={M}, N={N}): "
            f"{m_tiles*n_tiles - len(swz)} missing")

        # Formulation B (transposed launch grid=(n_tiles, m_tiles)).
        # Hardware rasterizes bx fastest, so consecutive linear CTAs share
        # m-tile (by) and span n-tiles (bx) — the desired n-fastest order.
        xp = {(by, bx) for bx in range(n_tiles) for by in range(m_tiles)}
        assert len(xp) == m_tiles * n_tiles, (
            f"B coverage hole at (M={M}, N={N})")
        assert xp == prod and swz == prod, (M, N)

        print(f"coverage ok: M={M:>6} N={N:>6} tiles {m_tiles}x{n_tiles} "
              f"(prod/A/B all bijective)")
    print("COVERAGE PROOF PASSED (all (m,n) tiles computed exactly once, "
          "both formulations)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    check_coverage()

    if not torch.cuda.is_available():
        print("no CUDA on this box — coverage proof only "
              "(parity/timing need a T4)")
        return

    dev = "cuda"
    from tests.bench_protocol import ab_median, report_line

    print("building OLD (prod) ...", flush=True)
    old = build("fwd_swizzle_old", OLD_CU,
                ["launch_packed_ternary_forward_tc_64"])
    print("building NEW (swizzle A + xpose B) ...", flush=True)
    new = build("fwd_swizzle_new", NEW_CU,
                ["launch_packed_ternary_forward_tc_64_swizzle",
                 "launch_packed_ternary_forward_tc_64_xpose"])
    print("built ok", flush=True)

    BATCH = 16384
    shapes = [
        ("head", 1024, 50272),
        ("fc2", 4096, 1024),
        ("fc1", 1024, 4096),
    ]

    # ── Bit-exact parity gate (pure compute, disjoint output tiles) ──
    print("\nPARITY (must be torch.equal — bit-exact):", flush=True)
    all_parity_ok = True
    for si, (name, K, N) in enumerate(shapes):
        W, X = make_fixture(dev, BATCH, K, N, seed=si)
        y_old = old.launch_packed_ternary_forward_tc_64(W, X)
        y_a = new.launch_packed_ternary_forward_tc_64_swizzle(W, X)
        y_b = new.launch_packed_ternary_forward_tc_64_xpose(W, X)
        ok_a = torch.equal(y_old, y_a)
        ok_b = torch.equal(y_old, y_b)
        all_parity_ok &= ok_a and ok_b
        print(f"  {name:<5} B={BATCH} K={K:>5} N={N:>6}  "
              f"swizzle={'BIT-EXACT' if ok_a else 'MISMATCH'}  "
              f"xpose={'BIT-EXACT' if ok_b else 'MISMATCH'}", flush=True)

    # Small/edge shapes too (tail tiles, single tile per dim)
    for ei, (B, K, N) in enumerate([(512, 64, 64), (100, 48, 80), (1, 16, 16)]):
        W, X = make_fixture(dev, B, K, N, seed=100 + ei)
        y_old = old.launch_packed_ternary_forward_tc_64(W, X)
        y_a = new.launch_packed_ternary_forward_tc_64_swizzle(W, X)
        y_b = new.launch_packed_ternary_forward_tc_64_xpose(W, X)
        ok_a = torch.equal(y_old, y_a)
        ok_b = torch.equal(y_old, y_b)
        all_parity_ok &= ok_a and ok_b
        print(f"  edge  B={B:>5} K={K:>3} N={N:>3}  "
              f"swizzle={'BIT-EXACT' if ok_a else 'MISMATCH'}  "
              f"xpose={'BIT-EXACT' if ok_b else 'MISMATCH'}", flush=True)

    if not all_parity_ok:
        print("\nPARITY FAILED — abort (tile map or store math broken)")
        sys.exit(1)
    print("\nPARITY PASSED", flush=True)

    # ── Timing: 10-trial median, interleaved A/B, null arm on first shape ──
    print("\nTIMING (ab_median: 10 trials, median, null arm on first shape):",
          flush=True)
    for i, (name, K, N) in enumerate(shapes):
        W, X = make_fixture(dev, BATCH, K, N, seed=200 + i)

        run_old = lambda: old.launch_packed_ternary_forward_tc_64(W, X)
        run_a = lambda: new.launch_packed_ternary_forward_tc_64_swizzle(W, X)
        run_b = lambda: new.launch_packed_ternary_forward_tc_64_xpose(W, X)

        res_a = ab_median(run_old, run_a, null_arm=(i == 0))
        print("  " + report_line(f"{name}/swz", res_a), flush=True)
        if res_a.sigma_null_pct is not None:
            print(f"    (null arm: sigma_null={res_a.sigma_null_pct:.2f}%, "
                  f"decision threshold="
                  f"{max(2*res_a.sigma_null_pct, 1.0):.2f}%)", flush=True)
        res_b = ab_median(run_old, run_b)
        print("  " + report_line(f"{name}/xps", res_b), flush=True)

    print("\nDone. Ship whichever formulation wins with identical ptxas "
          "resources (see report: swizzle adds an integer divide in the CTA "
          "prologue; xpose is resource-identical by construction).")


if __name__ == "__main__":
    main()
