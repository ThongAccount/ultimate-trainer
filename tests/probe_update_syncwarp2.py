"""In-process paired A/B: update TC32 batch-loop CTA barriers (prod vs warp-scope).

C12 hypothesis: the batch loop in gemm_update_tc_v2_32.cu calls __syncthreads()
twice per trip, but dY_smem and X_smem are warp-private slices (DYS/XS are
indexed by warp_id, and both load loops write only warp_id's slice). On T4 the
head grid is 1571 CTAs x 1024 batch trips = 103M CTA-barrier pairs per step.
Replacing the hot-path barrier with __syncwarp() removes CTA-wide convoy
without touching smem or registers.

This is NOT the previously falsified C4 (which swapped one barrier for __syncwarp
without checking whether the tiles were private). Here the whole hot path is
warp-private, so both barriers go.

Gates (all mandatory — a wrong answer here silently corrupts training):

1. Reference-transition gate (NEW kernel vs Python-computed ABSOLUTE
   expectation, tests/probe_reference_gate.py). The old-vs-new parity check
   below is vacuous in the strong sense (sweep #5): it only proves OLD == NEW,
   so any bug shared by BOTH arms — inverted counter sign, non-strict
   threshold compare, wrong column, double flip, missing reset — passes
   because both arms produce the same wrong answer. The reference stage pins
   the kernel to the production semantics read from gemm_update_tc_v2_32.cu:
   counter += -sign(dW) (moves AGAINST the gradient), flip when |cnt| >
   threshold STRICT, counter resets to 0 on flip, increment saturates at +1
   / decrement at -1. Catches: inverted sign convention, >= vs > threshold,
   wrong-row/column addressing, double flip, missing reset, zero-grad skip
   path clobbering counters, plain counter arithmetic.

2. Bit-exact W and counter parity OLD vs NEW on production shapes — catches
   variant-specific divergence (what the barrier swap could plausibly break).
   Refuses to report timings unless the gate is non-vacuous (flips > 0).

Timing: tests.bench_protocol.ab_median — 10 paired trials, median (not min),
clock-settle barrier between trials, A/A null arm on the first shape to
calibrate sigma_null. This is the mandatory C13 protocol (sweep #15); the old
3-trial min-of-3 is retired. A trailing "RESULTS <json>" line preserves the
machine-readable contract (colab_speedpass.py captures stdout; downstream
parsers look for the RESULTS JSON line, same shape as probe_fwd_spill).

Usage: python tests/probe_update_syncwarp2.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CUH = os.path.join(REPO, "kernels/packed_ternary/packed_ternary.cuh")
OLD_CU = os.path.join(REPO, "kernels/packed_ternary/gemm_update_tc_v2_32.cu")
NEW_CU = os.path.join(HERE, "update_syncwarp2.cu")

with open(CUH) as f:
    CUH_SRC = f.read()


def build(cu_path, name):
    from torch.utils.cpp_extension import load_inline

    with open(cu_path) as f:
        cu = f.read()
    combined = CUH_SRC + "\n" + cu.replace('#include "packed_ternary.cuh"', "")
    return load_inline(
        name=name,
        cpp_sources=r"""
        #include <cuda_runtime.h>
        #include <torch/extension.h>
        extern "C" {
            void launch_packed_ternary_update_tc_v2(
                const void* X, const void* dY, uint32_t* W, int16_t* counter,
                int batch_size, int in_features, int out_features,
                int stride_words, int16_t threshold, cudaStream_t stream);
        }
        void upd(torch::Tensor X, torch::Tensor dY, torch::Tensor W,
                 torch::Tensor counter, int64_t in_features, int64_t threshold) {
            launch_packed_ternary_update_tc_v2(
                X.data_ptr<at::Half>(), dY.data_ptr<at::Half>(),
                reinterpret_cast<uint32_t*>(W.data_ptr<int32_t>()),
                reinterpret_cast<int16_t*>(counter.data_ptr<int16_t>()),
                X.size(0), (int)in_features, (int)W.size(0),
                (int)W.size(1), (int16_t)threshold, nullptr);
        }
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("upd", &upd); }
        """,
        cuda_sources=[combined],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def main():
    from tests.bench_protocol import ab_median, report_line
    from tests.probe_reference_gate import run_reference_gate

    dev = "cuda"
    torch.manual_seed(0)
    BATCH = 16384
    THRESHOLD = 8  # small enough that flips actually happen during parity

    print("building OLD (production, CTA barriers) ...", flush=True)
    old = build(OLD_CU, "upd_sw_old")
    print("built OLD ok", flush=True)
    print("building NEW (warp-scope barriers) ...", flush=True)
    new = build(NEW_CU, "upd_sw_new")
    print("built NEW ok", flush=True)

    # ── Gate 1: reference-transition (ABSOLUTE semantics, NEW kernel) ──
    # One deterministic update on a crafted 64x64 fixture whose expected
    # W/counter state is computed in Python from the production rules.
    # This is the only stage that can catch bugs shared by OLD and NEW.
    print("\nREFERENCE-TRANSITION GATE (NEW kernel vs Python expectation):",
          flush=True)
    try:
        rg = run_reference_gate(new.upd, dev, threshold=THRESHOLD)
        print(f"  PASS: {rg['positions']} probe positions, "
              f"{rg['resets']} flips/counter-resets bit-exact "
              f"(sign=counter-against-grad, strict |cnt|>{THRESHOLD}, "
              f"reset-to-0, W saturates at +/-1)", flush=True)
    except AssertionError as e:
        raise SystemExit(f"REFERENCE GATE FAIL: {e} — refusing to continue")

    print(
        f"\n{'name':<6} {'in':>5} {'out':>6}   {'old_ms':>8} {'new_ms':>8} {'d%':>7}"
        f"   {'W':>6} {'counter':>8}",
        flush=True,
    )

    results = {}
    for si, (name, inn, out) in enumerate([
        ("fc1", 1024, 4096),
        ("fc2", 4096, 1024),
        ("head", 1024, 50272),
    ]):
        X = torch.randn(BATCH, inn, device=dev, dtype=torch.float16) * 0.1
        dY = torch.randn(BATCH, out, device=dev, dtype=torch.float16) * 0.1
        W0 = torch.randint(0, 4, (out, (inn + 15) // 16), device=dev, dtype=torch.int32)
        W0 = torch.where(W0 == 3, 0, W0)
        # Seed counters AT/PAST the threshold so a single pass triggers real
        # weight flips. With zero-initialised counters the gradient moves them
        # by +/-1 and never reaches threshold, so W is never mutated and the
        # parity gate is vacuous (measured: flips=0). The kernel's entire job is
        # the flip, so the gate must exercise it.
        C0 = torch.randint(
            -(THRESHOLD + 1), THRESHOLD + 2, (out * inn,), device=dev, dtype=torch.int16
        )

        # Gate 2: parity — both kernels mutate W and counter in place, so run
        # each on a private copy and compare bit-for-bit.
        Wo, Co = W0.clone(), C0.clone()
        Wn, Cn = W0.clone(), C0.clone()
        old.upd(X, dY, Wo, Co, inn, THRESHOLD)
        new.upd(X, dY, Wn, Cn, inn, THRESHOLD)
        torch.cuda.synchronize()
        w_same = torch.equal(Wo, Wn)
        c_same = torch.equal(Co, Cn)
        n_flip = int((Wo != W0).sum().item())
        n_ctr = int((Co != C0).sum().item())
        print(
            f"  parity {name}: W {'EXACT' if w_same else 'DIFF'} "
            f"counter {'EXACT' if c_same else 'DIFF'} "
            f"flips={n_flip} counters_changed={n_ctr}",
            flush=True,
        )
        if not (w_same and c_same):
            raise SystemExit(f"PARITY FAIL on {name} — refusing to report timings")
        if n_flip == 0:
            raise SystemExit(
                f"GATE VACUOUS on {name}: no weight flips occurred, so W parity "
                f"proves nothing about the path this kernel exists for. "
                f"threshold={THRESHOLD}"
            )

        # Timing: bench_protocol.ab_median — 10 paired trials, median,
        # clock-settle between trials, null arm on the first shape. Update is
        # destructive, so prepare_* rebuilds fresh state OUTSIDE the timed
        # region.
        res = ab_median(
            lambda: old.upd(X, dY, W0.clone(), C0.clone(), inn, THRESHOLD),
            lambda: new.upd(X, dY, W0.clone(), C0.clone(), inn, THRESHOLD),
            null_arm=(si == 0),
        )
        results[name] = res
        line = report_line(f"upsw2/{name}", res)
        print("  " + line, flush=True)
        if res.sigma_null_pct is not None:
            print(
                f"    (null arm: sigma_null={res.sigma_null_pct:.2f}%, "
                f"decision threshold={max(2 * res.sigma_null_pct, 1.0):.2f}%)",
                flush=True,
            )
        print(
            f"{name:<6} {inn:>5} {out:>6}   {res.a_ms:8.2f} {res.b_ms:8.2f} "
            f"{res.delta_pct:7.2f}   {'ok':>6} {'ok':>8}",
            flush=True,
        )

    # Machine-readable summary — same RESULTS-JSON-line contract as
    # probe_fwd_spill (colab_speedpass.py captures the whole stdout as "log";
    # downstream tooling greps this line).
    print("\nRESULTS " + json.dumps({
        name: {
            "old_ms": round(res.a_ms, 3),
            "new_ms": round(res.b_ms, 3),
            "delta_pct": round(res.delta_pct, 3),
            "sigma_null_pct": (round(res.sigma_null_pct, 3)
                               if res.sigma_null_pct is not None else None),
        }
        for name, res in results.items()
    }), flush=True)


if __name__ == "__main__":
    main()
