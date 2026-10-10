"""In-process paired A/B: update TC32 double-buffered batch loop (GO item #3).

Sweep #20's single best hypothesis: the update kernel's 1024 serial batch
iterations each pay ~2200 cycles of exposed load latency before the mma
(1.8% DRAM roofline, 12% TC peak — the machine idles at the barriers).
This probe tests software pipelining: issue tile i+1's LDGs into a second
smem buffer while iteration i's mma runs, with ONE barrier per steady-state
iteration (production pays two). Plain LDG, not cp.async — the overlap is
what matters (sm_75 cp.async + WMMA under load_inline is fragile; the
plain-load variant preserves the exact same overlap window).

PRE-REGISTERED RISK (the falsifier, decided before measurement): the second
buffer adds 4KB smem (8KB -> 12KB/block). On sm_75 that floors occupancy at
5 CTAs/SM (65536/12288) vs the production 8 (thread-cap), so the loss may
dominate the latency win. Predictions, both ways:
  - overlap wins:  fc1 ~45.4ms -> 36-41ms (-10 to -20%)
  - occupancy loss dominates: fc1 +5-10% slower

Falsification criterion (pre-registered): fc1 delta within +/-2% or slower
-> the pipelining class is CLOSED permanently per sweep #20.

Gate: W and counter must be BIT-EXACT vs production, and the gate must be
non-vacuous — counters are seeded past the threshold (probe_update_syncwarp2
pattern, lines ~76-102) so real weight flips occur; zero-flip parity proves
nothing about the path this kernel exists for.

Timing: tests.bench_protocol.ab_median (10-trial median, interleaved,
clock-settle barriers), null arm (A/A) on fc1 to calibrate sigma_null.
Update is destructive, so each timed call runs on a fresh W/C clone
(prepare is excluded from timing).

Usage: python tests/probe_update_pipeline.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CUH = os.path.join(REPO, "kernels/packed_ternary/packed_ternary.cuh")
OLD_CU = os.path.join(REPO, "kernels/packed_ternary/gemm_update_tc_v2_32.cu")
NEW_CU = os.path.join(HERE, "update_pipeline2.cu")

with open(CUH) as f:
    CUH_SRC = f.read()


def build(cu_path, name):
    from torch.utils.cpp_extension import load_inline

    with open(cu_path) as f:
        cu = f.read()
    combined = CUH_SRC + "\n" + cu.replace('#include "packed_ternary.cuh"', "")
    # Production flags per pack_update.py:453 (_load_up_tc_v2_32): -O2, no
    # fast-math. Probes must compile with the exact production flags so the
    # SASS under test is the SASS that would ship.
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
        extra_cuda_cflags=["-O2"],
    )


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required: this probe times the T4 update kernel")

    from tests.bench_protocol import ab_median, report_line
    from tests.probe_reference_gate import run_reference_gate

    dev = "cuda"
    torch.manual_seed(0)
    BATCH = 16384
    THRESHOLD = 8  # small enough that flips actually happen during parity

    print("building OLD (production, single-buffer, 2 barriers/iter) ...", flush=True)
    old = build(OLD_CU, "upd_pipe_old")
    print("built OLD ok", flush=True)
    print("building NEW (double-buffer, 1 barrier/iter) ...", flush=True)
    new = build(NEW_CU, "upd_pipe_new")
    print("built NEW ok", flush=True)

    # Reference-transition gate (sweep #5): parity alone cannot catch a bug
    # shared by both arms. This checks actual v2 semantics — sign=counter-
    # against-grad, strict |cnt| > threshold, reset-to-0, W saturates at
    # +/-1 — against a Python-computed expectation, on BOTH arms.
    print("\n── reference-transition gate (semantics, both arms) ──", flush=True)
    for arm, lib in [("OLD", old), ("NEW", new)]:
        rg = run_reference_gate(lib.upd, dev, threshold=THRESHOLD)
        print(f"  {arm} PASS: {rg['positions']} probe positions, "
              f"{rg['resets']} flips/counter-resets bit-exact", flush=True)


    shapes = [
        ("fc1", 1024, 4096),
        ("fc2", 4096, 1024),
        ("head", 1024, 50272),
    ]

    print("\n── parity gate (bit-exact W+counter, seeded counters) ──", flush=True)
    for name, inn, out in shapes:
        X = torch.randn(BATCH, inn, device=dev, dtype=torch.float16) * 0.1
        dY = torch.randn(BATCH, out, device=dev, dtype=torch.float16) * 0.1
        W0 = torch.randint(0, 4, (out, (inn + 15) // 16), device=dev, dtype=torch.int32)
        W0 = torch.where(W0 == 3, 0, W0)
        # Seed counters AT/PAST the threshold so a single pass triggers real
        # weight flips. With zero-initialised counters the gradient moves them
        # by +/-1 and never reaches threshold, so W is never mutated and the
        # parity gate is vacuous (measured: flips=0). The kernel's entire job
        # is the flip, so the gate must exercise it.
        C0 = torch.randint(
            -(THRESHOLD + 1), THRESHOLD + 2, (out * inn,), device=dev, dtype=torch.int16
        )

        # parity: both kernels mutate W and counter in place, so run each on a
        # private copy and compare bit-for-bit.
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

    # ── timing: bench_protocol ab_median (10-trial, interleaved, settle) ──
    # Null arm on fc1 calibrates sigma_null empirically before any delta is
    # believed (sweep #15 protocol).
    print("\n── timing (ab_median, 10 trials, interleaved, null arm on fc1) ──",
          flush=True)
    for name, inn, out in shapes:
        X = torch.randn(BATCH, inn, device=dev, dtype=torch.float16) * 0.1
        dY = torch.randn(BATCH, out, device=dev, dtype=torch.float16) * 0.1
        W0 = torch.randint(0, 4, (out, (inn + 15) // 16), device=dev, dtype=torch.int32)
        W0 = torch.where(W0 == 3, 0, W0)
        C0 = torch.randint(
            -(THRESHOLD + 1), THRESHOLD + 2, (out * inn,), device=dev, dtype=torch.int16
        )

        # fresh destructive state per call; clone cost lands in prepare,
        # which _time_once runs OUTSIDE the timed window.
        Wa = [None]
        Ca = [None]
        Wb = [None]
        Cb = [None]

        def prep_a():
            Wa[0], Ca[0] = W0.clone(), C0.clone()

        def prep_b():
            Wb[0], Cb[0] = W0.clone(), C0.clone()

        def run_a():
            old.upd(X, dY, Wa[0], Ca[0], inn, THRESHOLD)

        def run_b():
            new.upd(X, dY, Wb[0], Cb[0], inn, THRESHOLD)

        null = name == "fc1"  # null arm on first shape only (protocol contract)
        res = ab_median(run_a, run_b, prep_a, prep_b, null_arm=null)
        print(report_line(name, res), flush=True)

        # Pre-registered falsification readout (sweep #20): the pipelining
        # class closes if the fc1 delta is within +/-2% or slower.
        if name == "fc1" and res.delta_pct >= -2.0:
            print(
                "\nPRE-REGISTERED FALSIFIER HIT on fc1: "
                f"delta {res.delta_pct:+.2f}% is within ±2% or slower.\n"
                "Verdict: the update double-buffer/pipelining class is CLOSED "
                "per docs/speedpass/2026-10-10-20-agent-crack-sweep.md (#20). "
                "Remaining shapes are reported for the record only.",
                flush=True,
            )


if __name__ == "__main__":
    main()
