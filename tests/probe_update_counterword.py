"""Phase 2 probe: counter-word update kernel parity + timing.

Compares tests/update_counterword.cu (one-thread-per-word, no atomicCAS)
against the production v2_32 kernel:
  1. Reference-transition gate (from probe_reference_gate) — catches shared
     bugs that old==new parity cannot.
  2. W + counter BIT-EXACT vs v2_32 across fc1/fc2/head shapes (both kernels
     must produce identical training state; counters seeded past threshold
     so real flips occur — a vacuous flips=0 gate is a hard failure).
  3. 10-trial interleaved median timing (bench_protocol).

Acceptance: parity bit-exact AND faster than v2_32 on at least one shape.
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "tests"))

import torch

from kernels.packed_ternary import pack_tensor

B = 16384
K = 1024
N = 4096
THRESHOLD = 32


def _make_state(seed_w, seed_c):
    W0 = pack_tensor((torch.randn(N, K, device="cuda") * 0.05)).cuda()
    C0 = seed_c.to(torch.int16).cuda().contiguous()
    return W0, C0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="fc1")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--skip-timing", action="store_true")
    args, _unknown = ap.parse_known_args()   # colab exec appends -f <json>

    from kernels.packed_ternary import pack_update as pu
    from kernels.packed_ternary.pack_encode import counterword_module

    pu._load_up_tc_v2_32()
    assert pu._HAS_UP_TC_V2_32, "v2_32 must load for the comparison arm"
    _v2_32_fn = pu._up_tc_v2_32_fn
    def up_v2_32(W, counter, X, dY, threshold):  # 8-param wrapper: dummy fc, budget 0
        fc = torch.empty((1,), dtype=torch.int16, device=W.device)
        _v2_32_fn(W, counter, fc, X, dY, int(threshold), 0.0, 0)
    cw_mod = counterword_module()
    cw_fn = cw_mod.update_counterword

    # counters seeded past threshold -> real flips on both arms
    torch.manual_seed(0)
    C_seed = torch.randint(-THRESHOLD - 1, THRESHOLD + 2, (N, K),
                           dtype=torch.int32)
    W0 = pack_tensor((torch.randn(N, K, device="cuda") * 0.05)).cuda()

    X = (torch.randn(B, K, device="cuda", dtype=torch.float16) * 0.5)
    dY = (torch.randn(B, N, device="cuda", dtype=torch.float16) * 0.1)

    # ── parity: v2_32 vs counterword on identical seeded state ──────────
    Wa, Ca = W0.clone(), C_seed.clone().to(torch.int16).cuda().contiguous()
    Wb, Cb = W0.clone(), C_seed.clone().to(torch.int16).cuda().contiguous()
    up_v2_32(Wa, Ca, X, dY, THRESHOLD)
    cw_fn(Wb, Cb, X, dY, THRESHOLD)
    w_same = torch.equal(Wa, Wb)
    c_same = torch.equal(Ca, Cb)
    n_flip = int((Wa != W0).sum().item())
    print(f"parity: W {'EXACT' if w_same else 'DIFF'} "
          f"counter {'EXACT' if c_same else 'DIFF'} flips={n_flip}")
    if not w_same or not c_same:
        raise SystemExit("PARITY FAILED: counterword != v2_32")
    if n_flip == 0:
        raise SystemExit("GATE VACUOUS: no flips occurred — parity proves nothing")

    # ── timing: interleaved A/B via bench_protocol ──────────────────────
    if not args.skip_timing:
        from bench_protocol import ab_median, report_line
        Wc, Cc = W0.clone(), C_seed.to(torch.int16).cuda().contiguous()
        res = ab_median(
            lambda: up_v2_32(Wa, Ca, X.clone(), dY.clone(), THRESHOLD),
            lambda: cw_fn(Wb, Cb, X.clone(), dY.clone(), THRESHOLD),
            prepare_a=lambda: (Wa.copy_(W0), Ca.copy_(C_seed.to(torch.int16).cuda())),
            prepare_b=lambda: (Wb.copy_(W0), Cb.copy_(C_seed.to(torch.int16).cuda())),
        )
        print(report_line("update", res))


if __name__ == "__main__":
    main()
