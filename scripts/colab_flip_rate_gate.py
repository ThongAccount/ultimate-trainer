#!/usr/bin/env python3
"""colab one-shot: flip-rate trajectory gate.

The ternary optimizer has no learning rate and no damping: the counter
accumulates sign(dW) and the weight flips when |counter| > threshold, after
which the counter resets.  A weight that keeps receiving a *consistent*
gradient therefore flips every `threshold` steps forever — a limit cycle.
The observable signature of a limit cycle is a flip rate that does not decay.

This gate asks whether the flip activity decays when the gradient-noise share
is removed.  Two arms with identical seed / data / model / shapes, differing
only in the noise schedule:

  main arm    noise 1.0 -> 0.3 -> 0.0   (the "noisy" and "clean" windows)
  control arm noise 1.0 throughout      (must SUSTAIN its flip rate)

The control is what makes the main arm's decay meaningful rather than vacuous.

The pass criteria live in tools/flip_rate_verdict.py and were pre-registered
(2026-10-26) before any measurement.  This file only measures.

Mechanism of the measurement
----------------------------
The layer's weight is perturbed by one ternary grid step on a random subset
of weights, so a real residual exists for the optimizer to correct.  The
target is the layer's OWN output before the perturbation, so the target is
exactly representable on the ternary grid and a healthy optimizer can settle
on it (residual -> exactly 0).  Per step the update op is called directly
with dY = dY_true + noise, so the noise enters at the GRADIENT level (the
same dY the production backward would hand the update kernel).

Flip counting is exact: the packed weight words are compared field by field
(2 bits per ternary value, 16 values per int32 word).

Outputs (parsed by scripts/colab_official_suite.py):
  FLIPRATE: ... one line per (update backend x arm)
  FLIPRATE_CHECK: ... one line per criterion
  FLIPRATE_GATE: PASS|FAIL overall (shipped backend)
"""
import argparse
import json
import os
import subprocess
import sys

DRIVER = r'''
import os, sys, math, json
REPO = os.environ.get("REPO_DIR", "/content/ultimate-trainer")
os.chdir(REPO)
sys.path.insert(0, REPO)

import torch
from kernels.packed_ternary import PackedTernaryLinear, pack_tensor, unpack_tensor
try:
    from kernels.packed_ternary.custom_ops import update_tc_v2 as co_update_tc_v2
except Exception:  # custom_ops can fail to register; packed_linear re-exports it
    from kernels.packed_ternary.packed_linear import co_update_tc_v2
from tools.flip_rate_verdict import evaluate_flip_rate

STEPS       = int(os.environ.get("FLIP_STEPS", "1000"))
D_IN        = int(os.environ.get("FLIP_DIN", "512"))
D_OUT       = int(os.environ.get("FLIP_DOUT", "512"))
THRESHOLD   = int(os.environ.get("FLIP_THRESHOLD", "32"))
BATCH       = int(os.environ.get("FLIP_BATCH", "512"))
PERTURB     = float(os.environ.get("FLIP_PERTURB", "0.05"))
SEED        = int(os.environ.get("FLIP_SEED", "1234"))
BACKENDS    = os.environ.get("FLIP_BACKENDS", "custom,cublas").split(",")
DEAD_ZONE   = float(os.environ.get("FLIP_DEAD_ZONE", "0.0"))
FLIP_BUDGET = int(os.environ.get("FLIP_BUDGET", "0"))
# Window boundaries (steps): [0, WARM) warm-up, [WARM, NOISY_END) noisy,
# [NOISY_END, STEPS) clean.  Proportional so a short smoke run still has all
# three windows; at the official STEPS=1000 this is 300 / 300 / 400.
WARM        = int(round(0.30 * STEPS))
NOISY_END   = int(round(0.60 * STEPS))
SMOKE       = os.environ.get("FLIP_SMOKE", "0") == "1"
NOISE_MAIN  = [1.0, 0.3, 0.0]
NOISE_CTRL  = [1.0, 1.0, 1.0]

N_WEIGHTS = D_OUT * D_IN


def noise_scale_for(arm, step):
    sched = NOISE_MAIN if arm == "main" else NOISE_CTRL
    if step < WARM:
        return sched[0]
    if step < NOISY_END:
        return sched[1]
    return sched[2]


def count_flipped_fields(old, new):
    """Exact count of ternary values that changed between two packed tensors."""
    o = old.to(torch.int64) & 0xFFFFFFFF
    n = new.to(torch.int64) & 0xFFFFFFFF
    changed = torch.zeros_like(o)
    for f in range(16):
        changed += (((o >> (2 * f)) & 3) != ((n >> (2 * f)) & 3)).to(torch.int64)
    return int(changed.sum().item())


def build_layer():
    """Layer with a perturbed weight and an exactly-representable target."""
    torch.manual_seed(SEED)
    layer = PackedTernaryLinear(D_IN, D_OUT, threshold=THRESHOLD, bias=False).cuda()
    stride_words = layer.W_packed.shape[1]
    assert stride_words * 16 == D_IN, (
        f"D_IN={D_IN} leaves padding words; flip counts would be inflated")
    gamma = math.sqrt(2.0 / (D_IN + D_OUT))

    # The target IS the layer's current output, so it is exactly representable
    # on the ternary grid: a healthy optimizer can settle on it.
    X = torch.randn(BATCH, D_IN, device="cuda", dtype=torch.float32) / math.sqrt(D_IN)
    X = X.half()
    with torch.no_grad():
        y0 = layer(X).float().clone()

    # Perturb one grid step on a random subset: a real residual to correct.
    W = unpack_tensor(layer.W_packed, D_OUT, D_IN, gamma).cuda()
    g = torch.Generator(device="cpu"); g.manual_seed(SEED + 1)
    sel = (torch.rand(D_OUT, D_IN, generator=g) < PERTURB).cuda()
    Wp = W.clone()
    Wp[sel & (W > 0)] = 0.0
    Wp[sel & (W < 0)] = 0.0
    Wp[sel & (W == 0)] = gamma
    layer.W_packed = pack_tensor(Wp, gamma=gamma).cuda()
    return layer, X, y0, gamma


def run_arm(arm, update_backend):
    os.environ["PT_UPDATE_BACKEND"] = update_backend
    layer, X, y0, gamma = build_layer()
    gen = torch.Generator(device="cuda"); gen.manual_seed(SEED + 2)

    # Reference noise scale: RMS of the true gradient at step 0, so the noise
    # is an ABSOLUTE level that does not shrink as the model converges.
    with torch.no_grad():
        pred = layer(X)
        dY_true0 = (pred.float() - y0) * (2.0 / (BATCH * D_OUT))
    ref_rms = float(dY_true0.pow(2).mean().sqrt().item())

    flips = {"noisy": 0, "clean": 0}
    steps_in = {"noisy": 0, "clean": 0}
    buckets = []
    b_flips, b_resid = 0, 0.0
    resid_noisy, resid_clean = [], []

    for step in range(STEPS):
        ns = noise_scale_for(arm, step)
        with torch.no_grad():
            pred = layer(X)
            dY_true = (pred.float() - y0) * (2.0 / (BATCH * D_OUT))
            resid = float(dY_true.pow(2).mean().sqrt().item())
            noise = torch.randn(dY_true.shape, generator=gen, device="cuda",
                                dtype=torch.float32)
            dY = (dY_true + ns * ref_rms * noise).half()
            old = layer.W_packed.clone()
            co_update_tc_v2(layer.W_packed, layer.counter, layer.flip_count, X, dY,
                            THRESHOLD, DEAD_ZONE, FLIP_BUDGET)
            f = count_flipped_fields(old, layer.W_packed)

        window = "noisy" if WARM <= step < NOISY_END else ("clean" if step >= NOISY_END else "warm")
        if window in flips:
            flips[window] += f
            steps_in[window] += 1
            (resid_noisy if window == "noisy" else resid_clean).append(resid)
        b_flips += f
        b_resid += resid
        if (step + 1) % 100 == 0:
            buckets.append((step + 1, b_flips, b_resid / 100.0))
            b_flips, b_resid = 0, 0.0

    rates = {k: flips[k] / float(max(steps_in[k], 1) * N_WEIGHTS) for k in flips}
    out = {
        "arm": arm,
        "update_backend": update_backend,
        "steps": STEPS,
        "threshold": THRESHOLD,
        "weights": N_WEIGHTS,
        "batch": BATCH,
        "perturb_frac": PERTURB,
        "seed": SEED,
        "ref_rms": ref_rms,
        "dead_zone": DEAD_ZONE,
        "flip_budget": FLIP_BUDGET,
        "flip_rates": rates,
        "steps_in_window": steps_in,
        "resid_rms_noisy": (sum(resid_noisy) / len(resid_noisy)) if resid_noisy else 0.0,
        "resid_rms_clean": (sum(resid_clean) / len(resid_clean)) if resid_clean else 0.0,
        "buckets": buckets,
    }
    return out


results = []
for be in BACKENDS:
    for arm in ("main", "control"):
        r = run_arm(arm, be)
        results.append(r)
        print("FLIPRATE: arm={arm} update_backend={ub} steps={steps} threshold={th} "
              "weights={w} noisy_rate={nr:.8g} clean_rate={cr:.8g} "
              "resid_noisy={rn:.6g} resid_clean={rc:.6g}".format(
                  arm=r["arm"], ub=r["update_backend"], steps=r["steps"],
                  th=r["threshold"], w=r["weights"],
                  nr=r["flip_rates"]["noisy"], cr=r["flip_rates"]["clean"],
                  rn=r["resid_rms_noisy"], rc=r["resid_rms_clean"]), flush=True)
        for (bstep, bf, bres) in r["buckets"]:
            print(f"FLIPRATE_BUCKET: arm={r['arm']} update_backend={r['update_backend']} "
                  f"step={bstep} flips_per_step={bf / 100.0:.6g} "
                  f"flip_rate={bf / (100.0 * r['weights']):.8g} resid_rms={bres:.6g}",
                  flush=True)

by = {(r["update_backend"], r["arm"]): r for r in results}
overall_pass = True
for be in BACKENDS:
    main = by[(be, "main")]
    ctrl = by[(be, "control")]
    v = evaluate_flip_rate(
        main={"noisy": main["flip_rates"]["noisy"], "clean": main["flip_rates"]["clean"]},
        control={"noisy": ctrl["flip_rates"]["noisy"], "clean": ctrl["flip_rates"]["clean"]},
        threshold=THRESHOLD,
    )
    for c in v["checks"]:
        print(f"FLIPRATE_CHECK: update_backend={be} {c['name']} ok={c['ok']} "
              f"value={c['value']:.6g} threshold={c['threshold']:.6g} "
              f"detail={c['detail']}", flush=True)
    print(f"FLIPRATE_VERDICT: update_backend={be} passed={v['passed']}", flush=True)
    if be == BACKENDS[-1]:
        overall_pass = v["passed"]

print(f"FLIPRATE_GATE: {'PASS' if overall_pass else 'FAIL'} "
      f"(shipped backend = {BACKENDS[-1]})", flush=True)
if SMOKE:
    # A smoke run validates the plumbing, not the criteria: a short run
    # cannot satisfy the pre-registered thresholds.
    print("FLIPRATE_GATE: SMOKE (criteria not evaluated)", flush=True)
    sys.exit(0)
sys.exit(0 if overall_pass else 1)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/content/ultimate-trainer")
    ap.add_argument("--no-clone", action="store_true",
                    help="reuse an existing clone (the suite runner clones once)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--threshold", type=int, default=32)
    ap.add_argument("--backends", default="custom,cublas")
    ap.add_argument("--dead-zone", type=float, default=0.0,
                    help="counter dead-zone default 0.0 (no damping); set 1e-3 for the damping run")
    ap.add_argument("--flip-budget", type=int, default=0,
                    help="per-weight flip budget (0 = off); sweep 5/10/20/50 for damping")
    ap.add_argument("--smoke", action="store_true",
                    help="validate the plumbing only; do not enforce the criteria")
    args, _unknown = ap.parse_known_args()

    d = args.repo
    # Colab T4 images ship WITHOUT ninja — the JIT extension builds fail
    # until it is installed.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja"],
                   capture_output=True)
    if not args.no_clone:
        tok = os.environ.get("GITHUB_TOKEN", "")
        url = "https://github.com/ThongAccount/ultimate-trainer.git"
        if tok:
            url = url.replace("https://", f"https://{tok}@")
        if not os.path.isdir(os.path.join(d, ".git")):
            subprocess.run(["git", "clone", "--quiet", url, d], check=True)
        subprocess.run(["git", "-C", d, "fetch", "--quiet", "origin"], check=False)
        subprocess.run(["git", "-C", d, "reset", "--hard", "--quiet", "origin/master"], check=True)
    head = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()[:10]
    print(f"commit={head}", flush=True)

    with open(os.path.join(d, "flip_rate_driver.py"), "w") as fh:
        fh.write(DRIVER)

    env = {**os.environ, "REPO_DIR": d, "FLIP_STEPS": str(args.steps),
           "FLIP_THRESHOLD": str(args.threshold), "FLIP_BACKENDS": args.backends,
           "FLIP_DEAD_ZONE": str(args.dead_zone), "FLIP_BUDGET": str(args.flip_budget),
           "FLIP_SMOKE": "1" if args.smoke else "0"}
    # The driver must be run through a PIPE and re-printed: a child that
    # inherits stdout under a Jupyter kernel writes to a descriptor the CLI
    # never sees (this lost one full T4 allocation on 2026-10-26).
    sys.path.insert(0, d)
    from tools.stream_runner import run_streaming
    rc, out = run_streaming([sys.executable, "flip_rate_driver.py"], cwd=d, env=env)

    outdir = os.path.join(d, "results")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "flip_rate_gate.log"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    with open(os.path.join(outdir, "flip_rate_gate.json"), "w") as fh:
        json.dump({"commit": head, "steps": args.steps,
                   "threshold": args.threshold, "rc": rc, "lines": out},
                  fh, indent=2)
    print(f"ARTEFACT: {os.path.join(outdir, 'flip_rate_gate.json')}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
