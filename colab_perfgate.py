#!/usr/bin/env python3
"""Colab one-shot PERF GATE (C13 sweep item #19 / test-plan item 7).

Benchmarks the production kernel trio (fwd TC64, backward-dX TC64, update
v2_32) at the 3 production shapes plus one full train step, and FAILS (exit 1)
if any median regresses >=5% vs the committed baseline
`tests/perf_baseline_t4.json`.  WARN at >=2% (report only).

Probes measure; the gate FAILS.  This is the regression tripwire the sweep
found missing: until now nothing in the repo ever turned red on a perf loss.

Structure mirrors colab_pytest.py / colab_speedpass.py:
- clone committed HEAD via GITHUB_TOKEN (untracked files are invisible to
  the VM — commit first),
- pip install ninja,
- in-process measurement with tests/bench_protocol discipline (10-trial
  median, warmup 10, torch.cuda._sleep clock-settle between trials),
- `RESULTS JSON:` block on stdout, nothing else to parse,
- exit code gates CI: 0 = no FAIL, 1 = any FAIL.

Production call paths (do NOT bench a different kernel than prod runs):
- fwd:  pack_forward.packed_ternary_forward_tc_64  (TC64, -O3 --use_fast_math)
- dX:   custom_ops.backward_dx_tc  (auto-routes 64x64 at all 3 shapes)
- up:   custom_ops.update_tc_v2    (loads v2_32; THRESHOLD=32 like prod)
- step: train_gigatoken.train_step_cudagraph on the full 6-layer model
        (B=32, SEQ=512, K=1024, VOCAB=50272, THRESHOLD=32)

Usage
-----
Gate (fails on regression):
    TOK=$(gh auth token)
    colab run --gpu T4 --timeout 5400 --env "GITHUB_TOKEN=$TOK" colab_perfgate.py

First run / baseline regeneration (writes fresh numbers, inspect, commit):
    colab run --gpu T4 --timeout 5400 --env "GITHUB_TOKEN=$TOK" \
        colab_perfgate.py -- --update-baseline
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

REPO_URL = "github.com/ThongAccount/ultimate-trainer.git"
DEFAULT_CLONE = "/content/ultimate-trainer"
DEFAULT_BRANCH = "chore/speedpass"

BASELINE_NAME = "tests/perf_baseline_t4.json"

# Production config (train_gigatoken.py): B=32 * SEQ=512 = 16384 reduction
# rows; K=1024; VOCAB=50272; counter flip threshold 32.
BATCH = 16384
THRESHOLD = 32
SHAPES = [
    ("fc1", 1024, 4096),
    ("fc2", 4096, 1024),
    ("head", 1024, 50272),
]
KERNELS = ("fwd", "dx", "update")


def clone_repo(branch: str, dest: str, token: str) -> str:
    if os.path.isdir(os.path.join(dest, "kernels", "packed_ternary")):
        return dest
    if not token:
        raise SystemExit(
            "GITHUB_TOKEN not set (pass --env 'GITHUB_TOKEN=$(gh auth token)')"
        )
    url = f"https://x-access-token:{token}@{REPO_URL}"
    r = subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch, "--quiet", url, dest],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        err = (r.stderr or "").replace(token, "<token>")
        raise SystemExit(f"git clone failed (rc={r.returncode}): {err[-600:]}")
    return dest


def ensure_ninja() -> None:
    try:
        import ninja  # noqa: F401

        return
    except ImportError:
        pass
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "ninja"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise SystemExit(f"pip install ninja failed: {(r.stderr or '')[-400:]}")


def make_case(name: str, inn: int, out: int):
    """One (kernel, shape) bench case: production entry point + fresh state."""
    import torch

    from bench_protocol import solo_median

    dev = "cuda"
    torch.manual_seed(0)
    X = torch.randn(BATCH, inn, device=dev, dtype=torch.float16)
    dY = (torch.randn(BATCH, out, device=dev, dtype=torch.float16) * 0.5).half()
    W0 = torch.randint(0, 4, (out, (inn + 15) // 16), device=dev, dtype=torch.int32)
    W0 = torch.where(W0 == 3, 0, W0)  # ternary codes: no 3
    C0 = torch.zeros(out, inn, device=dev, dtype=torch.int16)

    W = W0.clone()
    C = C0.clone()

    import kernels.packed_ternary.custom_ops as co

    def prep():
        # update mutates W and counter; restore BEFORE timing so every trial
        # sees identical state (bench_protocol runs prepare outside timing).
        W.copy_(W0)
        C.copy_(C0)

    cases = {}
    cases["fwd"] = solo_median(
        lambda: co.forward_tc(W, X, inn),
        prepare=None,          # fwd is pure compute: Y is a fresh alloc
        null_arm=(name == "fc1"),
    )
    cases["dx"] = solo_median(
        lambda: co.backward_dx_tc(W, dY, inn),
        prepare=None,          # dX is pure compute
    )
    cases["update"] = solo_median(
        lambda: co.update_tc_v2(W, C, X, dY, THRESHOLD),
        prepare=prep,          # W + counter state restored per trial
    )
    del X, dY, W, C, W0, C0
    torch.cuda.empty_cache()
    return cases


def bench_step(repo: str):
    """One full train step on the production 6-layer model, 10-trial median.

    Mirrors train_gigatoken.py's no-sync step (profile=False) with the same
    wall-clock + synchronize discipline as bench_protocol._time_once.
    """
    import torch

    from bench_protocol import WARMUP, N_TRIALS, _settle

    import train_gigatoken as tg

    model = tg.build_model()
    model.train()
    x = torch.randint(0, tg.VOCAB, (tg.B, tg.SEQ), dtype=torch.long, device="cuda")
    y = torch.randint(0, tg.VOCAB, (tg.B, tg.SEQ), dtype=torch.long, device="cuda")

    def step():
        loss = tg.train_step_cudagraph(model, x, y, profile=False)
        torch.cuda.synchronize()
        return loss

    for _ in range(WARMUP):
        step()
    _settle()
    trials = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        step()
        trials.append((time.perf_counter() - t0) * 1000.0)
        _settle()

    import statistics

    return statistics.median(trials), trials


def main() -> int:
    ap = argparse.ArgumentParser(description="perf gate: trio + step vs baseline")
    ap.add_argument("--branch", default=DEFAULT_BRANCH)
    ap.add_argument("--repo-dir", default=DEFAULT_CLONE)
    ap.add_argument("--baseline", default=BASELINE_NAME,
                    help="baseline JSON path relative to the repo root")
    ap.add_argument("--update-baseline", action="store_true",
                    help="write fresh numbers back to the baseline file "
                         "(meant for regenerating after a visual inspection "
                         "of the printed per-kernel deltas, then committing)")
    args = ap.parse_args()

    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} gpu={gpu}",
          flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("no GPU — perf gate needs a T4")

    repo = clone_repo(args.branch, args.repo_dir, os.environ.get("GITHUB_TOKEN", ""))
    os.chdir(repo)
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.join(repo, "tests"))
    commit = (
        subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"])
        .decode().strip()
    )
    print(f"repo={repo} branch={args.branch} commit={commit}", flush=True)

    ensure_ninja()

    with open(os.path.join(repo, args.baseline)) as f:
        baseline = json.load(f)

    t0 = time.time()
    from bench_protocol import regress

    # A/A null on fc1/fwd (first measurement) calibrates sigma_null for the
    # session; reported in RESULTS so deltas near WARN can be sanity-checked.
    measurements = {}
    lines = []
    any_fail = False
    for name, inn, out in SHAPES:
        cases = make_case(name, inn, out)
        measurements[name] = cases
        for kernel in KERNELS:
            res = cases[kernel]
            base = baseline["kernels"][name][f"{kernel}_ms"]
            line = regress(f"{name}/{kernel}", res, base)
            lines.append(line)
            if "FAIL" in line:
                any_fail = True
            print(line + (
                f" sigma_null={res.sigma_null_pct:.2f}%"
                if res.sigma_null_pct is not None else ""),
                flush=True,
            )

    step_base = baseline["step_ms"]
    step_ms, step_trials = bench_step(repo)
    d = 100.0 * (step_ms - step_base) / step_base
    verdict = ("FAIL" if d >= 5.0 else ("WARN" if d >= 2.0 else "ok"))
    if args.update_baseline:
        import bench_protocol as bp

        fresh = {
            "schema": 1,
            "gpu": gpu,
            "_meta": {
                "status": "measured",
                "note": (f"regenerated by colab_perfgate.py --update-baseline "
                         f"at commit {commit}"),
                "protocol": (f"{bp.N_TRIALS} trials, median, warmup "
                             f"{bp.WARMUP}, torch.cuda._sleep settle "
                             "(bench_protocol C13)"),
            },
            "config": {
                "batch": BATCH,
                "threshold": THRESHOLD,
                "n_trials": bp.N_TRIALS,
                "warmup": bp.WARMUP,
                "shapes": {
                    n: {"in": i, "out": o} for n, i, o in SHAPES
                },
            },
            "kernels": {
                n: {
                    "fwd_ms": round(measurements[n]["fwd"].b_ms, 3),
                    "dx_ms": round(measurements[n]["dx"].b_ms, 3),
                    "update_ms": round(measurements[n]["update"].b_ms, 3),
                }
                for n, _, _ in SHAPES
            },
            "step_ms": round(step_ms, 3),
        }
        out_path = os.path.join(repo, args.baseline)
        with open(out_path, "w") as f:
            json.dump(fresh, f, indent=2)
            f.write("\n")
        print(f"baseline written: {out_path} — INSPECT the deltas above, "
              "then commit if they look sane", flush=True)

    wall = round(time.time() - t0, 1)
    res = {
        "suite": "perfgate",
        "commit": commit,
        "branch": args.branch,
        "gpu": gpu,
        "pytorch": torch.__version__,
        "baseline_status": baseline.get("_meta", {}).get("status", "?"),
        "update_baseline": args.update_baseline,
        "kernels": {
            n: {
                f"{k}_ms": round(measurements[n][k].b_ms, 3)
                for k in KERNELS
            }
            for n, _, _ in SHAPES
        },
        "step_ms": round(step_ms, 3),
        "deltas": lines,
        "any_fail": any_fail,
        "passed": not any_fail,
        "returncode": 1 if any_fail else 0,
        "wall_s": wall,
    }
    print("\nRESULTS JSON:", json.dumps(res, indent=2), flush=True)
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
