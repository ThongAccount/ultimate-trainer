#!/usr/bin/env python3
"""colab run one-shot: validate feat/scale-fold on ONE T4 (cublas_cublas arm).

Prereg: docs/speedpass/2026-11-02-scale-fold-prereg.md.

Steps:
  1. clone --branch feat/scale-fold, --expect-commit guard (rc=2 on mismatch)
  2. pytest gate (colab_pytest.TEST_MODULES) — incl. test_scale_fold_parity
  3. same-VM A/B: child A = PT_FOLD_SCALE unset, child B = PT_FOLD_SCALE=0.03125
     (both cublas_cublas, 60 real steps, warmup 10), interleaved per-step
     timing in ONE process is impossible (env read at import), so each child
     runs its own 60-step block; medians compared with the A/A null from a
     repeated child-A run. Each child prints per-step ms + flip counts at
     steps 1/29/59 — flip trajectory parity is the bit-stable gate.

rc: 0 pass, 1 fail, 2 commit-mismatch (never committed).
"""
import argparse
import json
import os
import subprocess
import sys

CHILD = r'''
import json, os, statistics, sys, time
REPO = os.environ.get("REPO_DIR", "/content/ultimate-trainer")
os.chdir(REPO); sys.path.insert(0, os.getcwd())
name = [a for a in sys.argv[1:] if not a.startswith("--")][0]
import torch
assert torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (7, 5)
import train_gigatoken as tg

B, T = 32, 512
samples = torch.load("data/tokenized/samples_512.pt", map_location="cpu",
                     weights_only=False)
flat = torch.cat([s.reshape(-1).long() for s in samples])
need = B * T
bs, i = [], 0
while len(bs) < 3 and (i + 1) * need <= flat.numel():
    ids = flat[i * need:(i + 1) * need].view(B, T).cuda()
    bs.append((ids[:, :-1].contiguous(), ids.contiguous()))
    i += 1

torch.manual_seed(0)
model = tg.build_model()
times, flips = [], []
STEPS = 60
for s in range(STEPS):
    x, y = bs[s % len(bs)]
    t0 = time.perf_counter()
    loss = tg.train_step_cudagraph(model, x, y, profile=False)
    lv = float(loss.item())
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000)
    if s in (1, 29, STEPS - 1):
        fl = sum(int(l.flip_count.sum().item()) for l in
                 list(model.fc1s) + list(model.fc2s) + [model.head])
        flips.append({"step": s, "loss": round(lv, 6), "flips": fl})
        print(f"  step {s:3d} loss {lv:8.4f} t {times[-1]:7.1f} ms flips={fl}",
              flush=True)
res = {"child": name, "fold": os.environ.get("PT_FOLD_SCALE", "off"),
       "median_ms": round(statistics.median(times[10:]), 2),
       "mean_ms": round(statistics.mean(times[10:]), 2),
       "checkpoints": flips}
print("RESULTS_JSON: " + json.dumps(res), flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/content/ultimate-trainer")
    ap.add_argument("--child", default="")
    ap.add_argument("--branch", default="feat/scale-fold")
    ap.add_argument("--expect-commit", default=os.environ.get("PT_EXPECT_COMMIT", ""))
    ap.add_argument("--tag", default="")
    args, _ = ap.parse_known_args()

    if args.child:
        exec(compile(CHILD, "<child>", "exec"), {"__name__": "__main__"})
        return 0

    d = args.repo
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja"],
                   capture_output=True)
    tok = os.environ.get("GITHUB_TOKEN", "")
    url = "https://github.com/ThongAccount/ultimate-trainer.git"
    if tok:
        url = url.replace("https://", f"https://{tok}@")
    if not os.path.isdir(os.path.join(d, ".git")):
        subprocess.run(["git", "clone", "--quiet", url, d], check=True)
    subprocess.run(["git", "-C", d, "fetch", "--quiet", "origin"], check=True)
    subprocess.run(["git", "-C", d, "reset", "--hard", "--quiet",
                    f"origin/{args.branch}"], check=True)
    head = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    tag = args.tag or head[:8]
    print(f"commit={head} branch={args.branch} tag={tag}", flush=True)
    if d not in sys.path:
        sys.path.insert(0, d)
    from tools.fetch_prebuilt import ensure_prebuilt
    ensure_prebuilt(d)
    from tools.artefact_provenance import (collect_provenance, mark_invalid,
                                           write_artefact)
    from tools.stream_runner import run_streaming
    provenance = collect_provenance(d, extra_config={"config": {
        "branch": args.branch, "expect_commit": args.expect_commit or None}})
    if args.expect_commit and not head.startswith(args.expect_commit):
        print(f"COMMIT-MISMATCH: HEAD={head[:10]} != expect", flush=True)
        art = {"commit": head, "provenance": provenance, "rc": 2}
        mark_invalid(art, "commit-mismatch refusal", root_cause_commit=head)
        print("ARTEFACT:", write_artefact(d, f"scale_fold.{tag}.json", art,
                                          root_cause_commit=head), flush=True)
        return 2

    import colab_pytest
    rc, out = run_streaming([sys.executable, "-m", "pytest", "-q", "--tb=short",
                             *colab_pytest.TEST_MODULES], cwd=d)
    log = os.path.join(d, "results", f"scale_fold.{tag}.pytest.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as fh:
        fh.write(f"commit={head}\nrc={rc}\n\n" + "\n".join(out))
    print(f"STEP_RC: pytest {rc}", flush=True)
    pytest_ok = rc == 0

    results = {}
    for name, extra in (("fold-off", {}), ("fold-off-null", {}),
                        ("fold-on", {"PT_FOLD_SCALE": "0.03125"})):
        env = {**os.environ, "REPO_DIR": d, "PT_BACKEND": "cublas",
               "PT_UPDATE_BACKEND": "cublas", **extra}
        print(f"=== CHILD {name} ===", flush=True)
        rc, out = run_streaming([sys.executable,
                                 os.path.join(d, "scripts", "colab_scale_fold_ab.py"),
                                 "--child", name], cwd=d, env=env)
        log = os.path.join(d, "results", f"scale_fold.{tag}.{name}.log")
        with open(log, "w") as fh:
            fh.write(f"commit={head}\nrc={rc}\n\n" + "\n".join(out))
        print(f"CHILD_RC: {name} {rc}", flush=True)
        for line in out:
            if line.startswith("RESULTS_JSON: "):
                results[name] = json.loads(line[len("RESULTS_JSON: "):])

    off = results.get("fold-off", {})
    null = results.get("fold-off-null", {})
    on = results.get("fold-on", {})
    import statistics
    sigma = None
    if off.get("median_ms") and null.get("median_ms"):
        sigma = abs(100 * (null["median_ms"] - off["median_ms"]) / off["median_ms"])
    delta_pct = None
    if off.get("median_ms") and on.get("median_ms"):
        delta_pct = 100 * (on["median_ms"] - off["median_ms"]) / off["median_ms"]
    # bit-stable gate: flip trajectories must match step-for-step
    traj_match = (off.get("checkpoints") == on.get("checkpoints"))
    passed = pytest_ok and delta_pct is not None and delta_pct < -max(sigma or 1.0, 1.0) \
        and traj_match
    art = {"commit": head, "tag": tag, "provenance": provenance,
           "verdict": "pass" if passed else "fail",
           "pytest_ok": pytest_ok,
           "sigma_null_pct_aa": sigma, "delta_pct": delta_pct,
           "flip_trajectory_match": traj_match, "children": results}
    path = write_artefact(d, f"scale_fold.{tag}.json", art, root_cause_commit=head)
    print("ARTEFACT:", path, flush=True)
    print(f"SCALE_FOLD: {'PASS' if passed else 'FAIL'} delta={delta_pct} "
          f"sigma={sigma} traj_match={traj_match} pytest={pytest_ok}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
