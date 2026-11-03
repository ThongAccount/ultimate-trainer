#!/usr/bin/env python3
"""colab run one-shot: sweep cuBLAS GEMM algos for the dW GEMM (item #7).

Prereg: docs/speedpass/2026-11-03-dw-algo-prereg.md. One T4.
1. clone --branch feat/dw-algo (+ --expect-commit, rc=2 on mismatch)
2. pytest gate
3. per shape (head 16352x50272x1024, fc 16352x4096x1024): time dw_gemm_f32
   for algo in [-1, 0..23] (invalid algos caught), 10 interleaved trials vs
   the -1 default; parity vs default: max|dW diff|, sign-disagreement rate.
4. step A/B cublas_cublas with PT_DW_ALGO=<best head algo> vs unset, 60 steps,
   flip trajectory compared.
rc: 0 pass, 1 fail, 2 commit-mismatch.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

SWEEP = r'''
import json, os, statistics, sys, time
REPO = os.environ.get("REPO_DIR", "/content/ultimate-trainer")
os.chdir(REPO); sys.path.insert(0, os.getcwd())
import torch
assert torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (7, 5)
from kernels.packed_ternary import update_cublas as uc
uc._load_update_cublas()
assert uc._HAS_UPDATE_CUBLAS, "update_cublas ext failed to load"
gemm = uc._dw_gemm_f32

torch.manual_seed(0)
res = {"shapes": {}}
for label, B, N, K in (("head", 16352, 50272, 1024), ("fc", 16352, 4096, 1024)):
    dY = torch.randn(B, N, device="cuda", dtype=torch.float16) * 0.01
    X = torch.randn(B, K, device="cuda", dtype=torch.float16)
    dW = torch.empty(N, K, device="cuda", dtype=torch.float32)
    dW_ref = torch.empty(N, K, device="cuda", dtype=torch.float32)

    def run(algo, out):
        gemm(dY, X, out, algo)

    def timeit(algo, trials=10):
        for _ in range(3):
            run(algo, dW)
        torch.cuda.synchronize()
        ts = []
        for _ in range(trials):
            t0 = time.perf_counter(); run(algo, dW); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        return statistics.median(ts)

    t_def = timeit(-1)
    run(-1, dW_ref)
    rows = []
    for algo in [-1] + list(range(24)):
        try:
            t = timeit(algo)
            run(algo, dW)
            torch.cuda.synchronize()
            diff = (dW - dW_ref).abs()
            sign_dis = ((dW > 0) != (dW_ref > 0)).float().mean().item()
            rows.append({"algo": algo, "median_ms": round(t, 3),
                         "delta_pct": round(100 * (t - t_def) / t_def, 2),
                         "max_abs_diff": float(diff.max()),
                         "sign_disagree_rate": sign_dis})
        except Exception as e:
            rows.append({"algo": algo, "error": str(e)[:80]})
    rows.sort(key=lambda r: r.get("median_ms", 1e9))
    res["shapes"][label] = {"default_ms": round(t_def, 3), "rows": rows[:12],
                            "all": rows}
    print(f"== {label}: default {t_def:.2f} ms; top: " +
          ", ".join(f"a{r['algo']}={r.get('median_ms')}ms" for r in rows[:5]),
          flush=True)

print("RESULTS_JSON: " + json.dumps(res), flush=True)
'''

STEP = r'''
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
for s in range(60):
    x, y = bs[s % len(bs)]
    t0 = time.perf_counter()
    loss = tg.train_step_cudagraph(model, x, y, profile=False)
    lv = float(loss.item())
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000)
    if s in (1, 29, 59):
        fl = sum(int(l.flip_count.sum().item()) for l in
                 list(model.fc1s) + list(model.fc2s) + [model.head])
        flips.append({"step": s, "loss": round(lv, 6), "flips": fl})
print("RESULTS_JSON: " + json.dumps({
    "child": name, "dw_algo": os.environ.get("PT_DW_ALGO", "default"),
    "median_ms": round(statistics.median(times[10:]), 2),
    "checkpoints": flips}), flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/content/ultimate-trainer")
    ap.add_argument("--mode", default="", choices=["", "sweep", "step"])
    ap.add_argument("--branch", default="feat/dw-algo")
    ap.add_argument("--expect-commit", default=os.environ.get("PT_EXPECT_COMMIT", ""))
    ap.add_argument("--tag", default="")
    ap.add_argument("--best-algo", default="-1")
    args, _ = ap.parse_known_args()

    if args.mode:
        exec(compile(SWEEP if args.mode == "sweep" else STEP, "<child>", "exec"),
             {"__name__": "__main__"})
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
    ensure_prebuilt(d)  # kernels-latest covers the OTHER kernels; the
    # update_cublas ext is branch-local and must JIT (UT_PREBUILT only
    # shadows kernels/** .so list — verify ext name is not in bundle)
    from tools.artefact_provenance import (collect_provenance, mark_invalid,
                                           write_artefact)
    from tools.stream_runner import run_streaming
    provenance = collect_provenance(d, extra_config={"config": {
        "branch": args.branch, "expect_commit": args.expect_commit or None,
        "best_algo": args.best_algo}})
    if args.expect_commit and not head.startswith(args.expect_commit):
        art = {"commit": head, "provenance": provenance, "rc": 2}
        mark_invalid(art, "commit-mismatch refusal", root_cause_commit=head)
        print("ARTEFACT:", write_artefact(d, f"dw_algo.{tag}.json", art,
                                          root_cause_commit=head), flush=True)
        return 2

    import colab_pytest
    rc, out = run_streaming([sys.executable, "-m", "pytest", "-q", "--tb=short",
                             *colab_pytest.TEST_MODULES], cwd=d)
    log = os.path.join(d, "results", f"dw_algo.{tag}.pytest.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as fh:
        fh.write(f"commit={head}\nrc={rc}\n\n" + "\n".join(out))
    print(f"STEP_RC: pytest {rc}", flush=True)

    script = os.path.join(d, "scripts", "colab_dw_algo_sweep.py")
    rc, out = run_streaming([sys.executable, script, "--mode", "sweep"], cwd=d,
                            env={**os.environ, "REPO_DIR": d})
    with open(os.path.join(d, "results", f"dw_algo.{tag}.sweep.log"), "w") as fh:
        fh.write("\n".join(out))
    sweep = None
    for line in out:
        if line.startswith("RESULTS_JSON: "):
            sweep = json.loads(line[len("RESULTS_JSON: "):])

    # pick best head algo with acceptable parity (sign disagreement ~ 0)
    best = -1
    if sweep:
        head_rows = [r for r in sweep["shapes"]["head"]["all"]
                     if "median_ms" in r and r["sign_disagree_rate"] < 1e-4]
        if head_rows:
            best = min(head_rows, key=lambda r: r["median_ms"])["algo"]
    print(f"BEST head algo: {best}", flush=True)

    steps = {}
    for name, extra in (("default", {}), ("default-null", {}),
                        ("best", {"PT_DW_ALGO": str(best)})):
        env = {**os.environ, "REPO_DIR": d, "PT_BACKEND": "cublas",
               "PT_UPDATE_BACKEND": "cublas", **extra}
        rc, out = run_streaming([sys.executable, script, "--mode", "step", name],
                                cwd=d, env=env)
        with open(os.path.join(d, "results", f"dw_algo.{tag}.{name}.log"), "w") as fh:
            fh.write("\n".join(out))
        for line in out:
            if line.startswith("RESULTS_JSON: "):
                steps[name] = json.loads(line[len("RESULTS_JSON: "):])

    d0, dn, db = steps.get("default", {}), steps.get("default-null", {}), steps.get("best", {})
    sigma = abs(100 * (dn.get("median_ms", 0) - d0.get("median_ms", 1)) /
                max(d0.get("median_ms", 1), 1e-9)) if d0 and dn else None
    delta = (100 * (db.get("median_ms", 1e9) - d0.get("median_ms", 1)) /
             max(d0.get("median_ms", 1), 1e-9)) if d0 and db else None
    traj = d0.get("checkpoints") == db.get("checkpoints")
    passed = rc == 0 and delta is not None and sigma is not None and \
        delta < -max(sigma, 1.0) and traj
    art = {"commit": head, "tag": tag, "provenance": provenance,
           "verdict": "pass" if passed else "fail", "best_algo": best,
           "pytest_rc": rc, "sigma_null_pct_aa": sigma, "delta_pct": delta,
           "flip_trajectory_match": traj, "sweep": sweep, "steps": steps}
    print("ARTEFACT:", write_artefact(d, f"dw_algo.{tag}.json", art,
                                     root_cause_commit=head), flush=True)
    print(f"DW_ALGO: {'PASS' if passed else 'FAIL'} best={best} delta={delta} "
          f"sigma={sigma} traj={traj}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
