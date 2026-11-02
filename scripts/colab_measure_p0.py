#!/usr/bin/env python3
"""colab run one-shot: P0 measurement bundle (Researcher list A items #1-#4, #9, #18).

One T4 session, master commit pinned via --expect-commit. Each child runs as a
subprocess with its own PT_BACKEND env (backend is read at import time), like
colab_step_ab.py. No code changes — measurement only.

Children:
  shipped-profile   cublas_cublas arm, torch.profiler(CPU+CUDA, memory) over
                    steps 20..120 + per-step trajectory -> item #1 (attribute the
                    510->575 ms within-run climb; kernel-sum vs wall step)
  shipped-climb     cublas_cublas arm, 200 steps, optional empty_cache at a
                    given step (--cache 1) -> item #2 (fragmentation falsifier on
                    the SHIPPED arm; custom arm already ruled out 2026-11-02)
  budget-ab         same-VM B=0 vs B=5 interleaved per-step medians on the
                    shipped arm -> item #3 (price of flip counting)
  custom-glue       custom arm, torch.profiler over steps 10..50 -> item #9
                    (attribute the ~1.2 s glue: embed/norm/gelu/scale/CE/opt)
  fc1fwd            custom arm, ab_median A/A + absolute median of fc1 forward
                    at B=16384 -> item #18 (resolve the 4.15 vs 7.33 ms conflict
                    under one protocol)

Every child also samples nvidia-smi clocks/power every 500 ms -> item #4
(clock/power confound check; rides along, free).

rc: 0 = all children ran (individual FAIL lines are data), 2 = commit mismatch.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

CHILD = r'''
import json, os, statistics, subprocess, sys, time
REPO = os.environ.get("REPO_DIR", "/content/ultimate-trainer")
os.chdir(REPO); sys.path.insert(0, os.getcwd())
name = sys.argv[1]
import torch
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (7, 5), "NOT a T4"

# item #4: clock/power sampler riding along
clocks_path = os.path.join(REPO, "results", f"clocks.{name}.csv")
clock_proc = subprocess.Popen(
    ["nvidia-smi", "--query-gpu=timestamp,clocks.sm,clocks.max.sm,power.draw,"
     "temperature.gpu,utilization.gpu", "--format=csv", "-lms", "500"],
    stdout=open(clocks_path, "w"), stderr=subprocess.DEVNULL)

def batches():
    B, T = 32, 512
    path = "data/tokenized/samples_512.pt"
    samples = torch.load(path, map_location="cpu", weights_only=False)
    flat = torch.cat([s.reshape(-1).long() for s in samples])
    need = B * T
    out, i = [], 0
    while len(out) < 3 and (i + 1) * need <= flat.numel():
        ids = flat[i * need:(i + 1) * need].view(B, T).cuda()
        out.append((ids[:, :-1].contiguous(), ids.contiguous()))
        i += 1
    return out

import train_gigatoken as tg
torch.manual_seed(0)

def step_times(model, bs, steps, warm, cache_at=-1):
    times, losses = [], []
    for s in range(steps):
        x, y = bs[s % len(bs)]
        t0 = time.perf_counter()
        loss = tg.train_step_cudagraph(model, x, y, profile=False)
        lv = float(loss.item())
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        losses.append(lv)
        if s == cache_at:
            torch.cuda.empty_cache()
    return times, losses

res = {"child": name, "gpu": torch.cuda.get_device_name(0)}
try:
    if name in ("shipped-profile", "shipped-climb", "budget-ab"):
        os.environ["PT_BACKEND"] = "cublas"
        os.environ["PT_UPDATE_BACKEND"] = "cublas"
    bs = batches()

    if name == "shipped-profile":
        model = tg.build_model()
        step_times(model, bs, 20, 0)  # warmup
        from torch.profiler import profile, ProfilerActivity
        times, losses = [], []
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     profile_memory=True) as prof:
            for s in range(100):
                x, y = bs[s % len(bs)]
                t0 = time.perf_counter()
                loss = tg.train_step_cudagraph(model, x, y, profile=False)
                lv = float(loss.item())
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
                losses.append(lv)
        tab = prof.key_averages().table(sort_by="self_device_time_total", row_limit=30)
        rows = []
        for e in prof.key_averages():
            rows.append({"key": e.key[:120],
                         "self_cuda_ms": round(e.self_device_time_total / 1000, 3),
                         "count": e.count})
        rows.sort(key=lambda r: -r["self_cuda_ms"])
        dev_sum = sum(r["self_cuda_ms"] for r in rows)
        res.update({
            "step_ms_checkpoints": {str(i): round(times[i], 1) for i in
                                    (0, 24, 49, 74, 99) if i < len(times)},
            "step_ms_median": round(statistics.median(times), 2),
            "loss_first_last": [losses[0], losses[-1]],
            "device_kernel_sum_ms": round(dev_sum, 1),
            "top_kernels": rows[:25],
            "profiler_table_head": tab.splitlines()[:12],
        })
    elif name == "shipped-climb":
        cache = int(os.environ.get("CACHE", "0"))
        at = int(os.environ.get("CACHE_AT", "150"))
        model = tg.build_model()
        times, losses = step_times(model, bs, 200, 10, cache_at=at if cache else -1)
        res.update({
            "cache_at": at if cache else None,
            "step_ms_checkpoints": {str(i): round(times[i], 1) for i in
                                    (1, 49, 99, 149, 150, 151, 199)},
            "median_first50": round(statistics.median(times[10:60]), 2),
            "median_last50": round(statistics.median(times[150:200]), 2),
            "loss_first_last": [losses[0], losses[-1]],
        })
    elif name == "budget-ab":
        # same-VM B=0 vs B=5, interleaved per step (item #3)
        tg.FLIP_BUDGET = 0
        m0 = tg.build_model()
        tg.FLIP_BUDGET = 5
        m5 = tg.build_model()
        tg.FLIP_BUDGET = 0
        step_times(m0, bs, 10, 0); step_times(m5, bs, 10, 0)  # warmup both
        t0s, t5s = [], []
        for s in range(60):
            x, y = bs[s % len(bs)]
            t = time.perf_counter(); tg.train_step_cudagraph(m0, x, y); torch.cuda.synchronize()
            t0s.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter(); tg.train_step_cudagraph(m5, x, y); torch.cuda.synchronize()
            t5s.append((time.perf_counter() - t) * 1000)
        flips5 = sum(int(l.flip_count.sum().item()) for l in list(m5.fc1s) + list(m5.fc2s) + [m5.head])
        res.update({
            "b0_median_ms": round(statistics.median(t0s), 2),
            "b5_median_ms": round(statistics.median(t5s), 2),
            "delta_ms": round(statistics.median(t5s) - statistics.median(t0s), 2),
            "delta_pct": round(100 * (statistics.median(t5s) - statistics.median(t0s)) / statistics.median(t0s), 3),
            "b5_total_flips_after_70_steps": flips5,
        })
    elif name == "custom-glue":
        model = tg.build_model()
        step_times(model, bs, 10, 0)
        from torch.profiler import profile, ProfilerActivity
        times = []
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     profile_memory=True) as prof:
            for s in range(40):
                x, y = bs[s % len(bs)]
                t0 = time.perf_counter()
                tg.train_step_cudagraph(model, x, y, profile=False)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
        rows = []
        for e in prof.key_averages():
            rows.append({"key": e.key[:120],
                         "self_cuda_ms": round(e.self_device_time_total / 1000, 3),
                         "self_cpu_ms": round(e.self_cpu_time_total / 1000, 3),
                         "count": e.count})
        rows.sort(key=lambda r: -r["self_cuda_ms"])
        res.update({
            "step_ms_median": round(statistics.median(times), 2),
            "device_kernel_sum_ms": round(sum(r["self_cuda_ms"] for r in rows), 1),
            "top_kernels": rows[:25],
            "top_cpu": sorted(rows, key=lambda r: -r["self_cpu_ms"])[:10],
        })
    elif name == "fc1fwd":
        sys.path.insert(0, REPO)
        from tests.bench_protocol import ab_median
        from kernels.packed_ternary import custom_ops
        model = tg.build_model()
        layer = model.fc1s[0]
        W = layer.W_packed
        K = layer.in_features
        X = torch.randn(16384, K, device="cuda", dtype=torch.float16)
        def run():
            custom_ops.forward_tc(W, X, K)
        r = ab_median(run, run, null_arm=True)
        res.update({
            "fc1_fwd_median_ms": round(r.a_ms, 3),  # _time_once returns ms
            "sigma_null_pct": round(r.sigma_null_pct, 4) if r.sigma_null_pct is not None else None,
            "B": 16384, "K": K, "N": W.size(0),
        })
    else:
        raise SystemExit(f"unknown child {name}")
finally:
    clock_proc.terminate()

print("RESULTS_JSON: " + json.dumps(res), flush=True)
'''

JOBS = [
    # (child, extra env)
    ("shipped-profile", {"PT_BACKEND": "cublas", "PT_UPDATE_BACKEND": "cublas"}),
    ("shipped-climb", {"PT_BACKEND": "cublas", "PT_UPDATE_BACKEND": "cublas",
                       "CACHE": "0", "CACHE_AT": "150"}),
    ("shipped-climb-cache", {"PT_BACKEND": "cublas", "PT_UPDATE_BACKEND": "cublas",
                             "CACHE": "1", "CACHE_AT": "150"}),
    ("budget-ab", {"PT_BACKEND": "cublas", "PT_UPDATE_BACKEND": "cublas"}),
    ("custom-glue", {"PT_BACKEND": "custom", "PT_UPDATE_BACKEND": "custom"}),
    ("fc1fwd", {"PT_BACKEND": "custom", "PT_UPDATE_BACKEND": "custom"}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/content/ultimate-trainer")
    ap.add_argument("--child", default="")
    ap.add_argument("--expect-commit", default=os.environ.get("PT_EXPECT_COMMIT", ""))
    ap.add_argument("--tag", default="")
    args, _unknown = ap.parse_known_args()

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
    subprocess.run(["git", "-C", d, "reset", "--hard", "--quiet", "origin/master"],
                   check=True)
    head = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    tag = args.tag or head[:8]
    print(f"commit={head} tag={tag}", flush=True)
    if d not in sys.path:
        sys.path.insert(0, d)
    from tools.fetch_prebuilt import ensure_prebuilt
    ensure_prebuilt(d)
    from tools.artefact_provenance import collect_provenance, mark_invalid, write_artefact
    from tools.stream_runner import run_streaming
    provenance = collect_provenance(d, extra_config={"config": {
        "expect_commit": args.expect_commit or None, "children": [j[0] for j in JOBS]}})

    if args.expect_commit and not head.startswith(args.expect_commit):
        print(f"COMMIT-MISMATCH: HEAD={head[:10]} != expect={args.expect_commit}",
              flush=True)
        art = {"commit": head, "provenance": provenance, "rc": 2}
        mark_invalid(art, "commit-mismatch refusal", root_cause_commit=head)
        print("ARTEFACT:", write_artefact(d, f"measure_p0.{tag}.json", art,
                                         root_cause_commit=head), flush=True)
        return 2

    results = {}
    for name, extra_env in JOBS:
        env = {**os.environ, "REPO_DIR": d, **extra_env}
        print(f"=== CHILD {name} ===", flush=True)
        rc, out = run_streaming([sys.executable, os.path.abspath(__file__),
                                 "--child", name], cwd=d, env=env)
        log = os.path.join(d, "results", f"measure_p0.{tag}.{name}.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w") as fh:
            fh.write(f"commit={head}\nchild={name}\nrc={rc}\n\n" + "\n".join(out))
        print(f"CHILD_RC: {name} {rc}", flush=True)
        for line in out:
            if line.startswith("RESULTS_JSON: "):
                results[name] = json.loads(line[len("RESULTS_JSON: "):])

    art = {"commit": head, "tag": tag, "provenance": provenance,
           "verdict": "measured", "children": results}
    path = write_artefact(d, f"measure_p0.{tag}.json", art, root_cause_commit=head)
    print("ARTEFACT:", path, flush=True)
    print("MEASURE_P0: DONE " + json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "top_kernels"}
         for k, v in results.items()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
