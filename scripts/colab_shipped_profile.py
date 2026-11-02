#!/usr/bin/env python3
"""Shipped-path P0 profiling (hyperopt audit 2026-11-02, item 2).

Runs the SHIPPED arm (PT_BACKEND=cublas, PT_UPDATE_BACKEND=cublas — set by the
caller's env) for 300 steps on synthetic data, twice: flip budget B=0 and B=5
(PT_FLIP_BUDGET). Records per-step wall ms (perf_counter + synchronize, same
discipline as bench_protocol), injects torch.cuda.empty_cache() at step 150 of
the B=5 arm to test whether the within-run climb (510 -> ~575 ms observed in
B05prod300) is allocator fragmentation, and profiles steps 100-200 with
torch.profiler to attribute the step by kernel.

Usage (on VM):
    PT_BACKEND=cublas PT_UPDATE_BACKEND=cublas python scripts/colab_shipped_profile.py \
        --repo /content/ultimate-trainer --expect-commit <sha>
Prints RESULTS JSON. Artefact -> results/shipped_profile.<tag>.json via caller.
"""
import argparse
import json
import os
import sys
import time


def run_arm(repo, flip_budget, steps, profile_lo, profile_hi, empty_cache_at):
    os.environ["PT_FLIP_BUDGET"] = str(flip_budget)
    for mod in [m for m in list(sys.modules) if m == "train_gigatoken" or m.startswith("kernels")]:
        del sys.modules[mod]
    sys.path.insert(0, repo)
    import torch
    import train_gigatoken as tg

    model = tg.build_model()
    model.train()
    torch.manual_seed(0)
    B, T = tg.B, tg.SEQ
    ids = torch.randint(0, tg.VOCAB, (B, T), device="cuda", dtype=torch.long)
    x, y = ids[:, :-1].contiguous(), ids.contiguous()

    times, losses = [], []
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA])
    for step in range(steps):
        if profile_lo <= step < profile_hi:
            prof.start()
        t0 = time.perf_counter()
        loss = tg.train_step_cudagraph(model, x, y, profile=False)
        torch.cuda.synchronize()
        if profile_lo <= step < profile_hi:
            prof.stop()
        times.append((time.perf_counter() - t0) * 1000.0)
        losses.append(float(loss.item()))
        if step == empty_cache_at:
            torch.cuda.empty_cache()
            print(f"[arm B={flip_budget}] empty_cache at step {step}", flush=True)
    return times, losses, prof


def summarize_profiler(prof):
    try:
        evts = prof.key_averages()
        rows = sorted(
            ((e.self_device_time_total, e.key) for e in evts if e.self_device_time_total > 0),
            reverse=True)[:15]
        return [{"kernel": k, "self_cuda_ms_total": round(v / 1000.0, 2)} for v, k in rows]
    except Exception as e:  # profiler schema differences across torch versions
        return [{"error": str(e)}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/content/ultimate-trainer")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--expect-commit", default=os.environ.get("PT_EXPECT_COMMIT", ""))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import subprocess
    head = subprocess.run(["git", "-C", args.repo, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    print(f"commit={head}", flush=True)
    if args.expect_commit and not head.startswith(args.expect_commit):
        print(f"COMMIT-MISMATCH: HEAD={head[:10]} != expect={args.expect_commit}", flush=True)
        return 2

    import torch
    print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} "
          f"PT_BACKEND={os.environ.get('PT_BACKEND')} "
          f"PT_UPDATE_BACKEND={os.environ.get('PT_UPDATE_BACKEND')}", flush=True)

    out = {"commit": head, "steps": args.steps,
           "backend": {"PT_BACKEND": os.environ.get("PT_BACKEND"),
                       "PT_UPDATE_BACKEND": os.environ.get("PT_UPDATE_BACKEND")}}
    for budget in (0, 5):
        times, losses, prof = run_arm(args.repo, budget, args.steps, 100, 200, 150)
        import statistics
        seg = lambda a, b: [round(t, 1) for t in times[a:b]]
        out[f"B{budget}"] = {
            "median_ms": round(statistics.median(times), 2),
            "p10_ms": round(sorted(times)[args.steps // 10], 2),
            "step1_ms": round(times[0], 1), "step50_ms": round(times[49], 1),
            "step99_ms": round(times[99], 1), "step150_ms": round(times[150], 1),
            "step299_ms": round(times[-1], 1),
            "first50": seg(0, 50), "last50": seg(-50, args.steps),
            "loss_1": losses[0], "loss_final": losses[-1],
            "profiler_top": summarize_profiler(prof),
        }
        print(f"B{budget}: median={out[f'B{budget}']['median_ms']}ms "
              f"step1={times[0]:.1f} step99={times[99]:.1f} step299={times[-1]:.1f} "
              f"loss {losses[0]:.4f}->{losses[-1]:.4f}", flush=True)

    print("\nRESULTS JSON:", json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
