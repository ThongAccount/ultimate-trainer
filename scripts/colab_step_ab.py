#!/usr/bin/env python3
"""colab run one-shot: whole-step A/B — PT_BACKEND=custom vs cublas.

Runs the production 6-layer trainer (train_gigatoken.build_model +
train_step_cudagraph) for N steps under each backend, in SEPARATE
subprocesses (backend is read at import time; also isolates CUDA context
state). Reports median step ms, tok/s, and the loss trajectory.

Prints RESULTS: lines for each backend so the wrapper summary is parseable.
"""
import os
import subprocess
import statistics
import sys

BACKENDS = ("custom", "cublas")
NSTEPS = int(os.environ.get("NSTEPS", "300"))

DRIVER = r'''
import os, sys, time, statistics
os.chdir("/content/ultimate-trainer"); sys.path.insert(0, os.getcwd())
import torch
backend = os.environ["PT_BACKEND"]
gpu = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
assert cap == (7, 5), f"NOT a T4: {gpu} {cap}"
print(f"backend={backend} gpu={gpu}", flush=True)
import train_gigatoken as tg

torch.manual_seed(0)
model = tg.build_model()
B = getattr(tg, "B", 32)
T = tg.SEQ
torch.manual_seed(0)
ids = torch.randint(0, tg.VOCAB, (B, T), device="cuda", dtype=torch.long)
x = ids[:, :-1].contiguous()
y = ids[:, 1:].contiguous()

STEPS = int(os.environ.get("NSTEPS", "300"))
WARM = 5
times, losses = [], []
for step in range(STEPS):
    t0 = time.perf_counter()
    loss = tg.train_step_cudagraph(model, x, y, profile=False)
    lv = float(loss.item())
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    if step >= WARM:
        times.append(t1 - t0)
    losses.append(lv)
    if step in (1, 49, 99, STEPS - 1):
        print(f"  step {step:4d} loss {lv:8.4f} t {(t1-t0)*1000:7.1f} ms", flush=True)
med = statistics.median(times) * 1000
toks = x.numel() / (med / 1000)
print(f"RESULTS: backend={backend} median_step_ms={med:.1f} tok_per_s={toks:.0f} "
      f"loss_50={losses[49]:.4f} loss_100={losses[99]:.4f} loss_final={losses[-1]:.4f} "
      f"steps={STEPS} gpu={gpu}", flush=True)
'''

def main():
    import subprocess as sp
    sp.run(["pip", "install", "-q", "ninja"], capture_output=True)
    d = "/content/ultimate-trainer"
    tok = os.environ.get("GITHUB_TOKEN", "")
    if not os.path.isdir(os.path.join(d, ".git")):
        url = "https://github.com/ThongAccount/ultimate-trainer.git"
        if tok:
            url = url.replace("https://", f"https://{tok}@")
        sp.run(["git", "clone", "--quiet", url, d], check=True)
    sp.run(["git", "-C", d, "fetch", "--quiet", "origin"], check=False)
    sp.run(["git", "-C", d, "reset", "--hard", "--quiet", "origin/chore/speedpass"], check=True)
    head = sp.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()[:10]
    print(f"commit={head}", flush=True)

    open(os.path.join(d, "step_ab_driver.py"), "w").write(DRIVER)
    failures = 0
    for backend in BACKENDS:
        env = {**os.environ, "PT_BACKEND": backend, "NSTEPS": str(NSTEPS)}
        r = sp.run([sys.executable, "step_ab_driver.py"], cwd=d, env=env)
        if r.returncode != 0:
            print(f"RESULTS: backend={backend} FAILED rc={r.returncode}", flush=True)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
