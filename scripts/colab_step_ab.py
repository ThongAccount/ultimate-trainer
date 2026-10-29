#!/usr/bin/env python3
"""colab run one-shot: whole-step A/B — PT_BACKEND x PT_UPDATE_BACKEND.

Runs the production 6-layer trainer (train_gigatoken.build_model +
train_step_cudagraph) for N steps under each backend configuration, in
SEPARATE subprocesses (backend is read at import time; also isolates CUDA
context state). Reports median step ms, tok/s, and the loss trajectory.

Three arms, which together reproduce the campaign's whole-step trajectory:

  arm                     PT_BACKEND   PT_UPDATE_BACKEND   expected (session 31)
  custom_custom           custom       custom              ~6,619 ms  (baseline)
  cublas_custom           cublas       custom              ~1,696 ms  (fwd/dX fix)
  cublas_cublas           cublas       cublas              ~519 ms    (shipped)

The historical 6,619 ms figure was measured on a Modal T4 with a different
torch build and power state, so the ACCEPTANCE-GRADE comparison is the
same-VM pair (custom_custom -> cublas_cublas), not the cross-VM ratio.

Data: DATA_MODE=real trains on the committed FineWeb tokenized samples
(data/tokenized/samples_512.pt, custom BPE, vocab 32,768 < the model's
50,272). DATA_MODE=synthetic reproduces the earlier protocol (a single fixed
random-token batch), under which the loss NaNs deterministically at step ~99
— a synthetic-data artefact, not a backend divergence.

Prints RESULTS: lines for each backend so the wrapper summary is parseable.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

# (PT_BACKEND, PT_UPDATE_BACKEND, label)
ARMS = (
    ("custom", "custom", "custom_custom"),
    ("cublas", "custom", "cublas_custom"),
    ("cublas", "cublas", "cublas_cublas"),
)

# Researcher's ship bar: >= 5% end-to-end step-time reduction.
SHIP_BAR_PCT = 5.0

# Cross-VM reference only (Modal T4, different torch/python/power state).
HISTORICAL_BASELINE_MS = 6619.0

DRIVER = r'''
import os, sys, time, statistics
REPO = os.environ.get("REPO_DIR", "/content/ultimate-trainer")
os.chdir(REPO); sys.path.insert(0, os.getcwd())
import torch
backend = os.environ["PT_BACKEND"]
update_backend = os.environ.get("PT_UPDATE_BACKEND", "custom")
label = os.environ.get("ARM_LABEL", f"{backend}_{update_backend}")
data_mode = os.environ.get("DATA_MODE", "real")
gpu = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
assert cap == (7, 5), f"NOT a T4: {gpu} {cap}"
print(f"backend={backend} update_backend={update_backend} label={label} "
      f'data={data_mode} gpu="{gpu}"', flush=True)
import train_gigatoken as tg

torch.manual_seed(0)
model = tg.build_model()
B = getattr(tg, "B", 32)
T = tg.SEQ

def real_batches():
    """Deterministic batches from the committed FineWeb tokenized samples."""
    path = "data/tokenized/samples_512.pt"
    samples = torch.load(path, map_location="cpu", weights_only=False)
    flat = torch.cat([s.reshape(-1).long() for s in samples])
    need = B * T
    out = []
    i = 0
    while len(out) < 3 and (i + 1) * need <= flat.numel():
        ids = flat[i * need:(i + 1) * need].view(B, T).cuda()
        out.append((ids[:, :-1].contiguous(), ids.contiguous()))
        i += 1
    assert out, "not enough tokens for one batch"
    return out

def synthetic_batch():
    torch.manual_seed(0)
    ids = torch.randint(0, tg.VOCAB, (B, T), device="cuda", dtype=torch.long)
    return [(ids[:, :-1].contiguous(), ids.contiguous())]

batches = real_batches() if data_mode == "real" else synthetic_batch()
print(f"batches={len(batches)} tokens_per_batch={B * T}", flush=True)

STEPS = int(os.environ.get("NSTEPS", "300"))
WARM = 5
times, all_times, losses = [], [], []
nan_seen = False
for step in range(STEPS):
    x, y = batches[step % len(batches)]
    t0 = time.perf_counter()
    loss = tg.train_step_cudagraph(model, x, y, profile=False)
    lv = float(loss.item())
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    all_times.append(t1 - t0)
    if step >= WARM:
        times.append(t1 - t0)
    losses.append(lv)
    if lv != lv:
        nan_seen = True
    if step in (1, 49, 99, STEPS - 1):
        print(f"  step {step:4d} loss {lv:8.4f} t {(t1-t0)*1000:7.1f} ms", flush=True)
med = statistics.median(times if times else all_times) * 1000
toks = x.numel() / (med / 1000)


def loss_at(i):
    return f"{losses[i]:.4f}" if i < len(losses) else "nan"


print(f"RESULTS: backend={backend} update_backend={update_backend} label={label} "
      f"median_step_ms={med:.1f} tok_per_s={toks:.0f} "
      f"loss_1={loss_at(1)} loss_50={loss_at(50)} loss_100={loss_at(100)} "
      f"loss_final={losses[-1]:.4f} nan={int(nan_seen)} steps={STEPS} "
      f'data={data_mode} gpu="{gpu}"', flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/content/ultimate-trainer")
    ap.add_argument("--no-clone", action="store_true",
                    help="reuse an existing clone (the suite runner clones once)")
    ap.add_argument("--nsteps", type=int, default=300)
    ap.add_argument("--data", default="real", choices=["real", "synthetic"])
    ap.add_argument("--backends", default=",".join(lbl for _b, _u, lbl in ARMS))
    ap.add_argument("--expect-commit", default=os.environ.get("PT_EXPECT_COMMIT", ""),
                    help="rc=2 unless the clone HEAD matches this sha (prefix ok); "
                         "stale-clone guard (ut-fb-10 root cause #1)")
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
        # check=True: a failed fetch must abort loudly, never silently reset
        # to a stale origin/master (the exact ut-fb-10 burn).
        subprocess.run(["git", "-C", d, "fetch", "--quiet", "origin"], check=True)
        subprocess.run(["git", "-C", d, "reset", "--hard", "--quiet", "origin/master"], check=True)
    head = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()  # full 40-char
    if d not in sys.path:
        sys.path.insert(0, d)
    from tools.artefact_provenance import collect_provenance, write_artefact
    # collect BEFORE step_ab_driver.py is written into the repo root (untracked
    # file would flip git_dirty)
    provenance = collect_provenance(d, extra_config={"config": {
        "nsteps": args.nsteps, "data": args.data, "backends": args.backends,
        "ship_bar_pct": SHIP_BAR_PCT,
        "historical_baseline_ms": HISTORICAL_BASELINE_MS,
        "expect_commit": args.expect_commit or None}})
    print(f"commit={head}", flush=True)

    # Stale-clone guard: rc=2 + quarantined artefact BEFORE the driver write
    # and any GPU arm runs (ut-fb-10 root cause #1).
    expect = args.expect_commit
    if expect and not head.startswith(expect):
        print(f"COMMIT-MISMATCH: clone HEAD={head[:10]} does not match "
              f"--expect-commit={expect}; refusing to run (stale-clone guard, ut-fb-10)",
              flush=True)
        artefact = {"commit": head, "provenance": provenance, "rc": 2,
                    "config": {"expect_commit": expect}}
        path = write_artefact(d, "whole_step_ab.json", artefact, root_cause_commit=head)
        print(f"ARTEFACT: {path}", flush=True)
        return 2

    with open(os.path.join(d, "step_ab_driver.py"), "w") as fh:
        fh.write(DRIVER)

    selected = [a for a in ARMS if a[2] in args.backends.split(",")]
    by = {}
    failures = 0
    for backend, update_backend, label in selected:
        env = {**os.environ, "REPO_DIR": d, "PT_BACKEND": backend,
               "PT_UPDATE_BACKEND": update_backend, "ARM_LABEL": label,
               "NSTEPS": str(args.nsteps), "DATA_MODE": args.data}
        print(f"--- arm {label} (PT_BACKEND={backend} "
              f"PT_UPDATE_BACKEND={update_backend}) ---", flush=True)
        sys.path.insert(0, d)
        from tools.stream_runner import run_streaming
        rc, out = run_streaming([sys.executable, "step_ab_driver.py"], cwd=d, env=env)
        for line in out:
            if line.startswith("RESULTS:"):
                rec = {}
                for tok in shlex.split(line)[1:]:  # quoted gpu keeps "Tesla T4"
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        rec[k] = v
                try:
                    rec["median_step_ms"] = float(rec.get("median_step_ms", "nan"))
                except ValueError:
                    pass
                if "label" in rec:
                    by[rec["label"]] = rec
        if rc != 0:
            print(f"RESULTS: backend={backend} update_backend={update_backend} "
                  f"label={label} FAILED rc={rc}", flush=True)
            failures += 1

    # ── verdict ────────────────────────────────────────────────────────────
    have = all(isinstance(by.get(lbl, {}).get("median_step_ms"), float)
               for lbl in ("custom_custom", "cublas_cublas"))
    if have:
        base = by["custom_custom"]["median_step_ms"]
        ship = by["cublas_cublas"]["median_step_ms"]
        reduction = (base - ship) / base * 100.0
        speedup = base / ship
        hist = HISTORICAL_BASELINE_MS / ship
        print(f"AB_VERDICT: ship_bar_pct={SHIP_BAR_PCT:.1f} "
              f"baseline_ms={base:.1f} shipped_ms={ship:.1f} "
              f"reduction_pct={reduction:.2f} speedup_same_vm={speedup:.2f}x "
              f"speedup_vs_historical_6619={hist:.2f}x "
              f"passed={reduction >= SHIP_BAR_PCT}", flush=True)
        if reduction < SHIP_BAR_PCT:
            failures += 1
    else:
        print("AB_VERDICT: incomplete (missing baseline or shipped arm) "
              "passed=False", flush=True)
        failures += 1

    # Persist on the VM as well as printing: under a Jupyter kernel the
    # stdout path is fragile, and the numbers must survive it.
    outdir = os.path.join(d, "results")
    os.makedirs(outdir, exist_ok=True)
    artefact = {"commit": head, "provenance": provenance, "nsteps": args.nsteps,
                "data": args.data, "ship_bar_pct": SHIP_BAR_PCT,
                "historical_baseline_ms": HISTORICAL_BASELINE_MS,
                "arms": by, "failures": failures}
    path = write_artefact(d, "whole_step_ab.json", artefact, root_cause_commit=head)
    print(f"ARTEFACT: {path}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
