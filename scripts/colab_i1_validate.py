#!/usr/bin/env python3
"""I1 fused CE+scale validation on Colab T4.

Clones the branch, runs smoke + pytest + parity + A/B whole-step comparison.
UT_PREBUILT=0 required (branch kernel must not be shadowed by master bundle).
"""
import os, subprocess, sys, json, time

REPO = "https://x-access-token:{}@github.com/ThongAccount/ultimate-trainer.git".format(
    os.environ.get("GITHUB_TOKEN", ""))
BRANCH = os.environ.get("PT_BRANCH", "feat/i1-fused-ce-scale")
EXPECT = os.environ.get("PT_EXPECT_COMMIT", "")
CLONE = "/content/ultimate-trainer"

def clone():
    if os.path.exists(CLONE):
        subprocess.run(["git", "-C", CLONE, "fetch", "origin", BRANCH], check=True)
        subprocess.run(["git", "-C", CLONE, "checkout", BRANCH], check=True)
        subprocess.run(["git", "-C", CLONE, "pull", "origin", BRANCH], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, CLONE], check=True)
    os.environ["UT_PREBUILT"] = "0"
    commit = subprocess.check_output(["git", "-C", CLONE, "rev-parse", "HEAD"]).decode().strip()
    print(f"commit={commit}", flush=True)
    if EXPECT and not commit.startswith(EXPECT):
        print(f"COMMIT-MISMATCH: {commit} != {EXPECT}", flush=True)
        sys.exit(2)
    return commit

def install_deps():
    print("=== INSTALL DEPS ===", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja"], check=True)
    print("ninja installed", flush=True)

def smoke():
    print("=== SMOKE ===", flush=True)
    r = subprocess.run([sys.executable, "tests/probe_fwd_spill.py", "--probe", "fwdspill", "--skip-timing"],
                       cwd=CLONE, capture_output=True, text=True, env={**os.environ, "UT_PREBUILT": "0"})
    print(r.stdout[-2000:] if r.stdout else "", flush=True)
    print(r.stderr[-1000:] if r.stderr else "", flush=True)
    return r.returncode

def pytest():
    print("=== PYTEST ===", flush=True)
    r = subprocess.run([sys.executable, "colab_pytest.py"],
                       cwd=CLONE, capture_output=True, text=True,
                       env={**os.environ, "UT_PREBUILT": "0", "BRANCH": BRANCH})
    print(r.stdout[-2000:] if r.stdout else "", flush=True)
    print(r.stderr[-1000:] if r.stderr else "", flush=True)
    return r.returncode

def parity():
    print("=== I1 PARITY ===", flush=True)
    r = subprocess.run([sys.executable, "-c", """
import torch, sys
sys.path.insert(0, ".")
from train_gigatoken import _fused_ce_scale

# Test fused CE+scale vs eager CE
B, T, VOCAB = 32, 511, 50272
torch.manual_seed(0)
logits = torch.randn(B, T, VOCAB, device="cuda", dtype=torch.float16)
targets = torch.randint(0, VOCAB, (B, T), device="cuda")

# Eager CE
loss_eager = torch.nn.functional.cross_entropy(
    logits.view(-1, VOCAB), targets.view(-1), reduction='mean')

# Fused CE+scale
loss_fused = _fused_ce_scale(logits, targets, VOCAB)

print(f"eager: {loss_eager.item():.6f}")
print(f"fused: {loss_fused.item():.6f}")
print(f"diff: {abs(loss_eager.item() - loss_fused.item()):.6e}")

# Bit-exact parity (within fp16 tolerance)
assert abs(loss_eager.item() - loss_fused.item()) < 1e-3, "loss parity FAIL"
print("PARITY PASS", flush=True)
"""], cwd=CLONE, capture_output=True, text=True, env={**os.environ, "UT_PREBUILT": "0"})
    print(r.stdout[-2000:] if r.stdout else "", flush=True)
    print(r.stderr[-1000:] if r.stderr else "", flush=True)
    return r.returncode

def ab():
    print("=== I1 A/B ===", flush=True)
    r = subprocess.run([sys.executable, "-c", """
import torch, sys, time
sys.path.insert(0, ".")
from train_gigatoken import _fused_ce_scale

B, T, VOCAB = 32, 511, 50272
torch.manual_seed(0)
logits = torch.randn(B, T, VOCAB, device="cuda", dtype=torch.float16)
targets = torch.randint(0, VOCAB, (B, T), device="cuda")

# Warmup
for _ in range(5):
    loss = _fused_ce_scale(logits, targets, VOCAB)
torch.cuda.synchronize()

# Measure eager CE
times_eager = []
for _ in range(10):
    t0 = time.perf_counter()
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, VOCAB), targets.view(-1), reduction='mean')
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times_eager.append((t1 - t0) * 1000)

# Measure fused CE+scale
times_fused = []
for _ in range(10):
    t0 = time.perf_counter()
    loss = _fused_ce_scale(logits, targets, VOCAB)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times_fused.append((t1 - t0) * 1000)

import statistics
print(f"eager CE: {statistics.median(times_eager):.3f} ms")
print(f"fused CE+scale: {statistics.median(times_fused):.3f} ms")
print(f"delta: {statistics.median(times_fused) - statistics.median(times_eager):.3f} ms")
"""], cwd=CLONE, capture_output=True, text=True, env={**os.environ, "UT_PREBUILT": "0"})
    print(r.stdout[-2000:] if r.stdout else "", flush=True)
    print(r.stderr[-1000:] if r.stderr else "", flush=True)
    return r.returncode

def main():
    commit = clone()
    install_deps()
    results = {"commit": commit, "branch": BRANCH, "steps": {}}
    rc = smoke()
    results["steps"]["smoke"] = rc
    if rc != 0:
        print("SMOKE FAILED — aborting", flush=True)
        print(json.dumps(results, indent=2), flush=True)
        return 1
    rc = pytest()
    results["steps"]["pytest"] = rc
    if rc != 0:
        print("PYTEST FAILED — aborting", flush=True)
        print(json.dumps(results, indent=2), flush=True)
        return 1
    rc = parity()
    results["steps"]["parity"] = rc
    if rc != 0:
        print("PARITY FAILED — aborting", flush=True)
        print(json.dumps(results, indent=2), flush=True)
        return 1
    rc = ab()
    results["steps"]["ab"] = rc
    print(json.dumps(results, indent=2), flush=True)
    return rc

if __name__ == "__main__":
    sys.exit(main())