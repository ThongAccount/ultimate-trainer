#!/usr/bin/env python3
"""I2 step-climb root cause on Colab T4.

Measures step_ms on the shipped arm (cublas_cublas) over 300 steps,
logging at checkpoints to identify the climb pattern.
"""
import os, subprocess, sys, json, time

REPO = "https://x-access-token:{}@github.com/ThongAccount/ultimate-trainer.git".format(
    os.environ.get("GITHUB_TOKEN", ""))
BRANCH = os.environ.get("PT_BRANCH", "feat/i2-step-climb")
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

def climb():
    print("=== I2 CLIMB ===", flush=True)
    r = subprocess.run([sys.executable, "-c", """
import torch, sys, time
sys.path.insert(0, ".")
from kernels.packed_ternary import pack_tensor, pack_update as pu
from kernels.packed_ternary import custom_ops as co

# Build a small model step to measure
B, K, N = 16352, 1024, 4096
torch.manual_seed(0)
W = pack_tensor(torch.where(torch.rand(N, K, device="cuda") < 0.5,
                             torch.zeros(N, K, device="cuda"),
                             torch.ones(N, K, device="cuda")).int())
X = (torch.randn(B, K, device="cuda") * 0.1).half()
dY = (torch.randn(B, N, device="cuda") * 0.1).half()
counter = torch.zeros(N, K, dtype=torch.int16, device="cuda")
threshold = 32

# Warmup
for _ in range(5):
    y = co.forward_tc(W, X, K)
    dx = co.backward_dx_tc(W, dY, K)
    co.update_tc_v2(W, counter, X, dY, threshold)
torch.cuda.synchronize()

# Measure 300 steps
steps = [0, 50, 100, 150, 200, 250, 299]
times = []
for step in range(300):
    if step == 150:
        print("--- empty_cache() at step 150 ---", flush=True)
        torch.cuda.empty_cache()
    t0 = time.perf_counter()
    y = co.forward_tc(W, X, K)
    dx = co.backward_dx_tc(W, dY, K)
    co.update_tc_v2(W, counter, X, dY, threshold)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000
    if step in steps:
        print(f"step {step}: {ms:.3f} ms", flush=True)
        times.append((step, ms))

print("\\n=== CLIMB SUMMARY ===")
for step, ms in times:
    print(f"step {step}: {ms:.3f} ms")
"""], cwd=CLONE, capture_output=True, text=True, env={**os.environ, "UT_PREBUILT": "0"})
    print(r.stdout[-3000:] if r.stdout else "", flush=True)
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
    rc = climb()
    results["steps"]["climb"] = rc
    print(json.dumps(results, indent=2), flush=True)
    return rc

if __name__ == "__main__":
    sys.exit(main())