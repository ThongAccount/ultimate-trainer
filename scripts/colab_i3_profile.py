#!/usr/bin/env python3
"""I3 custom-arm glue attribution on Colab T4.

Profiles the custom arm with torch profiler to identify the ~1.2s of
non-kernel "glue" (Python dispatch, tensor movement, etc.).
"""
import os, subprocess, sys, json, time

REPO = "https://x-access-token:{}@github.com/ThongAccount/ultimate-trainer.git".format(
    os.environ.get("GITHUB_TOKEN", ""))
BRANCH = os.environ.get("PT_BRANCH", "feat/i3-glue-attribution")
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

def profile():
    print("=== I3 PROFILE ===", flush=True)
    r = subprocess.run([sys.executable, "-c", """
import torch, sys, time
sys.path.insert(0, ".")
from torch.profiler import profile, ProfilerActivity
from kernels.packed_ternary import pack_tensor, pack_update as pu
from kernels.packed_ternary import custom_ops as co

# Build a small model step to profile
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
for _ in range(3):
    y = co.forward_tc(W, X, K)
    dx = co.backward_dx_tc(W, dY, K)
    co.update_tc_v2(W, counter, X, dY, threshold)

torch.cuda.synchronize()

# Profile
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, with_stack=True) as prof:
    for _ in range(10):
        y = co.forward_tc(W, X, K)
        dx = co.backward_dx_tc(W, dY, K)
        co.update_tc_v2(W, counter, X, dY, threshold)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
print("\\n=== TOP 10 BY SELF CUDA ===")
for evt in prof.key_averages():
    if evt.self_cuda_time_total > 0:
        print(f"{evt.key}: {evt.self_cuda_time_total/1000:.3f} ms x{evt.count}")
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
    rc = profile()
    results["steps"]["profile"] = rc
    print(json.dumps(results, indent=2), flush=True)
    return rc

if __name__ == "__main__":
    sys.exit(main())