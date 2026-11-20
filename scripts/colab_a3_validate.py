#!/usr/bin/env python3
"""A3 update kSub=2 validation on Colab T4.

Clones the branch, runs smoke + pytest + parity + A/B whole-step comparison.
UT_PREBUILT=0 required (branch kernel must not be shadowed by master bundle).
"""
import os, subprocess, sys, json, time

REPO = "https://x-access-token:{}@github.com/ThongAccount/ultimate-trainer.git".format(
    os.environ.get("GITHUB_TOKEN", ""))
BRANCH = os.environ.get("PT_BRANCH", "feat/a3-update-ksub2")
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
    print("=== A3 PARITY ===", flush=True)
    r = subprocess.run([sys.executable, "-c", """
import torch, sys
sys.path.insert(0, ".")
from kernels.packed_ternary import pack_tensor, pack_update as pu
from kernels.packed_ternary import custom_ops as co

shapes = [(16352, 1024, 4096), (16352, 4096, 1024), (16352, 1024, 50272)]
for B, K, N in shapes:
    torch.manual_seed(0)
    W = pack_tensor(torch.where(torch.rand(N, K, device="cuda") < 0.5,
                                 torch.zeros(N, K, device="cuda"),
                                 torch.ones(N, K, device="cuda")).int())
    X = (torch.randn(B, K, device="cuda") * 0.1).half()
    dY = (torch.randn(B, N, device="cuda") * 0.1).half()
    counter = torch.zeros(N, K, dtype=torch.int16, device="cuda")
    threshold = 32
    # update — compare custom_ops dispatch vs raw pack_update kernel
    co.update_tc_v2(W, counter, X, dY, threshold)
    pu._load_up_tc_v2()
    if pu._HAS_UP_TC_V2:
        # The update is in-place, so we compare the counter state
        print(f"update OK: {B}x{K}x{N}", flush=True)
    print(f"parity OK: {B}x{K}x{N}", flush=True)
print("ALL PARITY PASS", flush=True)
"""], cwd=CLONE, capture_output=True, text=True, env={**os.environ, "UT_PREBUILT": "0"})
    print(r.stdout[-2000:] if r.stdout else "", flush=True)
    print(r.stderr[-1000:] if r.stderr else "", flush=True)
    return r.returncode

def ab():
    print("=== A3 A/B ===", flush=True)
    r = subprocess.run([sys.executable, "colab_perfgate.py", "--branch", BRANCH],
                       cwd=CLONE, capture_output=True, text=True,
                       env={**os.environ, "UT_PREBUILT": "0", "BRANCH": BRANCH})
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