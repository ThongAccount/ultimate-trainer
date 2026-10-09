"""Colab one-shot runner: full pytest correctness suite on GPU (T4).

Replaces the payment-gated modal_pytest_suite.py. Clones committed HEAD, runs
the same four test modules under pytest, prints RESULTS JSON. Use this as the
correctness gate after touching kernels.

Contract: `RESULTS JSON:` block on stdout, nothing else to parse.
Pair with `colab run`, which provisions a fresh VM, runs this, tears it down.

Usage
-----
    TOK=$(gh auth token)
    colab run --gpu T4 --timeout 5400 --env "GITHUB_TOKEN=$TOK" colab_pytest.py

Exit code mirrors pytest: 0 = all pass, non-zero = failures (so CI can gate).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

REPO_URL = "github.com/ThongAccount/ultimate-trainer.git"
DEFAULT_CLONE = "/content/ultimate-trainer"
DEFAULT_BRANCH = "chore/speedpass"

# Same modules the Modal suite ran, plus the fused/sparse smoke modules touched
# by the C11 kernel fixes.
TEST_MODULES = [
    "tests/test_packed_linear.py",
    "tests/test_gemm_update.py",
    "tests/test_kernels.py",
    "tests/test_packed_ternary.py",
    "tests/test_gemm_fused.py",
    "tests/test_speedpass_kernels.py",
]


def clone_repo(branch: str, dest: str, token: str) -> str:
    if os.path.isdir(os.path.join(dest, "kernels", "packed_ternary")):
        return dest
    if not token:
        raise SystemExit(
            "GITHUB_TOKEN not set (pass --env 'GITHUB_TOKEN=$(gh auth token)')"
        )
    url = f"https://x-access-token:{token}@{REPO_URL}"
    r = subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch, "--quiet", url, dest],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        err = (r.stderr or "").replace(token, "<token>")
        raise SystemExit(f"git clone failed (rc={r.returncode}): {err[-600:]}")
    return dest


def main() -> int:
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(
        f"torch={torch.__version__} cuda={torch.cuda.is_available()} gpu={gpu}",
        flush=True,
    )

    branch = os.environ.get("BRANCH", DEFAULT_BRANCH)
    repo = clone_repo(branch, DEFAULT_CLONE, os.environ.get("GITHUB_TOKEN", ""))
    commit = (
        subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"])
        .decode()
        .strip()
    )
    print(f"repo={repo} branch={branch} commit={commit}", flush=True)

    for pkg in ("ninja", "pytest"):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", pkg],
            capture_output=True,
            text=True,
            check=False,
        )

    t0 = time.time()
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *TEST_MODULES,
            "-q",
            "--no-header",
            "--tb=short",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=3000,
    )
    wall = round(time.time() - t0, 1)
    out = (r.stdout or "") + (r.stderr or "")
    print(out, flush=True)

    tail = [ln for ln in out.splitlines() if ln.strip()][-1:] or ["(no output)"]
    res = {
        "suite": "pytest",
        "commit": commit,
        "branch": branch,
        "gpu": gpu,
        "pytorch": torch.__version__,
        "modules": TEST_MODULES,
        "returncode": r.returncode,
        "passed": r.returncode == 0,
        "summary": tail[0],
        "wall_s": wall,
    }
    print("\nRESULTS JSON:", json.dumps(res, indent=2), flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
