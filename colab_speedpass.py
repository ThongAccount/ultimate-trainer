#!/usr/bin/env python3
"""Colab one-shot runner for Ultimate Trainer speedpass probes (GPU).

Replaces the payment-gated Modal runners (modal_probe_dx_regsweep.py,
modal_e2e_host_ab.py). Colab T4 is the current GPU backend.

Contract: prints a `RESULTS JSON:` block to stdout, nothing else to parse.
Pair with `colab run`, which provisions a fresh VM, runs this, tears it down.

Usage
-----
    TOK=$(gh auth token)
    colab run --gpu T4 --timeout 5400 --env "GITHUB_TOKEN=$TOK" colab_speedpass.py \
        --probe regsweep
    colab run --gpu T4 --timeout 5400 --env "GITHUB_TOKEN=$TOK" colab_speedpass.py \
        --probe hostab

Wraps tests/probe_dx_regsweep.py and tests/e2e_host_ab.py in-process (same
pattern as bdh colab_scale.py). GitHub token comes from the environment and is
scrubbed from any error text we print.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time

REPO_URL = "github.com/ThongAccount/ultimate-trainer.git"
DEFAULT_CLONE = "/content/ultimate-trainer"
DEFAULT_BRANCH = "chore/speedpass"


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


def ensure_ninja() -> None:
    try:
        import ninja  # noqa: F401

        return
    except ImportError:
        pass
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "ninja"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise SystemExit(f"pip install ninja failed: {(r.stderr or '')[-400:]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--probe",
        default="regsweep",
        choices=("regsweep", "hostab"),
        help="regsweep: dX maxrregcount sweep; hostab: sync+clone e2e A/B",
    )
    ap.add_argument("--branch", default=DEFAULT_BRANCH)
    ap.add_argument("--repo-dir", default=DEFAULT_CLONE)
    args = ap.parse_args()

    import torch

    print(
        f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
        flush=True,
    )
    if not torch.cuda.is_available():
        raise SystemExit("no GPU — probe needs T4")

    tok = os.environ.get("GITHUB_TOKEN", "")
    repo = clone_repo(args.branch, args.repo_dir, tok)
    sys.path.insert(0, repo)
    commit = (
        subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"])
        .decode()
        .strip()
    )
    print(f"repo={repo} branch={args.branch} commit={commit}", flush=True)

    ensure_ninja()
    os.chdir(repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    # tests/ probes do sys.path.insert of repo themselves; ensure cwd imports work
    sys.path.insert(0, os.path.join(repo, "tests"))

    t0 = time.time()
    buf = io.StringIO()
    if args.probe == "regsweep":
        import probe_dx_regsweep as probe
    else:
        import e2e_host_ab as probe
    with contextlib.redirect_stdout(buf):
        probe.main()
    wall = round(time.time() - t0, 1)
    log = buf.getvalue()
    print(log, flush=True)

    res = {
        "probe": args.probe,
        "commit": commit,
        "branch": args.branch,
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "wall_s": wall,
        "log": log.splitlines(),
    }
    print("\nRESULTS JSON:", json.dumps(res, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
