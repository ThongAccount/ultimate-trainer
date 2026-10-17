"""Colab one-shot: setup (clone/install) + Phase 0 roofline in-process.

Mirrors colab_speedpass.py's structure (clone via GITHUB_TOKEN, ensure ninja,
run in-process, print RESULTS JSON). Avoids `colab run bash -c` (the CLI only
accepts a script path).

Usage on Colab T4:
    TORCH_CUDA_ARCH_LIST=7.5 python3 scripts/colab_bench.py --phase 0 \
        --batch 16384 --out /content/roofline.json
"""
import argparse
import json
import os
import subprocess
import sys

REPO_URL = "https://github.com/ThongAccount/ultimate-trainer.git"
DEST = "/content/ultimate-trainer"


def ensure_repo(commit: str = "HEAD") -> str:
    tok = os.environ.get("GITHUB_TOKEN", "")
    if not os.path.isdir(os.path.join(DEST, ".git")):
        url = REPO_URL.replace("https://", f"https://{tok}@") if tok else REPO_URL
        subprocess.run(["git", "clone", "--quiet", url, DEST], check=True)
    else:
        subprocess.run(["git", "-C", DEST, "fetch", "--quiet", "origin"], check=True)
    subprocess.run(
        ["git", "-C", DEST, "checkout", "--quiet", commit], check=True
    )
    subprocess.run(
        ["git", "-C", DEST, "pull", "--quiet", "origin",
         subprocess.run(["git", "-C", DEST, "branch", "--show-current"],
                        capture_output=True, text=True).stdout.strip()],
        check=False,
    )
    head = subprocess.check_output(
        ["git", "-C", DEST, "rev-parse", "HEAD"], text=True).strip()
    print(f"repo={DEST} commit={head}", flush=True)
    return DEST


def ensure_deps():
    subprocess.run(["pip", "install", "-q", "ninja"], check=False,
                   capture_output=True)
    subprocess.run(
        ["pip", "install", "-q", "-r", os.path.join(DEST, "requirements.txt")],
        check=False, capture_output=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="0", choices=("0", "1"))
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--out", default="/content/roofline.json")
    ap.add_argument("--no-setup", action="store_true")
    args = ap.parse_args()

    print("---- nvidia-smi ----", flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,clocks.max.sm,"
                    "clocks.sm,power.limit", "--format=csv"])

    import torch
    assert torch.cuda.is_available(), "no CUDA"
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"torch={torch.__version__} cuda=True gpu={name} cap={cap}", flush=True)
    assert cap == (7, 5), f"NOT a T4: {name} cap={cap} — invalid numbers, aborting"

    if not args.no_setup:
        ensure_repo(args.commit)
        ensure_deps()
        os.chdir(DEST)
        for p in (DEST, os.path.join(DEST, "tests")):
            if p not in sys.path:
                sys.path.insert(0, p)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")

    sys.path.insert(0, "tests")
    import bench_roofline as br

    rows = []
    for sname, K, N in br.SHAPES:
        print(f"--- {sname} B={args.batch} K={K} N={N} ---", flush=True)
        row = br.bench_shape(sname, K, N, args.batch, "cuda")
        rows.append(row)
        for k, v in row.items():
            if isinstance(v, float):
                print(f"  {k:26s} {v:10.3f}", flush=True)
            elif v is not None:
                print(f"  {k:26s} {v}", flush=True)

    payload = {"roofline": rows, "env": br.gpu_info(), "batch": args.batch}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"RESULT JSON: {json.dumps(payload, indent=2)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
