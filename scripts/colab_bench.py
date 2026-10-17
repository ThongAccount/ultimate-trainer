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


def ensure_repo(commit: str = "HEAD", branch: str = "chore/speedpass") -> str:
    tok = os.environ.get("GITHUB_TOKEN", "")
    if not os.path.isdir(os.path.join(DEST, ".git")):
        url = REPO_URL.replace("https://", f"https://{tok}@") if tok else REPO_URL
        subprocess.run(["git", "clone", "--quiet", url, DEST], check=True)
    else:
        subprocess.run(["git", "-C", DEST, "fetch", "--quiet", "origin"], check=True)
    # Pin the branch FIRST — a fresh clone lands on the default (master), and
    # `checkout HEAD` there silently keeps the stale commit.
    subprocess.run(
        ["git", "-C", DEST, "checkout", "--quiet", branch], check=True
    )
    subprocess.run(
        ["git", "-C", DEST, "reset", "--hard", "--quiet", f"origin/{branch}"],
        check=True,
    )
    if commit != "HEAD":
        subprocess.run(
            ["git", "-C", DEST, "checkout", "--quiet", commit], check=True
        )
        subprocess.run(
            ["git", "-C", DEST, "reset", "--hard", "--quiet", commit], check=True
        )
    head = subprocess.check_output(
        ["git", "-C", DEST, "rev-parse", "HEAD"], text=True).strip()
    print(f"repo={DEST} branch={branch} commit={head}", flush=True)
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
    ap.add_argument("--shapes", default="fc1",
                    help="comma list of fc1,fc2,head — keep each Colab "
                         "invocation under the tool's ~300s deadline")
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

    tests_dir = os.path.join(os.getcwd(), "tests") if not args.no_setup else "tests"
    sys.path.insert(0, tests_dir)
    import bench_roofline as br

    wanted = args.shapes.split(",") if args.shapes != "all" else None
    rows = []
    for sname, K, N in br.SHAPES:
        if wanted and sname not in wanted:
            continue
        print(f"--- {sname} B={args.batch} K={K} N={N} ---", flush=True)
        row = br.bench_shape(sname, K, N, args.batch, "cuda")
        rows.append(row)
        for k, v in row.items():
            if isinstance(v, float):
                print(f"  {k:26s} {v:10.3f}", flush=True)
            elif v is not None:
                print(f"  {k:26s} {v}", flush=True)

    payload = {"roofline": rows, "env": br.gpu_info(), "batch": args.batch}
    out = args.out
    if len(rows) == 1:
        base, ext = os.path.splitext(out)
        out = f"{base}_{rows[0]['shape']}{ext or '.json'}"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"WROTE {out}", flush=True)
    print(f"RESULT JSON: {json.dumps(payload, indent=2)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
