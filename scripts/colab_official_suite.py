#!/usr/bin/env python3
"""colab run one-shot: the official ultimate-trainer suite on ONE GPU.

Runs both outstanding campaign measurements in a single Colab VM so the T4 is
allocated once and the JIT extension builds are shared:

  Phase 1  whole-step A/B  (scripts/colab_step_ab.py)
           three arms: custom_custom -> cublas_custom -> cublas_cublas,
           N steps each, on the committed FineWeb tokenized data.
  Phase 2  flip-rate trajectory gate  (scripts/colab_flip_rate_gate.py)
           1k steps x 2 noise schedules x 2 update backends.

Both phases clone/reset the repo exactly once; the second phase reuses the
clone (--no-clone) so it does not throw away the warm JIT caches.

Exit code 0 only if BOTH phases pass.  A flip-rate gate failure is a RESULT,
not a crash: the suite still prints every measurement.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/content/ultimate-trainer")
    ap.add_argument("--nsteps", type=int, default=300)
    ap.add_argument("--flip-steps", type=int, default=1000)
    ap.add_argument("--data", default="real", choices=["real", "synthetic"])
    ap.add_argument("--skip-ab", action="store_true")
    ap.add_argument("--skip-flip", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true",
                    help="skip the plumbing smoke pass (NOT recommended: it "
                         "catches script bugs before the official run)")
    args, _unknown = ap.parse_known_args()

    d = args.repo
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja"],
                   capture_output=True)

    # ── clone once ─────────────────────────────────────────────────────────
    tok = os.environ.get("GITHUB_TOKEN", "")
    url = "https://github.com/ThongAccount/ultimate-trainer.git"
    if tok:
        url = url.replace("https://", f"https://{tok}@")
    if not os.path.isdir(os.path.join(d, ".git")):
        subprocess.run(["git", "clone", "--quiet", url, d], check=True)
    subprocess.run(["git", "-C", d, "fetch", "--quiet", "origin"], check=False)
    subprocess.run(["git", "-C", d, "reset", "--hard", "--quiet", "origin/master"], check=True)
    head = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()[:10]
    print(f"commit={head}", flush=True)

    rc_ab = rc_flip = 0
    # ── smoke pass first ───────────────────────────────────────────────────
    # A plumbing bug must not be discovered after an hour of GPU time. The
    # smoke runs the SAME scripts with tiny parameters on the SAME VM (so the
    # JIT builds are paid once), and only if both complete do we start the
    # official measurements.
    if not args.skip_smoke:
        print("=== SMOKE: validating the plumbing (tiny parameters) ===", flush=True)
        s_ab = subprocess.run(
            [sys.executable, os.path.join(d, "scripts", "colab_step_ab.py"),
             "--no-clone", "--repo", d, "--nsteps", "3", "--data", args.data],
            capture_output=True, text=True)
        print(s_ab.stdout[-4000:], flush=True)
        if s_ab.returncode != 0:
            print(s_ab.stderr[-4000:], file=sys.stderr, flush=True)
        s_flip = subprocess.run(
            [sys.executable, os.path.join(d, "scripts", "colab_flip_rate_gate.py"),
             "--no-clone", "--repo", d, "--steps", "20", "--smoke"],
            capture_output=True, text=True)
        print(s_flip.stdout[-4000:], flush=True)
        if s_flip.returncode != 0:
            print(s_flip.stderr[-4000:], file=sys.stderr, flush=True)
        if s_ab.returncode != 0 or s_flip.returncode != 0:
            print(f"SMOKE: FAILED (ab_rc={s_ab.returncode} flip_rc={s_flip.returncode}) "
                  "— aborting before the official run", flush=True)
            return 1
        print("SMOKE: OK — starting the official measurements", flush=True)

    if not args.skip_ab:
        print("=== PHASE 1: whole-step A/B ===", flush=True)
        rc_ab = subprocess.run(
            [sys.executable, os.path.join(d, "scripts", "colab_step_ab.py"),
             "--no-clone", "--repo", d, "--nsteps", str(args.nsteps),
             "--data", args.data]).returncode
        print(f"PHASE1_RC: {rc_ab}", flush=True)

    if not args.skip_flip:
        print("=== PHASE 2: flip-rate trajectory gate ===", flush=True)
        rc_flip = subprocess.run(
            [sys.executable, os.path.join(d, "scripts", "colab_flip_rate_gate.py"),
             "--no-clone", "--repo", d, "--steps", str(args.flip_steps)]).returncode
        print(f"PHASE2_RC: {rc_flip}", flush=True)

    ok = (rc_ab == 0) and (rc_flip == 0)
    print(f"OFFICIAL_SUITE: {'PASS' if ok else 'FAIL'} "
          f"ab_rc={rc_ab} flip_rc={rc_flip}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
