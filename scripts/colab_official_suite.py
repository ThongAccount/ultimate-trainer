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

OUTPUT (important): `colab run` executes this file inside a Jupyter kernel,
so a child that merely INHERITS stdout writes to a descriptor the CLI never
sees — that silently lost one full T4 allocation on 2026-10-26. Every child
is therefore run through a PIPE and re-printed here (tools/stream_runner.py),
and the full transcript is also written to results/official_suite.log on the
VM so the numbers survive even if stdout breaks again.

Exit code 0 only if BOTH phases pass.  A flip-rate gate failure is a RESULT,
not a crash: the suite still prints every measurement.
"""
import argparse
import json
import os
import subprocess
import sys

# NOTE: `colab run` executes this file inside a Jupyter kernel, where
# __file__ is NOT defined — never reference it here (it cost one T4
# allocation on 2026-10-26).


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
                          capture_output=True, text=True).stdout.strip()
    print(f"commit={head}", flush=True)

    sys.path.insert(0, d)
    from tools.stream_runner import run_streaming

    transcript = []

    def phase(name, cmd):
        rc, out = run_streaming(cmd)
        transcript.append(f"### {name} (rc={rc})")
        transcript.extend(out)
        return rc, out

    # ── smoke pass first ───────────────────────────────────────────────────
    # A plumbing bug must not be discovered after an hour of GPU time. The
    # smoke runs the SAME scripts with tiny parameters on the SAME VM (so the
    # JIT builds are paid once), and only if both complete do we start the
    # official measurements.
    if not args.skip_smoke:
        print("=== SMOKE: validating the plumbing (tiny parameters) ===", flush=True)
        s_ab_rc, s_ab = phase(
            "SMOKE A/B", [sys.executable, os.path.join(d, "scripts", "colab_step_ab.py"),
                          "--no-clone", "--repo", d, "--nsteps", "3", "--data", args.data])
        s_flip_rc, s_flip = phase(
            "SMOKE FLIP", [sys.executable, os.path.join(d, "scripts", "colab_flip_rate_gate.py"),
                           "--no-clone", "--repo", d, "--steps", "20", "--smoke"])
        if s_ab_rc != 0 or s_flip_rc != 0:
            print(f"SMOKE: FAILED (ab_rc={s_ab_rc} flip_rc={s_flip_rc}) "
                  "— aborting before the official run", flush=True)
            _write_artefacts(d, head, transcript, s_ab_rc, s_flip_rc, "smoke_failed")
            return 1
        print("SMOKE: OK — starting the official measurements", flush=True)

    rc_ab = rc_flip = 0
    if not args.skip_ab:
        print("=== PHASE 1: whole-step A/B ===", flush=True)
        rc_ab, _ = phase(
            "PHASE 1 A/B", [sys.executable, os.path.join(d, "scripts", "colab_step_ab.py"),
                            "--no-clone", "--repo", d, "--nsteps", str(args.nsteps),
                            "--data", args.data])
        print(f"PHASE1_RC: {rc_ab}", flush=True)

    if not args.skip_flip:
        print("=== PHASE 2: flip-rate trajectory gate ===", flush=True)
        rc_flip, _ = phase(
            "PHASE 2 FLIP", [sys.executable, os.path.join(d, "scripts", "colab_flip_rate_gate.py"),
                             "--no-clone", "--repo", d, "--steps", str(args.flip_steps)])
        print(f"PHASE2_RC: {rc_flip}", flush=True)

    ok = (rc_ab == 0) and (rc_flip == 0)
    print(f"OFFICIAL_SUITE: {'PASS' if ok else 'FAIL'} "
          f"ab_rc={rc_ab} flip_rc={rc_flip}", flush=True)
    _write_artefacts(d, head, transcript, rc_ab, rc_flip,
                     "pass" if ok else "fail")
    return 0 if ok else 1


def _write_artefacts(d, head, transcript, rc_ab, rc_flip, verdict):
    """Persist the full transcript + a summary next to the repo on the VM."""
    outdir = os.path.join(d, "results")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "official_suite.log"), "w") as fh:
        fh.write(f"commit={head}\n")
        fh.write("\n".join(transcript) + "\n")
    with open(os.path.join(outdir, "official_suite.json"), "w") as fh:
        json.dump({"commit": head, "ab_rc": rc_ab, "flip_rc": rc_flip,
                   "verdict": verdict, "transcript": transcript},
                  fh, indent=2)
    print(f"ARTEFACT: {os.path.join(outdir, 'official_suite.json')}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
