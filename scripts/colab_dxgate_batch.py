#!/usr/bin/env python3
"""colab run one-shot: dX-gate + batch-scale + shipped-path profile on ONE T4.

Pre-registered in docs/speedpass/2026-11-02-dxgate-shipped-prereg.md.

  0. smoke: tests/probe_dx_gate.py --parity-only (plumbing + parity check)
  1. dxgate: tests/probe_dx_gate.py          dX 64x64 vs 32x32 at M=16352/16384
                                               + fwd/dX/update batch scaling
                                               M in {16352, 16384, 32768}
  2. shipped_profile: scripts/colab_shipped_profile.py
                     PT_BACKEND=cublas PT_UPDATE_BACKEND=cublas, B=0 vs B=5,
                     300 steps, empty_cache @150, torch.profiler steps 100-200

rc: 0 = all steps ran (win/loss is a RESULT in the logs), 1 = a gate failed,
2 = commit-mismatch refusal (INVALID).
"""
import argparse
import os
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/content/ultimate-trainer")
    ap.add_argument("--branch", default="master")
    ap.add_argument("--expect-commit", default=os.environ.get("PT_EXPECT_COMMIT", ""))
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip", default="")
    args, _unknown = ap.parse_known_args()

    d = args.repo
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja"],
                   capture_output=True)

    tok = os.environ.get("GITHUB_TOKEN", "")
    url = "https://github.com/ThongAccount/ultimate-trainer.git"
    if tok:
        url = url.replace("https://", f"https://{tok}@")
    if not os.path.isdir(os.path.join(d, ".git")):
        subprocess.run(["git", "clone", "--quiet", url, d], check=True)
    subprocess.run(["git", "-C", d, "fetch", "--quiet", "origin"], check=True)
    subprocess.run(["git", "-C", d, "reset", "--hard", "--quiet",
                    f"origin/{args.branch}"], check=True)
    head = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    tag = args.tag or head[:8]
    print(f"commit={head} branch={args.branch} tag={tag}", flush=True)

    if d not in sys.path:
        sys.path.insert(0, d)
    from tools.fetch_prebuilt import ensure_prebuilt
    ensure_prebuilt(d)
    from tools.stream_runner import run_streaming
    from tools.artefact_provenance import collect_provenance
    provenance = collect_provenance(d, extra_config={"config": {
        "branch": args.branch, "tag": tag, "skip": sorted(skip),
        "expect_commit": args.expect_commit or None}})

    if args.expect_commit and not head.startswith(args.expect_commit):
        print(f"COMMIT-MISMATCH: HEAD={head[:10]} != expect={args.expect_commit}", flush=True)
        _write(d, tag, head, [], {"commit_mismatch": True}, "invalid", provenance)
        return 2

    transcript = []

    def step(name, cmd, env=None):
        print(f"=== STEP {name}: {' '.join(cmd)} ===", flush=True)
        rc, out = run_streaming(cmd, cwd=d, env=env)
        transcript.append(f"### {name} (rc={rc})")
        transcript.extend(out)
        log = os.path.join(d, "results", f"dxgate_batch.{tag}.{name}.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w") as fh:
            fh.write(f"commit={head}\ncmd={' '.join(cmd)}\nrc={rc}\n")
            fh.write("\n".join(out) + "\n")
        print(f"STEP_RC: {name} {rc}", flush=True)
        return rc

    results = {}
    if "smoke" not in skip:
        results["smoke"] = step("smoke", [sys.executable,
                               os.path.join(d, "tests", "probe_dx_gate.py"),
                               "--parity-only"])
        if results["smoke"] != 0:
            print("SMOKE: FAILED — aborting", flush=True)
            return _finish(d, tag, head, transcript, results, "smoke_failed", provenance)
    if "dxgate" not in skip:
        results["dxgate"] = step("dxgate", [sys.executable,
                               os.path.join(d, "tests", "probe_dx_gate.py")])
    if "shipped" not in skip:
        env = {**os.environ, "PT_BACKEND": "cublas", "PT_UPDATE_BACKEND": "cublas"}
        results["shipped_profile"] = step(
            "shipped_profile",
            [sys.executable, os.path.join(d, "scripts", "colab_shipped_profile.py"),
             "--repo", d, "--expect-commit", head, "--tag", tag], env=env)

    ok = all(r == 0 for r in results.values()) and results
    verdict = "pass" if ok else "fail"
    print(f"DXGATE_BATCH: {'PASS' if ok else 'FAIL'} results={results}", flush=True)
    return _finish(d, tag, head, transcript, results, verdict, provenance)


def _finish(d, tag, head, transcript, results, verdict, provenance=None):
    _write(d, tag, head, transcript, results, verdict, provenance)
    if verdict == "pass":
        return 0
    if verdict == "invalid":
        return 2
    return 1


def _write(d, tag, head, transcript, results, verdict, provenance=None):
    from tools.artefact_provenance import mark_invalid, write_artefact
    outdir = os.path.join(d, "results")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"dxgate_batch.{tag}.log"), "w") as fh:
        fh.write(f"commit={head}\n")
        fh.write("\n".join(transcript) + "\n")
    artefact = {"commit": head, "tag": tag, "provenance": provenance,
                "verdict": verdict, "step_rcs": results, "transcript": transcript}
    if verdict in ("invalid", "smoke_failed"):
        mark_invalid(artefact, f"verdict={verdict} rcs={results}", root_cause_commit=head)
    path = write_artefact(d, f"dxgate_batch.{tag}.json", artefact, root_cause_commit=head)
    print(f"ARTEFACT: {path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
