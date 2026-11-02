#!/usr/bin/env python3
"""colab run one-shot: validate feat/dx-gate-16 on ONE T4 (prebuilt kernels OK —
the change is a pure-Python dispatch gate, no .cu touched).

  1. pytest gate (colab_pytest.TEST_MODULES, incl. the new M=16352 dX test)
  2. colab_perfgate.py check vs the measured baseline (step at M=16352 shows
     the whole-step effect of the dX 64x64 gate; kernel rows at B=16384 must
     be unchanged — %64 already passed there)

Pre-registered in docs/speedpass/2026-11-02-dx-gate-16-prereg.md.
rc: 0 = pass, 1 = gate FAIL (valid result), 2 = commit-mismatch (invalid).
"""
import argparse
import os
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/content/ultimate-trainer")
    ap.add_argument("--branch", default="feat/dx-gate-16")
    ap.add_argument("--expect-commit", default=os.environ.get("PT_EXPECT_COMMIT", ""))
    ap.add_argument("--tag", default="")
    args, _unknown = ap.parse_known_args()

    d = args.repo
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
        "branch": args.branch, "tag": tag,
        "expect_commit": args.expect_commit or None}})

    if args.expect_commit and not head.startswith(args.expect_commit):
        print(f"COMMIT-MISMATCH: HEAD={head[:10]} != expect={args.expect_commit}", flush=True)
        _write(d, tag, head, [], {"commit_mismatch": True}, "invalid", provenance)
        return 2

    transcript = []

    def step(name, cmd):
        print(f"=== STEP {name} ===", flush=True)
        rc, out = run_streaming(cmd, cwd=d)
        transcript.append(f"### {name} (rc={rc})")
        transcript.extend(out)
        log = os.path.join(d, "results", f"gate16.{tag}.{name}.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w") as fh:
            fh.write(f"commit={head}\ncmd={' '.join(cmd)}\nrc={rc}\n\n")
            fh.write("\n".join(out) + "\n")
        print(f"STEP_RC: {name} {rc}", flush=True)
        return rc

    import colab_pytest
    results = {}
    results["pytest"] = step("pytest", [sys.executable, "-m", "pytest", "-q",
                                        "--tb=short", *colab_pytest.TEST_MODULES])
    results["perfgate_check"] = step("perfgate_check", [
        sys.executable, os.path.join(d, "colab_perfgate.py"),
        "--branch", args.branch, "--repo-dir", d, "--expect-commit", head])

    ok = all(r == 0 for r in results.values())
    verdict = "pass" if ok else "fail"
    print(f"GATE16: {'PASS' if ok else 'FAIL'} results={results}", flush=True)
    _write(d, tag, head, transcript, results, verdict, provenance)
    return 0 if ok else (2 if verdict == "invalid" else 1)


def _write(d, tag, head, transcript, results, verdict, provenance=None):
    from tools.artefact_provenance import mark_invalid, write_artefact
    outdir = os.path.join(d, "results")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"gate16.{tag}.log"), "w") as fh:
        fh.write(f"commit={head}\n")
        fh.write("\n".join(transcript) + "\n")
    artefact = {"commit": head, "tag": tag, "provenance": provenance,
                "verdict": verdict, "step_rcs": results, "transcript": transcript}
    if verdict == "invalid":
        mark_invalid(artefact, f"rcs={results}", root_cause_commit=head)
    print("ARTEFACT:", write_artefact(d, f"gate16.{tag}.json", artefact,
                                      root_cause_commit=head), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
