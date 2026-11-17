#!/usr/bin/env python3
"""colab run one-shot: probe batch 2 on ONE T4, production kernels from the
prebuilt bundle (UT_PREBUILT_TAG), probe variants JIT on the VM.

Pre-registered in docs/speedpass/2026-11-03-probe-batch-2-prereg.md
(committed BEFORE any Colab allocation). Order and decision rules there are
binding. Items: #10 dX K64 unroll-2, #11 fwd uint4 W loads, #12 clone cost,
#15 small-dims parity, #6 fused-CE probe, #17 flip-budget B{3,4} sensitivity.

  0. smoke: tests/probe_smoke2.py (compile+parity of both variants, tiny shape)
  1. pytest correctness gate (colab_pytest.TEST_MODULES, cwd=clone)
  2. tests/probe_dx_unroll2.py    #10
  3. tests/probe_fwd_uint4.py     #11
  4. tests/probe_misc.py          #12 #15 #6
  5. scripts/colab_flip_rate_gate.py --no-clone --steps 400 --flip-budget 3   #17
  6. scripts/colab_flip_rate_gate.py --no-clone --steps 400 --flip-budget 4   #17
  7. colab_perfgate.py check + regen vs committed baseline

Every child runs through tools/stream_runner.run_streaming (a Jupyter-kernel
child that merely inherits stdout is silently lost). Logs land in
results/probe_batch2.<tag>.* on the VM; summary -> results/probe_batch2.<tag>.json.

rc semantics: 0 = all gates passed (probe win/loss is a RESULT, not a
failure), 1 = a gate failed, 2 = commit-mismatch refusal (INVALID).
"""
import argparse
import os
import subprocess
import sys

# NOTE: `colab run` executes this file inside a Jupyter kernel, where
# __file__ is NOT defined — never reference it here.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/content/ultimate-trainer")
    ap.add_argument("--branch", default="feat/probe-batch-2")
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

    expect = args.expect_commit
    if expect and not head.startswith(expect):
        print(f"COMMIT-MISMATCH: clone HEAD={head[:10]} != --expect-commit={expect}; "
              f"refusing to run", flush=True)
        _write(d, tag, head, [], {"commit_mismatch": {"head": head[:10], "expect": expect}},
               "invalid", provenance)
        return 2

    transcript = []

    def step(name, cmd, cwd=None):
        print(f"=== STEP {name}: {' '.join(cmd)} ===", flush=True)
        rc, out = run_streaming(cmd, cwd=cwd or d)
        transcript.append(f"### {name} (rc={rc})")
        transcript.extend(out)
        log = os.path.join(d, "results", f"probe_batch2.{tag}.{name}.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w") as fh:
            fh.write(f"commit={head}\ncmd={' '.join(cmd)}\nrc={rc}\n")
            fh.write("\n".join(out) + "\n")
        print(f"STEP_RC: {name} {rc}", flush=True)
        return rc, out

    results = {}

    if "smoke" not in skip:
        rc, _ = step("smoke", [sys.executable, os.path.join(d, "tests", "probe_smoke2.py")])
        results["smoke"] = rc
        if rc != 0:
            print("SMOKE: FAILED — aborting before the batch", flush=True)
            return _finish(d, tag, head, transcript, results, "smoke_failed", provenance)

    if "pytest" not in skip:
        import colab_pytest
        rc, _ = step("pytest", [sys.executable, "-m", "pytest", "-q", "--tb=short",
                                *colab_pytest.TEST_MODULES])
        results["pytest"] = rc

    probes = [
        ("dx_unroll2", "tests/probe_dx_unroll2.py"),    # #10
        ("fwd_uint4", "tests/probe_fwd_uint4.py"),      # #11
        ("misc", "tests/probe_misc.py"),                # #12 #15 #6
    ]
    for name, rel in probes:
        if "probes" in skip or name in skip:
            continue
        rc, _ = step(name, [sys.executable, os.path.join(d, rel)])
        results[name] = rc

    if "flip17" not in skip:
        for fb in (3, 4):
            name = f"flip_b{fb}"
            if name in skip:
                continue
            rc, _ = step(name, [sys.executable, os.path.join(d, "scripts", "colab_flip_rate_gate.py"),
                                "--repo", d, "--no-clone", "--steps", "400",
                                "--flip-budget", str(fb), "--expect-commit", head])
            results[name] = rc

    if "perfgate" not in skip:
        rc, _ = step("perfgate_check", [
            sys.executable, os.path.join(d, "colab_perfgate.py"),
            "--branch", args.branch, "--repo-dir", d, "--expect-commit", head])
        results["perfgate_check"] = rc
        rc, _ = step("perfgate_regen", [
            sys.executable, os.path.join(d, "colab_perfgate.py"),
            "--branch", args.branch, "--repo-dir", d, "--expect-commit", head,
            "--update-baseline"])
        results["perfgate_regen"] = rc

    gates = [v for k, v in results.items() if k != "perfgate_regen"]
    ok = all(r == 0 for r in gates) and len(gates) > 0
    verdict = "pass" if ok else "fail"
    print(f"PROBE_BATCH2: {'PASS' if ok else 'FAIL'} results={results}", flush=True)
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
    with open(os.path.join(outdir, f"probe_batch2.{tag}.log"), "w") as fh:
        fh.write(f"commit={head}\n")
        fh.write("\n".join(transcript) + "\n")
    artefact = {"commit": head, "tag": tag, "provenance": provenance,
                "verdict": verdict, "step_rcs": results, "transcript": transcript}
    if verdict in ("invalid", "smoke_failed"):
        mark_invalid(artefact, f"verdict={verdict} rcs={results}",
                     root_cause_commit=head)
    path = write_artefact(d, f"probe_batch2.{tag}.json", artefact,
                          root_cause_commit=head)
    print(f"ARTEFACT: {path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
