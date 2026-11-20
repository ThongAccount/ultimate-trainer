#!/usr/bin/env python3
"""colab run one-shot: the C13 GO probe batch on ONE T4, kernels prebuilt by CI.

Pre-registered in docs/speedpass/2026-11-01-c13-probe-batch-prereg.md (committed
BEFORE any Colab allocation). Order and decision rules there are binding.

  0. smoke: probe_fwd_spill parity-only (plumbing check, cheap)
  1. pytest correctness gate (colab_pytest.TEST_MODULES, cwd=clone)
  2. tests/probe_fwd_spill.py        GO: fwd warp-private spill epilogue
  3. tests/probe_fwd_swizzle.py      GO: fwd n-outer rasterization swizzle
  4. tests/probe_update_pipeline.py  GO: update double-buffer (falsifier armed)
  5. tests/probe_fwd_wdedup.py       GO: fwd W-word register dedup
  6. colab_perfgate.py               gate vs committed baseline
  7. colab_perfgate.py --update-baseline   regen baseline under current protocol

Every child is run through tools/stream_runner.run_streaming (a Jupyter-kernel
child that merely inherits stdout is silently lost — that cost one full T4
allocation on 2026-10-26). Per-step logs land in results/probe_batch.<tag>.*
on the VM; the summary goes to results/probe_batch.<tag>.json via write_artefact.

rc semantics: 0 = all gates passed (probe win/loss is a RESULT inside the logs,
not a failure), 1 = a gate failed, 2 = commit-mismatch refusal (INVALID).
"""
import argparse
import json
import os
import subprocess
import sys

# NOTE: `colab run` executes this file inside a Jupyter kernel, where
# __file__ is NOT defined — never reference it here.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/content/ultimate-trainer")
    ap.add_argument("--branch", default="master")
    ap.add_argument("--expect-commit", default=os.environ.get("PT_EXPECT_COMMIT", ""),
                    help="rc=2 unless clone HEAD matches this sha (prefix ok)")
    ap.add_argument("--tag", default="",
                    help="artefact tag; default = short clone HEAD")
    ap.add_argument("--skip", default="",
                    help="comma-separated step names to skip (smoke,pytest,probes,perfgate)")
    args, _unknown = ap.parse_known_args()

    d = args.repo
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja"],
                   capture_output=True)

    # ── clone once ─────────────────────────────────────────────────────────
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
    # Prebuilt kernels from the kernels-latest release: skips on-VM nvcc JIT
    # for the production extensions. Must run before any child triggers
    # load_inline so children inherit TORCH_EXTENSIONS_DIR. Probe VARIANTS
    # (new extension names) still JIT — one small compile each.
    from tools.fetch_prebuilt import ensure_prebuilt
    ensure_prebuilt(d)
    from tools.stream_runner import run_streaming
    from tools.artefact_provenance import collect_provenance, mark_invalid, write_artefact
    provenance = collect_provenance(d, extra_config={"config": {
        "branch": args.branch, "tag": tag, "skip": sorted(skip),
        "expect_commit": args.expect_commit or None}})

    # stale-clone guard: rc=2 + quarantined artefact BEFORE any measurement
    expect = args.expect_commit
    if expect and not head.startswith(expect):
        print(f"COMMIT-MISMATCH: clone HEAD={head[:10]} does not match "
              f"--expect-commit={expect}; refusing to run", flush=True)
        _write(d, tag, head, [], {"commit_mismatch": {"head": head[:10], "expect": expect}},
               "invalid", provenance)
        return 2

    transcript = []

    def step(name, cmd, cwd=None):
        print(f"=== STEP {name}: {' '.join(cmd)} ===", flush=True)
        rc, out = run_streaming(cmd, cwd=cwd or d)
        transcript.append(f"### {name} (rc={rc})")
        transcript.extend(out)
        log = os.path.join(d, "results", f"probe_batch.{tag}.{name}.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w") as fh:
            fh.write(f"commit={head}\ncmd={' '.join(cmd)}\nrc={rc}\n")
            fh.write("\n".join(out) + "\n")
        print(f"STEP_RC: {name} {rc}", flush=True)
        return rc, out

    results = {}

    # ── 0. smoke: parity-only, must not burn an hour discovering plumbing ──
    if "smoke" not in skip:
        rc, _ = step("smoke", [sys.executable, os.path.join(d, "tests", "probe_fwd_spill.py"),
                               "--probe", "fwdspill", "--skip-timing"])
        results["smoke"] = rc
        if rc != 0:
            print("SMOKE: FAILED — aborting before the batch", flush=True)
            return _finish(d, tag, head, transcript, results, "smoke_failed", provenance)

    # ── 1. pytest correctness gate ─────────────────────────────────────────
    if "pytest" not in skip:
        import colab_pytest  # repo root is on sys.path; TEST_MODULES only
        rc, _ = step("pytest", [sys.executable, "-m", "pytest", "-q", "--tb=short",
                                *colab_pytest.TEST_MODULES])
        results["pytest"] = rc

    # ── 2-5. the four kernel probes ────────────────────────────────────────
    probes = [
        ("fwdspill", "tests/probe_fwd_spill.py"),
        ("fwdswizzle", "tests/probe_fwd_swizzle.py"),
        ("uppipeline", "tests/probe_update_pipeline.py"),
        ("fwdwdedup", "tests/probe_fwd_wdedup.py"),
    ]
    for name, rel in probes:
        if "probes" in skip or name in skip:
            continue
        rc, _ = step(name, [sys.executable, os.path.join(d, rel)])
        results[name] = rc

    # ── 6-7. perf gate: check vs committed baseline, then regen ────────────
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

    # only steps that actually ran this session count as gates
    gates = [v for k, v in results.items() if k != "perfgate_regen"]
    ok = all(r == 0 for r in gates) and len(gates) > 0
    verdict = "pass" if ok else "fail"
    print(f"PROBE_BATCH: {'PASS' if ok else 'FAIL'} results={results}", flush=True)
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
    with open(os.path.join(outdir, f"probe_batch.{tag}.log"), "w") as fh:
        fh.write(f"commit={head}\n")
        fh.write("\n".join(transcript) + "\n")
    artefact = {"commit": head, "tag": tag, "provenance": provenance,
                "verdict": verdict, "step_rcs": results, "transcript": transcript}
    if verdict in ("invalid", "smoke_failed"):
        mark_invalid(artefact, f"verdict={verdict} rcs={results}",
                     root_cause_commit=head)
    path = write_artefact(d, f"probe_batch.{tag}.json", artefact,
                          root_cause_commit=head)
    print(f"ARTEFACT: {path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
