"""Artefact provenance — the one block every results/*.json carries.

Why: batch-2 audit found every artefact writer except
scripts/colab_flip_rate_gate.py (since 07001be) emits results JSON with no
commit/config fields; ut-fb-10 showed such an artefact is indistinguishable
from a valid run once committed.

Rules: importable WITHOUT torch (lazy import inside collect_provenance; all
torch fields degrade to None) — CPU boxes / pytest / CI safe. Secrets: only
a WHITELIST of env vars is read (_ENV_WHITELIST); GITHUB_TOKEN can never
leak. git read via subprocess; a non-git dir degrades to None fields instead
of raising.
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone

SCHEMA_VERSION = 2
_ENV_WHITELIST = ("PT_BACKEND", "PT_UPDATE_BACKEND", "TORCH_CUDA_ARCH_LIST")
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_ANY_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
DEFAULT_DOCS_LINK = "docs/artefact-provenance.md"


def _git(repo_dir, *args):
    try:
        r = subprocess.run(["git", "-C", str(repo_dir), *args],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _torch_info():
    """Lazy torch import — module must work with torch absent (CPU-only)."""
    try:
        import torch
    except Exception:  # ImportError or a broken CUDA driver at import time
        return {"torch": None, "cuda": None, "gpu": None}
    gpu = None
    try:
        if torch.cuda.is_available():
            maj, min_ = torch.cuda.get_device_capability(0)
            gpu = {"name": torch.cuda.get_device_name(0), "capability": [maj, min_]}
    except Exception:
        gpu = None
    # getattr: a stubbed torch module in sys.modules (e.g. the minimal stub
    # installed by tests/test_flip_budget_guards.py on torch-free CI boxes)
    # has no __version__/version — provenance must degrade, not crash.
    return {"torch": getattr(torch, "__version__", None),
            "cuda": getattr(getattr(torch, "version", None), "cuda", None),
            "gpu": gpu}


def collect_provenance(repo_dir, extra_config: dict | None = None) -> dict:
    """Collect the provenance block for a run inside git repo `repo_dir`.

    extra_config keys override/extend the block (e.g. {"wall_s": wall,
    "branch": args.branch, "config": {...}} — use branch for detached shallow
    clones). Call this while the clone is still CLEAN: writers drop driver
    scripts into the repo root, which would flip git_dirty to true.
    """
    commit = _git(repo_dir, "rev-parse", "HEAD")            # full 40-char
    branch = _git(repo_dir, "rev-parse", "--abbrev-ref", "HEAD")
    if branch in (None, "HEAD"):                             # detached clone
        branch = "detached"
    dirty = _git(repo_dir, "status", "--porcelain")
    prov = {
        "schema_version": SCHEMA_VERSION,
        "commit": commit,
        "branch": branch,
        "git_dirty": (dirty != "") if dirty is not None else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        **_torch_info(),
        "env": {k: os.environ.get(k) for k in _ENV_WHITELIST},  # whitelist only
    }
    if extra_config:
        prov.update(extra_config)
    return prov


def mark_invalid(artefact: dict, reason: str, root_cause_commit: str) -> dict:
    """Stamp an artefact invalid in place; ensures a provenance block exists
    so the machine reader rejects it even if the writer crashed early."""
    prov = artefact.get("provenance")
    if not isinstance(prov, dict):
        prov = artefact["provenance"] = {"schema_version": SCHEMA_VERSION}
    prov["valid"] = False
    prov["invalid_reason"] = reason
    prov["root_cause_commit"] = root_cause_commit
    artefact["valid"] = False
    return artefact


def validate_provenance(artefact) -> list[str]:
    """Machine reader: list of rejection reasons; empty == valid.
    Hard rejects: missing provenance block; missing schema_version / commit /
    git_dirty; commit not hex; schema>=2 with non-40-char commit; valid:false
    anywhere; top-level commit inconsistent with provenance.commit."""
    if not isinstance(artefact, dict):
        return ["artefact is not a JSON object"]
    prov = artefact.get("provenance")
    if not isinstance(prov, dict):
        return ["missing provenance block"]
    errors: list[str] = []
    if artefact.get("valid") is False or prov.get("valid") is False:
        reason = prov.get("invalid_reason")
        errors.append("valid=false" + (f": {reason}" if reason else ""))
    sv = prov.get("schema_version")
    if sv is None:
        errors.append("missing schema_version")
    elif not isinstance(sv, int):
        errors.append(f"schema_version not an int: {sv!r}")
    commit = prov.get("commit")
    if not commit:
        errors.append("missing commit")
    elif not _ANY_SHA_RE.match(str(commit)):
        errors.append(f"commit not a hex sha: {commit!r}")
    elif isinstance(sv, int) and sv >= 2 and not _FULL_SHA_RE.match(str(commit)):
        errors.append(f"schema_version>=2 requires full 40-char commit, got {commit!r}")
    gd = prov.get("git_dirty")
    if gd is None:
        errors.append("missing git_dirty")
    elif not isinstance(gd, bool):
        errors.append(f"git_dirty not a bool: {gd!r}")
    top = artefact.get("commit")
    if isinstance(top, str) and commit and \
       not str(commit).startswith(top) and not top.startswith(str(commit)):
        errors.append(f"top-level commit {top!r} != provenance.commit {commit!r}")
    return errors


def write_artefact(repo_dir, filename: str, artefact: dict,
                   root_cause_commit: str | None = None,
                   docs_link: str = DEFAULT_DOCS_LINK) -> str:
    """Validate, then route: valid -> results/<filename>; invalid ->
    results/quarantine/<name>.<commit>.json + one line in results/INVALID.md.
    Never writes an invalid artefact into results/<filename>. Returns path."""
    results_dir = os.path.join(str(repo_dir), "results")
    os.makedirs(results_dir, exist_ok=True)
    problems = validate_provenance(artefact)
    if not problems:
        path = os.path.join(results_dir, filename)
        with open(path, "w") as fh:
            json.dump(artefact, fh, indent=2, sort_keys=True)
            fh.write("\n")
        return path
    prov = artefact.get("provenance") or {}
    mark_invalid(artefact, "; ".join(problems),
                 root_cause_commit or prov.get("commit") or "unknown")
    qdir = os.path.join(results_dir, "quarantine")
    os.makedirs(qdir, exist_ok=True)
    stem = filename[:-5] if filename.endswith(".json") else filename
    commit = prov.get("commit") or root_cause_commit or "unknown"
    path = os.path.join(qdir, f"{stem}.{commit}.json")
    with open(path, "w") as fh:
        json.dump(artefact, fh, indent=2, sort_keys=True)
        fh.write("\n")
    _append_ledger(results_dir, os.path.relpath(path, str(repo_dir)),
                   artefact, docs_link)
    return path


def _append_ledger(results_dir: str, rel_path: str, artefact: dict,
                   docs_link: str) -> None:
    prov = artefact.get("provenance") or {}
    ledger = os.path.join(results_dir, "INVALID.md")
    header = ("| file | commit | schema_version | valid | reason | "
              "root_cause_commit | docs link |\n|---|---|---|---|---|---|---|\n")
    if not os.path.exists(ledger):
        with open(ledger, "w") as fh:
            fh.write("# INVALID artefacts (quarantined)\n\n"
                     "Written by tools/artefact_provenance.write_artefact. "
                     "Copies live in results/quarantine/<name>.<commit>.json.\n\n")
            fh.write(header)
    reason = str(prov.get("invalid_reason", "")).replace("|", "\\|")
    line = ("| {f} | {c} | {sv} | {v} | {r} | {rc} | {d} |\n".format(
        f=rel_path, c=prov.get("commit"), sv=prov.get("schema_version"),
        v=prov.get("valid"), r=reason, rc=prov.get("root_cause_commit"),
        d=docs_link))
    with open(ledger, "a") as fh:
        fh.write(line)
