"""CPU-only tests for tools/artefact_provenance.py — no torch required.

Run: python -m pytest tests/test_artefact_provenance.py -q
The module under test imports torch lazily; with torch absent all torch
fields degrade to None, which these tests assert explicitly.
"""
import json
import os
import platform
import re
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.artefact_provenance import (  # noqa: E402
    SCHEMA_VERSION,
    collect_provenance,
    mark_invalid,
    validate_provenance,
    write_artefact,
)


def _init_repo(path):
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"],
                   check=True, capture_output=True)
    (path / "f.txt").write_text("x")
    subprocess.run(["git", "-C", str(path), "add", "f.txt"], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "init"],
                   check=True, capture_output=True)


def test_clean_repo_full_sha(tmp_path):
    _init_repo(tmp_path)
    prov = collect_provenance(tmp_path)
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert prov["commit"] == head
    assert re.fullmatch(r"[0-9a-f]{40}", prov["commit"])  # full sha, not prefix
    assert prov["git_dirty"] is False
    assert prov["branch"] == "main"
    assert prov["schema_version"] == SCHEMA_VERSION
    assert prov["python"] == platform.python_version()
    assert set(prov["env"]) == {"PT_BACKEND", "PT_UPDATE_BACKEND", "TORCH_CUDA_ARCH_LIST"}


def test_dirty_states(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "untracked.txt").write_text("u")
    assert collect_provenance(tmp_path)["git_dirty"] is True
    (tmp_path / "f.txt").write_text("modified")
    assert collect_provenance(tmp_path)["git_dirty"] is True


def test_not_a_repo_degrades(tmp_path):
    prov = collect_provenance(tmp_path)  # bare dir, no git
    assert prov["commit"] is None
    assert prov["git_dirty"] is None
    errors = validate_provenance({"provenance": prov})
    assert any("commit" in e for e in errors)
    assert any("git_dirty" in e for e in errors)


def test_no_torch_degrades(monkeypatch, tmp_path):
    _init_repo(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch raises
    prov = collect_provenance(tmp_path)
    assert prov["torch"] is None and prov["cuda"] is None and prov["gpu"] is None


def test_env_whitelist_no_secrets(monkeypatch, tmp_path):
    _init_repo(tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "secret-value-123")
    monkeypatch.setenv("PT_BACKEND", "cublas")
    prov = collect_provenance(tmp_path)
    assert set(prov["env"]) == {"PT_BACKEND", "PT_UPDATE_BACKEND", "TORCH_CUDA_ARCH_LIST"}
    assert prov["env"]["PT_BACKEND"] == "cublas"
    assert "secret-value-123" not in json.dumps(prov)


def test_validate_matrix():
    assert validate_provenance({}) == ["missing provenance block"]
    assert validate_provenance("not-a-dict") == ["artefact is not a JSON object"]
    bad = validate_provenance({"provenance": {}})
    assert any("schema_version" in e for e in bad)
    assert any("commit" in e for e in bad)
    assert any("git_dirty" in e for e in bad)
    # schema 2 requires full 40-char commit
    v2short = {"provenance": {"schema_version": 2, "commit": "abc1234567",
                               "git_dirty": False}}
    assert any("40-char" in e for e in validate_provenance(v2short))
    # legacy schema 1 accepts a prefix
    v1 = {"provenance": {"schema_version": 1, "commit": "abc1234567",
                         "git_dirty": False}}
    assert validate_provenance(v1) == []
    # git_dirty must be bool
    badgd = {"provenance": {"schema_version": 1, "commit": "abc1234567",
                            "git_dirty": "yes"}}
    assert any("git_dirty" in e for e in validate_provenance(badgd))
    # valid:false rejects and surfaces the reason
    inv = {"provenance": {"schema_version": 2, "commit": "a" * 40,
                          "git_dirty": False, "valid": False,
                          "invalid_reason": "commit-mismatch refusal (rc=2)"}}
    errs = validate_provenance(inv)
    assert any("valid=false" in e and "commit-mismatch" in e for e in errs)
    # top-level commit inconsistent with provenance.commit
    mm = {"commit": "f" * 40,
          "provenance": {"schema_version": 2, "commit": "a" * 40, "git_dirty": False}}
    assert any("!=" in e for e in validate_provenance(mm))
    # well-formed v2 passes
    ok = {"commit": "a" * 40,
          "provenance": {"schema_version": 2, "commit": "a" * 40, "git_dirty": False}}
    assert validate_provenance(ok) == []


def test_mark_invalid():
    a = {"provenance": {"schema_version": 2}}
    a = mark_invalid(a, "boom", "b" * 40)
    assert a["valid"] is False
    assert a["provenance"]["valid"] is False
    assert a["provenance"]["invalid_reason"] == "boom"
    assert a["provenance"]["root_cause_commit"] == "b" * 40
    assert validate_provenance(a)


def test_write_artefact_routing(tmp_path):
    _init_repo(tmp_path)
    prov = collect_provenance(tmp_path)
    good = {"commit": prov["commit"], "provenance": prov}
    p = write_artefact(tmp_path, "ok.json", good)
    assert p.endswith(os.path.join("results", "ok.json"))
    assert os.path.exists(p)
    assert not os.path.exists(os.path.join(str(tmp_path), "results", "quarantine"))
    # invalid -> quarantine + ledger, never results/<name>.json
    bad = {"commit": prov["commit"], "provenance": {"schema_version": 2}}  # missing fields
    p2 = write_artefact(tmp_path, "bad.json", bad, root_cause_commit=prov["commit"])
    assert "quarantine" in p2 and prov["commit"] in p2
    assert not os.path.exists(os.path.join(str(tmp_path), "results", "bad.json"))
    ledger = os.path.join(str(tmp_path), "results", "INVALID.md")
    lines = open(ledger).read().splitlines()
    table = [ln for ln in lines if ln.startswith("| results/quarantine/")]
    assert len(table) == 1 and len(table[0].split("|")) == 9  # 7 fields + edges


def test_extra_config_overrides(tmp_path):
    _init_repo(tmp_path)
    prov = collect_provenance(tmp_path, extra_config={"wall_s": 12.5, "branch": "feat/x"})
    assert prov["wall_s"] == 12.5
    assert prov["branch"] == "feat/x"
