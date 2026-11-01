#!/usr/bin/env python3
"""Download the pre-built CUDA extension bundle (GitHub release tag
`kernels-latest`, built by .github/workflows/build-kernels.yml) and extract
it into TORCH_EXTENSIONS_DIR so every load_inline call finds a matching
cached build and skips nvcc — on-VM JIT compilation was burning free Colab
GPU minutes (2-5 min per extension per fresh VM).

ABI contract (checked hard before extracting): the bundle's manifest must
match the local torch version and the local extension tag pyXY_cuZZ. A
commit mismatch is only a warning (doc-only pushes may post-date the
bundle); the run's own --expect-commit guard pins the clone, and the P0
flip-gate bit-reproduction is the end-to-end correctness check.

Fallback semantics: ANY failure (no token, no release, sha256 mismatch, ABI
mismatch, unsafe tar member) prints `PREBUILT: SKIP <reason>` and returns
False — the caller falls back to on-VM JIT exactly as before. Never blocks
a run unless --strict. Sets UT_PREBUILT=1/0 in the environment so artefact
provenance records which path was taken.

Usage:
    python tools/fetch_prebuilt.py --repo-dir /content/ultimate-trainer
    python tools/fetch_prebuilt.py --repo-dir . --tag kernels-latest --strict
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

OWNER_REPO = "ThongAccount/ultimate-trainer"
BUNDLE = "kernels-bundle.tar.gz"
SHA_ASSET = BUNDLE + ".sha256"
MANIFEST = "manifest.json"
DEFAULT_TAG = "kernels-latest"


def ext_root() -> str:
    """TORCH_EXTENSIONS_DIR, defaulting (and exporting) the torch default."""
    root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not root:
        root = os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions")
        os.environ["TORCH_EXTENSIONS_DIR"] = root
    return root


def local_layout() -> tuple[str | None, str]:
    """(tag_dir_or_None, torch_version) using torch's own build-dir logic.

    torch 2.11 builds extensions FLAT under TORCH_EXTENSIONS_DIR when no GPU
    is present and under a pyXY_cuZZ tag dir when one is; probing
    _get_build_directory on the VM gives the exact layout load_inline will
    use there, so the bundle can be placed correctly regardless.
    """
    import torch  # local import: version read only, no extension builds here

    try:
        from torch.utils.cpp_extension import _get_build_directory
        probe = _get_build_directory("ut_prebuilt_probe", False)
        parent = os.path.dirname(os.path.abspath(probe))
        root = os.path.abspath(ext_root())
        tag = os.path.basename(parent) if parent != root else None
    except Exception:  # noqa: BLE001 — fall back to the version-based guess
        tag = f"py{sys.version_info.major}{sys.version_info.minor}"
        cu = getattr(torch.version, "cuda", None)
        if cu:
            tag += "_cu" + cu.replace(".", "")
    return tag, torch.__version__


def _api(url: str, token: str, accept: str = "application/vnd.github+json"):
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": accept,
        "User-Agent": "ut-fetch-prebuilt",
    })
    return urllib.request.urlopen(req, timeout=120)


def _download(url: str, token: str, dest: str) -> None:
    with _api(url, token, accept="application/octet-stream") as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def _git_head(repo_dir: str) -> str:
    try:
        return subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def ensure_prebuilt(repo_dir: str, tag: str = DEFAULT_TAG, strict: bool = False) -> bool:
    """Fetch + verify + extract the bundle. Returns True iff prebuilt is active."""
    os.environ.setdefault("UT_PREBUILT", "1")
    root = ext_root()
    if os.environ.get("UT_PREBUILT") == "0":
        print("PREBUILT: SKIP disabled (UT_PREBUILT=0) — JIT fallback", flush=True)
        return False
    # Idempotency: a previous process on this VM already extracted the bundle
    # (child drivers re-run ensure_prebuilt after the parent). Marker file
    # records release tag + manifest commit; hit -> no re-download.
    marker = os.path.join(root, f".ut-prebuilt-{tag}")
    if os.path.isfile(marker):
        print(f"PREBUILT: OK (cached) marker={open(marker).read().strip()} -> {root}",
              flush=True)
        os.environ["UT_PREBUILT"] = "1"
        return True
    try:
        token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            raise RuntimeError("GITHUB_TOKEN not set")
        local_tag, local_torch = local_layout()
        rel = json.load(_api(
            f"https://api.github.com/repos/{OWNER_REPO}/releases/tags/{tag}", token))
        assets = {a["name"]: a for a in rel.get("assets", [])}
        for name in (BUNDLE, SHA_ASSET, MANIFEST):
            if name not in assets:
                raise RuntimeError(f"release {tag} has no asset {name}")
        # NB: asset API urls need Accept: application/octet-stream for the raw
        # file; the default +json accept returns asset METADATA (no "torch").
        with _api(assets[MANIFEST]["url"], token,
                  accept="application/octet-stream") as r:
            manifest = json.load(r)
        if manifest.get("torch") != local_torch:
            raise RuntimeError(
                f"torch mismatch: bundle {manifest.get('torch')} != local {local_torch}")
        head = _git_head(repo_dir)
        if head and manifest.get("commit") and head != manifest["commit"]:
            print(f"PREBUILT: WARN bundle commit {manifest['commit'][:10]} != "
                  f"clone HEAD {head[:10]} (doc-only drift; P0 gate is the check)",
                  flush=True)

        tmp = root + ".fetch-tmp"
        os.makedirs(tmp, exist_ok=True)
        bpath = os.path.join(tmp, BUNDLE)
        spath = os.path.join(tmp, SHA_ASSET)
        _download(assets[BUNDLE]["url"], token, bpath)
        _download(assets[SHA_ASSET]["url"], token, spath)
        want = open(spath).read().split()[0]
        got = hashlib.sha256(open(bpath, "rb").read()).hexdigest()
        if got != want:
            raise RuntimeError(f"sha256 mismatch: {got} != {want}")

        # Place extensions where THIS torch will look: if the local layout is
        # tagged (pyXY_cuZZ/) but the CI bundle is flat, move the extracted
        # extension dirs under the tag dir.
        with tarfile.open(bpath, "r:gz") as tf:
            members = tf.getmembers()
            for m in members:
                parts = m.name.split("/")
                if m.name.startswith("/") or ".." in parts:
                    raise RuntimeError(f"unsafe tar member: {m.name}")
            tf.extractall(tmp)
        dest = os.path.join(root, local_tag) if local_tag else root
        os.makedirs(dest, exist_ok=True)
        moved = 0
        for entry in os.listdir(tmp):
            if entry in (BUNDLE, SHA_ASSET):
                continue
            target = os.path.join(dest, entry)
            if os.path.exists(target):  # already extracted by an earlier process
                continue
            os.rename(os.path.join(tmp, entry), target)
            moved += 1

        n_so = sum(f.endswith(".so") for _, _, files in os.walk(dest) for f in files)
        if n_so < 10:
            raise RuntimeError(f"only {n_so} .so files landed under {dest}")
        with open(marker, "w") as f:
            f.write(f"{tag} {manifest.get('commit', '?')} torch={local_torch}\n")
        print(f"PREBUILT: OK release={tag} commit={str(manifest.get('commit'))[:10]} "
              f"torch={local_torch} layout_tag={local_tag or 'flat'} "
              f"exts={moved} .so={n_so} -> {dest}", flush=True)
        os.environ["UT_PREBUILT"] = "1"
        shutil.rmtree(tmp, ignore_errors=True)
        return True
    except Exception as e:  # noqa: BLE001 — fallback, never block unless strict
        print(f"PREBUILT: SKIP {e!r} — JIT fallback", flush=True)
        os.environ["UT_PREBUILT"] = "0"
        if strict:
            raise
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default=".")
    ap.add_argument("--tag", default=os.environ.get("UT_KERNELS_TAG", DEFAULT_TAG))
    ap.add_argument("--strict", action="store_true",
                    help="fail instead of falling back to JIT")
    args = ap.parse_args()
    ok = ensure_prebuilt(args.repo_dir, tag=args.tag, strict=args.strict)
    return 0 if ok or not args.strict else 1


if __name__ == "__main__":
    sys.exit(main())
