#!/usr/bin/env python3
"""Build every load_inline CUDA extension the Colab production path uses and
pack the torch extension cache into dist/kernels-bundle.tar.gz.

Runs in CI (.github/workflows/build-kernels.yml) on a CUDA-devel container
with torch==2.11.0+cu130 and python 3.13 — matching the Colab T4 VM exactly
(artefact provenance: python 3.13.16, torch 2.11.0+cu130, cuda 13.0). No GPU
is needed to COMPILE; the resulting .so files load on any sm_75 VM whose
torch extension tag (py313_cu130) matches. TORCH_CUDA_ARCH_LIST=7.5 is set
centrally by kernels/common.py.

The bundle is the whole TORCH_EXTENSIONS_DIR (minus object/PTX intermediates):
load_inline skips nvcc when the cached `version` hash matches the sources,
so extracting this bundle into TORCH_EXTENSIONS_DIR on the VM turns every
JIT build into a plain dlopen.

Usage:
    python tools/build_kernel_bundle.py --out dist
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import surface of the production path (train_gigatoken.py + the three
# scripts/colab_*.py drivers). Import-time extensions build on import; the
# lazy `_load*` / `_ensure_loaded` loaders are forced below.
KERNEL_MODULES = [
    "kernels",
    "kernels.elementwise.addsub",
    "kernels.ternary.ternary_matmul",
    "kernels.fused_ternary.fused_ternary",
    "kernels.compressed_attn.compressed_attn",
    "kernels.selective_attn.selective_attn",
    "kernels.block_sparse_ternary.block_sparse_ternary",
    "kernels.subqsa_combine.subqsa_combine",
    "kernels.packed_ternary",
    "kernels.packed_ternary.pack_forward",
    "kernels.packed_ternary.pack_update",
    "kernels.packed_ternary.pack_encode",
    "kernels.packed_ternary.pack_decode",
    "kernels.packed_ternary.custom_ops",
    "kernels.packed_ternary.update_cublas",
    "kernels.packed_ternary.cublas_backend",
]

# Intermediates that slim the bundle but are never needed to *load* a build.
_EXCLUDE_SUFFIXES = (".o", ".ptx", ".cubin", ".fatbin", ".ninja_deps", ".ninja_log")


def force_builds() -> int:
    """Import every kernel module, then call every zero-arg lazy loader."""
    n_ok = 0
    for modname in KERNEL_MODULES:
        mod = importlib.import_module(modname)
        for attr in sorted(dir(mod)):
            if attr.startswith("_load") or attr == "_ensure_loaded":
                fn = getattr(mod, attr)
                if callable(fn):
                    try:
                        fn()
                        n_ok += 1
                        print(f"  [built] {modname}.{attr}", flush=True)
                    except Exception as e:  # noqa: BLE001 — record, keep building
                        print(f"  [warn ] {modname}.{attr}: {e!r}", flush=True)
    return n_ok


def py_tag() -> str:
    import torch

    tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    cu = getattr(torch.version, "cuda", None)
    if cu:
        tag += "_cu" + cu.replace(".", "")
    return tag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "dist"))
    args = ap.parse_args()

    sys.path.insert(0, REPO)
    work = tempfile.mkdtemp(prefix="ut_ext_bundle_")
    ext_root = os.path.join(work, "torch_extensions")
    os.environ["TORCH_EXTENSIONS_DIR"] = ext_root  # must precede any load_inline

    import torch

    print(f"torch={torch.__version__} cuda={torch.version.cuda} "
          f"python={sys.version.split()[0]} TORCH_CUDA_ARCH_LIST="
          f"{os.environ.get('TORCH_CUDA_ARCH_LIST')}", flush=True)

    t0 = time.time()
    n = force_builds()
    dt = time.time() - t0

    tag = py_tag()
    if not os.path.isdir(os.path.join(ext_root, tag)):
        print(f"FATAL: expected extension tag dir {tag} under {ext_root}; "
              f"found {os.listdir(ext_root) if os.path.isdir(ext_root) else 'none'}")
        return 1

    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    nvcc = "n/a"
    if shutil.which("nvcc"):
        nvcc = subprocess.run(["nvcc", "--version"], capture_output=True,
                              text=True).stdout.strip().splitlines()[-1]

    os.makedirs(args.out, exist_ok=True)
    bundle = os.path.join(args.out, "kernels-bundle.tar.gz")

    def keep(member):
        return None if member.name.endswith(_EXCLUDE_SUFFIXES) else member

    with tarfile.open(bundle, "w:gz") as tf:
        tf.add(ext_root, arcname=".", filter=keep)

    manifest = {
        "commit": commit,
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "python": sys.version.split()[0],
        "py_tag": tag,
        "arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "nvcc": nvcc,
        "extensions_built": n,
        "build_seconds": round(dt, 1),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    sha = hashlib.sha256(open(bundle, "rb").read()).hexdigest()
    size_mb = os.path.getsize(bundle) / 1e6
    print(f"BUNDLE: {bundle} sha256={sha} size={size_mb:.1f}MB exts={n} "
          f"build_s={dt:.0f} py_tag={tag} commit={commit[:10]}", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
