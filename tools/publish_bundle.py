#!/usr/bin/env python3
"""Publish the kernel bundle to the `kernels-latest` GitHub release.

Stdlib only — the CUDA CI container has neither curl nor gh. Creates the
release if missing, deletes same-name assets (clobber), uploads
kernels-bundle.tar.gz + .sha256 + manifest.json from dist/.

Usage: GH_TOKEN=... python tools/publish_bundle.py [dist_dir]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

REPO = "ThongAccount/ultimate-trainer"
TAG = os.environ.get("UT_KERNELS_TAG", "kernels-latest")
API = f"https://api.github.com/repos/{REPO}"
UP = f"https://uploads.github.com/repos/{REPO}"
FILES = ["kernels-bundle.tar.gz", "kernels-bundle.tar.gz.sha256", "manifest.json"]


def req(method, url, token, data=None, ctype=None):
    r = urllib.request.Request(url, method=method, data=data, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "ut-publish-bundle",
        **({"Content-Type": ctype} if ctype else {}),
    })
    return urllib.request.urlopen(r, timeout=300)


def main() -> int:
    token = os.environ.get("GH_TOKEN", "")
    if not token:
        print("FATAL: GH_TOKEN not set")
        return 1
    dist = sys.argv[1] if len(sys.argv) > 1 else "dist"

    bundle = os.path.join(dist, "kernels-bundle.tar.gz")
    with open(bundle, "rb") as f:
        blob = f.read()
    sha = hashlib.sha256(blob).hexdigest()
    with open(os.path.join(dist, "kernels-bundle.tar.gz.sha256"), "w") as f:
        f.write(f"{sha}  {bundle}\n")
    print(f"bundle sha256={sha} size={len(blob)}", flush=True)

    try:
        rel = json.load(req("GET", f"{API}/releases/tags/{TAG}", token))
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        rel = json.load(req("POST", f"{API}/releases", token, data=json.dumps({
            "tag_name": TAG,
            "name": f"{TAG} (auto-built)",
            "body": ("Auto-built load_inline extension bundle (py313_cu130, "
                     "sm_75, torch 2.11.0+cu130). Overwritten by the "
                     "build-kernels workflow on every kernels/** change. "
                     "Do not edit."),
            "prerelease": True,
            "draft": False,
        }).encode(), ctype="application/json"))
    print(f"release {TAG} id={rel['id']}", flush=True)

    for a in rel.get("assets", []):
        if a["name"] in FILES:
            req("DELETE", f"{API}/releases/assets/{a['id']}", token)
            print(f"deleted old asset {a['name']}", flush=True)

    ctypes = {".tar.gz": "application/gzip", ".sha256": "text/plain",
              ".json": "application/json"}
    for name in FILES:
        data = open(os.path.join(dist, name), "rb").read()
        ext = name[name.rfind("."):]
        up = json.load(req("POST",
                           f"{UP}/releases/{rel['id']}/assets?name={name}",
                           token, data=data, ctype=ctypes.get(ext,
                                                              "application/octet-stream")))
        print(f"uploaded {name} ({len(data)} bytes) asset id {up['id']}", flush=True)
    print(f"RELEASE: {TAG} updated with {', '.join(FILES)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
