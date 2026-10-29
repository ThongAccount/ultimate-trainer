#!/usr/bin/env bash
# Colab T4 session setup: clone/pull repo at a pinned commit, install deps,
# prebuild kernels (TORCH_CUDA_ARCH_LIST defaults to 7.5 centrally in kernels/common.py).
#
# Usage: bash scripts/colab_setup.sh [commit] [repo-dir]
# Every benchmark record must quote the git commit from the JSON output.
set -euo pipefail

COMMIT="${1:-HEAD}"
DEST="${2:-/content/ultimate-trainer}"
REPO_URL="https://github.com/ThongAccount/ultimate-trainer.git"
# Stale-clone guard: export EXPECT_COMMIT (or PT_EXPECT_COMMIT) to refuse
# running unless the resulting HEAD matches (prefix ok). Exit 2 = invalid.
EXPECT_COMMIT="${EXPECT_COMMIT:-${PT_EXPECT_COMMIT:-}}"

echo "---- nvidia-smi ----"
nvidia-smi --query-gpu=name,compute_cap,clocks.max.sm,clocks.sm,power.limit --format=csv || true

echo "---- python/torch ----"
python3 - <<'EOF'
import sys, subprocess
print("python:", sys.version.split()[0])
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))
except ImportError:
    print("torch: NOT INSTALLED")
EOF

echo "---- clone/pull ${COMMIT} -> ${DEST} ----"
if [ -d "${DEST}/.git" ]; then
    git -C "${DEST}" fetch --all --quiet
else
    git clone --quiet "${REPO_URL}" "${DEST}"
fi
git -C "${DEST}" checkout --quiet "${COMMIT}"
if [ "${COMMIT}" = "HEAD" ]; then
    git -C "${DEST}" pull --quiet origin "$(git -C "${DEST}" branch --show-current)" \
        || { echo "git pull failed" >&2; exit 2; }
fi
HEAD_SHA="$(git -C "${DEST}" rev-parse HEAD)"
if [ -n "${EXPECT_COMMIT}" ] && [ "${HEAD_SHA#"${EXPECT_COMMIT}"}" = "${HEAD_SHA}" ]; then
    echo "COMMIT-MISMATCH: HEAD=${HEAD_SHA} does not match EXPECT_COMMIT=${EXPECT_COMMIT}; refusing to run" >&2
    exit 2
fi
echo "commit: ${HEAD_SHA}"

echo "---- deps ----"
pip install -q ninja >/dev/null 2>&1 || pip install -q ninja
pip install -q -r "${DEST}/requirements.txt" >/dev/null 2>&1 || pip install -q -r "${DEST}/requirements.txt"

echo "---- import smoke ----"
cd "${DEST}"
python3 - <<'EOF'
import os, sys
sys.path.insert(0, os.getcwd())
import torch
assert torch.cuda.is_available(), "no CUDA"
name = torch.cuda.get_device_name(0)
assert torch.cuda.get_device_capability(0) == (7, 5), f"NOT a T4: {name} {torch.cuda.get_device_capability(0)}"
print(f"T4 confirmed: {name}")
from kernels.packed_ternary import pack_forward, pack_update
print("kernels import OK")
EOF
echo "SETUP DONE"
