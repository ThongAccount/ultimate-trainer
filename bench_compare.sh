#!/usr/bin/env bash
# bench_compare.sh — clean-state Discrete vs AdamW benchmark for GPU env.
#
# Guarantees a valid comparison:
#   1. Hard-syncs to origin/master (discards any stale local state).
#   2. Wipes the torch extension cache (kills stale .so symbol mismatches).
#   3. Runs the Shakespeare comparison with configurable steps.
#
# Usage:  bash bench_compare.sh [steps]   (default 200)

set -euo pipefail

STEPS="${1:-200}"

echo "== [1/4] git sync =="
git fetch origin
git reset --hard origin/master
git log --oneline -3

echo "== [2/4] clean torch extension cache =="
rm -rf "${TORCH_EXTENSIONS_DIR:-/root/.cache/torch_extensions}"/*

echo "== [3/4] deps + device check =="
python -c "import ninja" 2>/dev/null || { echo "installing ninja..."; pip install -q ninja; }

python - <<'PY'
import torch
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | device: "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
assert torch.cuda.is_available(), "benchmark requires a CUDA device"
PY

echo "== [4/4] benchmark (steps=${STEPS}) =="
python train_shakespeare_optimized.py --compare --steps "${STEPS}"