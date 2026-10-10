"""Session fixture: on a CUDA machine, hard-fail if any production kernel
fails to load (sweep #19: every CUDA test silently skipped/returned on
load failure — 66 passed ≠ 66 ran).

Loading is attempted here (session-scoped); the flags are checked after each
module's own lazy-load has run by simply asserting the flags at session start
after forcing all loads.  If CUDA is present but a prod kernel flag is None,
pytest fails immediately instead of letting dependent tests silently pass.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


def pytest_sessionstart(session):
    if os.environ.get("OMP_SKIP_KERNEL_GATE") == "1":
        return
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return  # CPU box: CUDA tests legitimately skip — expected locally.

    from kernels.packed_ternary import pack_update as pu
    from kernels.packed_ternary import pack_forward as pf

    # Force every production kernel load attempt.
    pu._load_dx_tc_32()
    pu._load_dx_tc()
    pu._load_up_tc_v2()
    pu._load_up_tc_v2_32()
    pf._load_tc_32()
    pf._load_tc_64()

    missing = []
    if pu._dx_tc_32_fn is None:
        missing.append("dx_tc_32 (gemm_backward_dx_tc_32.cu)")
    if pu._dx_tc_fn is None:
        missing.append("dx_tc (gemm_backward_dx_tc.cu)")
    if pu._up_tc_v2_fn is None:
        missing.append("up_tc_v2 (gemm_update_tc_v2.cu)")
    if pu._up_tc_v2_32_fn is None:
        missing.append("up_tc_v2_32 (gemm_update_tc_v2_32.cu)")
    if pf._forward_fn_tc is None:
        missing.append("forward_tc_32 (gemm_forward_tc_32.cu)")
    if pf._forward_fn_tc_64 is None:
        missing.append("forward_tc_64 (gemm_forward_tc.cu)")

    if missing:
        pytest.exit(
            f"PROD KERNEL GATE: CUDA present but kernels failed to load: "
            f"{', '.join(missing)}",
            returncode=1,
        )
