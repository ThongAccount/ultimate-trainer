"""Pre-compiled kernel loader — loads libpacked_ternary.so once.

Usage:
    1. Build:   cd kernels/packed_ternary && make -j$(nproc)
    2. Load:    from kernels.packed_ternary.prebuilt import get_lib
                lib = get_lib()
                lib.launch_packed_ternary_tc(...)

Falls back to load_inline JIT if the .so doesn't exist.
"""

import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SO_PATH = os.path.join(HERE, "build", "libpacked_ternary.so")

_lib = None


def is_available() -> bool:
    """Check if pre-built .so exists."""
    return os.path.isfile(SO_PATH)


def get_lib():
    """Load the pre-built shared library (once)."""
    global _lib
    if _lib is not None:
        return _lib
    if not is_available():
        raise FileNotFoundError(
            f"Pre-built kernel library not found at {SO_PATH}\n"
            f"Build with: cd {HERE} && make -j$(nproc)"
        )
    _lib = torch.ops.load_library(SO_PATH)
    return _lib


def build_if_needed():
    """Build the .so if it doesn't exist or is outdated."""
    if is_available():
        return
    import subprocess
    print("[packed_ternary] Building kernels...")
    subprocess.check_call(["make", f"-j{os.cpu_count()}", "all"], cwd=HERE)
    print("[packed_ternary] Build complete.")
