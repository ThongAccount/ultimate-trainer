"""Shared kernel-build defaults.

Imported first by kernels/__init__.py so the env var is set before any
submodule calls torch.utils.cpp_extension.load_inline.
"""
import os

# Kernels JIT via load_inline against the INSTALLED torch ABI; default to the
# Colab T4 target (sm_75). Override via the environment for other GPUs.
DEFAULT_TORCH_CUDA_ARCH_LIST = "7.5"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", DEFAULT_TORCH_CUDA_ARCH_LIST)
