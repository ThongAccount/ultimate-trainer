"""Speedpass kernel parity tests — fused combine + block-sparse ternary on GPU.

These run the actual CUDA kernels (not the eager fallbacks) and assert
parity against PyTorch references, with CUDA_LAUNCH_BLOCKING=1 so OOB
errors surface at the exact kernel call.
"""

import os

import torch
import pytest

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from kernels.subqsa_combine.subqsa_combine import (
    SubQSACombineFn, _subqsa_combine_eager, _HAS_SUBQSA_COMBINE,
)
from kernels.block_sparse_ternary.block_sparse_ternary import (
    block_sparse_ternary_matmul, compute_block_mask, _HAS_CUDA,
)


@pytest.fixture(scope="module")
def tensors():
    torch.manual_seed(0)
    B, T, H, D = 2, 64, 2, 32
    DO = H * D
    return {
        "B": B, "T": T, "H": H, "D": D, "DO": DO,
        "x": torch.randn(B, T, DO, device="cuda"),
        "o_cmp": torch.randn(B, H, T, D, device="cuda"),
        "o_slc": torch.randn(B, H, T, D, device="cuda"),
        "o_win": torch.randn(B, H, T, D, device="cuda"),
        "gate_w1": torch.randn(64, DO, device="cuda") * 0.1,
        "gate_w2": torch.randn(3 * H, 64, device="cuda") * 0.1,
        "out_norm_weight": torch.randn(DO, device="cuda"),
        "o_proj_weight": torch.randn(DO, DO, device="cuda") * 0.05,
        "gamma": 0.1,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fused_combine_kernel_parity(tensors):
    """Fused CUDA combine kernel matches eager reference exactly."""
    if not _HAS_SUBQSA_COMBINE:
        pytest.skip("subqsa_combine CUDA extension unavailable")
    kw = {k: tensors[k] for k in
          ("x", "o_cmp", "o_slc", "o_win", "gate_w1", "gate_w2",
           "out_norm_weight", "o_proj_weight", "gamma")}
    ref = _subqsa_combine_eager(**kw, block_mask=None)
    y = SubQSACombineFn.apply(
        kw["x"], kw["o_cmp"], kw["o_slc"], kw["o_win"],
        kw["gate_w1"], kw["gate_w2"], kw["out_norm_weight"],
        kw["o_proj_weight"], kw["gamma"], None,
    )
    torch.cuda.synchronize()
    err = (y - ref).abs().max().item()
    assert err < 1e-3, f"fused kernel parity err {err}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_sparse_ternary_kernel_parity(tensors):
    """Block-sparse ternary CUDA kernel matches masked reference."""
    if not _HAS_CUDA:
        pytest.skip("block_sparse_ternary CUDA extension unavailable")
    DO = tensors["DO"]
    opw = tensors["o_proj_weight"]
    gamma = tensors["gamma"]

    num_n = (DO + 63) // 64
    num_k = (DO + 15) // 16
    mask = compute_block_mask(torch.tensor([[0, 1]], device="cuda"), 2, 64, num_n, num_k)

    xm = torch.randn(tensors["B"] * tensors["T"], DO, device="cuda")
    y = block_sparse_ternary_matmul(xm, opw, gamma, mask)
    torch.cuda.synchronize()

    wq = torch.clamp(torch.round(opw / gamma), -1, 1) * gamma
    ref = xm @ wq.t()
    for tn in range(num_n):
        for tk in range(num_k):
            bit = tn * num_k + tk
            if not (mask[bit // 64] & (1 << (bit % 64))):
                ref[:, tn * 64:(tn + 1) * 64] = 0.0
    err = (y - ref).abs().max().item()
    assert err < 1e-3, f"sparse kernel parity err {err}"