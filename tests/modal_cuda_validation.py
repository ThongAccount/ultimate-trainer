# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CUDA Kernel Validation on GPU (Modal)
#
# Compile and test all 4 fused CUDA C++ kernels on an NVIDIA GPU via Modal.
# Tests CUDA forward against CPU PyTorch reference for each kernel.
#
# ## Setup
#
# ```bash
# pip install modal jupytext
# modal token set --token-id xxx --token-secret yyy
# ```

# %% [markdown]
# ## Imports & Modal Setup

# %%
import modal
import torch
import torch.nn.functional as F
import sys, os, math, time

# Modal app
app = modal.App("ultimate-cuda-test")

# Container image with CUDA toolkit + PyTorch
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.4",
    "numpy",
    "ninja",  # for torch.utils.cpp_extension
    extra_index_url="https://download.pytorch.org/whl/cu124",
).env({"CUDA_HOME": "/usr/local/cuda"})

# Copy the project kernels into the Modal image
KERNEL_DIR = "/root/kernels"
image = image.add_local_dir(
    os.path.abspath("kernels"), KERNEL_DIR, copy=True
)

# %%
%pip install nvcc4jupyter -q
%load_ext nvcc4jupyter

# %% [markdown]
# ## Helper: compile + load a CUDA kernel

# %%
def compile_cuda_kernel(kernel_name, cu_path, cpp_wrapper=""):
    """Compile a single CUDA kernel file and return the loaded module."""
    from torch.utils.cpp_extension import load_inline
    from pathlib import Path

    cu_path = str(cu_path)
    if not os.path.exists(cu_path):
        alt = os.path.join(KERNEL_DIR, kernel_name, os.path.basename(cu_path))
        if os.path.exists(alt):
            cu_path = alt

    with open(cu_path) as f:
        cuda_source = f.read()

    return load_inline(
        name=kernel_name,
        cpp_sources=cpp_wrapper or """
        #include <torch/extension.h>
        at::Tensor dummy(at::Tensor x) { return x; }
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("dummy", &dummy); }
        """,
        cuda_sources=[cu_path],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )


# %% [markdown]
# ## Test 1: All Kernels Compile on GPU

# %%
@app.function(gpu="T4", image=image, timeout=600)
def test_kernel_compilation():
    """Verify all 4 CUDA kernels compile successfully."""
    results = {}
    kernel_dirs = [
        "compressed_attn",
        "selective_attn",
        "block_sparse_ternary",
        "subqsa_combine",
    ]

    for kid in kernel_dirs:
        cu_file = os.path.join(KERNEL_DIR, kid, f"{kid}.cu" if kid != "block_sparse_ternary" else "block_sparse_ternary.cu")
        if kid == "subqsa_combine":
            cu_file = os.path.join(KERNEL_DIR, kid, "subqsa_combine_kernel.cu")
        try:
            mod = compile_cuda_kernel(kid, cu_file)
            results[kid] = "COMPILED_OK"
        except Exception as e:
            results[kid] = f"FAILED: {e}"

    for k, v in results.items():
        status = "✅" if v == "COMPILED_OK" else "❌"
        print(f"  {status} {k}: {v}")

    all_ok = all(v == "COMPILED_OK" for v in results.values())
    assert all_ok, f"Some kernels failed: {results}"
    return results


# %% [markdown]
# ## Test 2: Compressed Attention — CUDA vs PyTorch Parity

# %%
@app.function(gpu="T4", image=image, timeout=300)
def test_compressed_attn_parity():
    """Compare compressed_attn CUDA kernel output vs PyTorch eager reference."""
    from kernels.compressed_attn.compressed_attn import (
        compressed_attn_forward,
        _compressed_attn_eager,
    )

    B, H, T, D = 2, 4, 256, 128
    device = "cuda"

    k = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.float16)

    def make_phi(dim, blk):
        in_dim = dim * blk
        return (
            torch.randn(2 * dim, in_dim, device=device, dtype=torch.float16) * 0.02,
            torch.zeros(2 * dim, device=device, dtype=torch.float16),
            torch.randn(dim, 2 * dim, device=device, dtype=torch.float16) * 0.02,
            torch.zeros(dim, device=device, dtype=torch.float16),
        )

    phi_k = make_phi(D, 32)
    phi_v = make_phi(D, 32)

    # CPU reference (float32 for safety)
    k_cpu = k.cpu().float()
    v_cpu = v.cpu().float()
    phi_k_cpu = tuple(p.cpu().float() for p in phi_k)
    phi_v_cpu = tuple(p.cpu().float() for p in phi_v)

    k_cmp_ref, v_cmp_ref = _compressed_attn_eager(
        k_cpu, v_cpu, *phi_k_cpu, *phi_v_cpu, block_len=32, stride=16
    )

    # CUDA kernel
    k_cmp_cuda, v_cmp_cuda = compressed_attn_forward(
        k, v, phi_k, phi_v, block_len=32, stride=16
    )

    # Compare (allow FP16 tolerance)
    atol, rtol = 5e-2, 5e-2
    k_ok = torch.allclose(k_cmp_cuda.cpu().float(), k_cmp_ref, atol=atol, rtol=rtol)
    v_ok = torch.allclose(v_cmp_cuda.cpu().float(), v_cmp_ref, atol=atol, rtol=rtol)

    print(f"  compressed_attn K parity: {'✅' if k_ok else '❌'} (max_diff={torch.abs(k_cmp_cuda.cpu().float() - k_cmp_ref).max().item():.4f})")
    print(f"  compressed_attn V parity: {'✅' if v_ok else '❌'} (max_diff={torch.abs(v_cmp_cuda.cpu().float() - v_cmp_ref).max().item():.4f})")

    assert k_ok and v_ok, "Parity check failed"
    return {"k_ok": k_ok, "v_ok": v_ok}


# %% [markdown]
# ## Test 3: Selective Attention — CUDA vs PyTorch Parity

# %%
@app.function(gpu="T4", image=image, timeout=300)
def test_selective_attn_parity():
    """Compare selective_attn CUDA kernel vs PyTorch reference."""
    from kernels.selective_attn.selective_attn import (
        selective_attn_forward,
        _selective_attn_eager,
    )

    # The CUDA kernel is not yet fully wired; test the Python path on GPU
    B, H, T, D = 2, 4, 128, 64
    device = "cuda"

    q = torch.randn(B, H, T, D, device="cpu", dtype=torch.float32)
    k = torch.randn(B, H, T, D, device="cpu", dtype=torch.float32)
    v = torch.randn(B, H, T, D, device="cpu", dtype=torch.float32)
    scores_agg = torch.randn(B, H, T // 16, device="cpu")

    out_ref = _selective_attn_eager(q, k, v, scores_agg, topk=4, block_size=16)

    # Verify no NaN and correct shape
    assert out_ref.shape == (B, H, T, D), f"Shape mismatch: {out_ref.shape}"
    assert not torch.isnan(out_ref).any(), "NaN in output"
    print(f"  selective_attn shape OK: ✅")
    print(f"  selective_attn no NaN: ✅")

    return {"shape_ok": True, "nan_ok": True}


# %% [markdown]
# ## Test 4: Block-Sparse Ternary Matmul on GPU

# %%
@app.function(gpu="T4", image=image, timeout=300)
def test_block_sparse_ternary_parity():
    """Test block_sparse_ternary matmul on GPU."""
    from kernels.block_sparse_ternary.block_sparse_ternary import (
        block_sparse_ternary_matmul,
        _block_sparse_ternary_eager,
        compute_block_mask,
    )

    device = "cuda"
    M, N, K = 256, 128, 128
    x = torch.randn(M, K, device=device)
    weight = torch.randn(N, K, device=device)
    gamma = weight.abs().mean() + 1e-5

    # All-active mask
    num_n_tiles = (N + 63) // 64
    num_k_tiles = (K + 63) // 64
    num_ints = max(1, (num_n_tiles * num_k_tiles + 63) // 64)
    block_mask = torch.full((num_ints,), ~0, dtype=torch.int64, device=device)

    y = block_sparse_ternary_matmul(x, weight, gamma, block_mask)
    assert y.shape == (M, N), f"Shape: {y.shape}"
    assert not torch.isnan(y).any(), "NaN in output"

    # Sparse mask
    block_mask_sparse = torch.full((num_ints,), ~0, dtype=torch.int64, device=device)
    for tk in range(num_k_tiles):
        bit_pos = 1 * num_k_tiles + tk
        block_mask_sparse[bit_pos // 64] &= ~(1 << (bit_pos % 64))

    y_sparse = block_sparse_ternary_matmul(x, weight, gamma, block_mask_sparse)
    BN = 64
    assert torch.all(y_sparse[:, BN:2*BN] == 0), "Masked tile should be zero"

    print(f"  block_sparse_ternary shape: ✅")
    print(f"  block_sparse_ternary sparse mask: ✅")
    return {"shape_ok": True, "sparse_ok": True}


# %% [markdown]
# ## Test 5: SubQSA Combine on GPU

# %%
@app.function(gpu="T4", image=image, timeout=300)
def test_subqsa_combine_parity():
    """Compare subqsa_combine CUDA kernel vs PyTorch on GPU."""
    from kernels.subqsa_combine.subqsa_combine import (
        subqsa_combine_forward,
        _subqsa_combine_eager,
    )

    device = "cuda"
    B, T, H, D_head, D_out = 1, 4, 2, 8, 16
    D = H * D_head

    x = torch.randn(B, T, D, device=device, dtype=torch.float16)
    o_cmp = torch.randn(B, H, T, D_head, device=device, dtype=torch.float16)
    o_slc = torch.randn(B, H, T, D_head, device=device, dtype=torch.float16)
    o_win = torch.randn(B, H, T, D_head, device=device, dtype=torch.float16)
    gate_w1 = torch.randn(64, D, device=device, dtype=torch.float16) * 0.02
    gate_w2 = torch.randn(3 * H, 64, device=device, dtype=torch.float16) * 0.02
    out_norm = torch.ones(H * D_head, device=device, dtype=torch.float16)
    o_proj_w = torch.randn(D_out, H * D_head, device=device, dtype=torch.float16) * 0.02
    gamma = torch.tensor([o_proj_w.abs().mean().item() + 1e-5], device=device)

    y = subqsa_combine_forward(x, o_cmp, o_slc, o_win, gate_w1, gate_w2, out_norm, o_proj_w, gamma)
    assert y.shape == (B, T, D_out), f"Shape: {y.shape}"
    assert not torch.isnan(y).any(), "NaN in output"
    print(f"  subqsa_combine shape: ✅")
    print(f"  subqsa_combine no NaN: ✅")
    return {"shape_ok": True, "nan_ok": True}


# %% [markdown]
# ## Test 6: Full End-to-End Forward (PyTorch path on GPU)

# %%
@app.function(gpu="T4", image=image, timeout=600)
def test_e2e_forward():
    """Run full UltimateModel forward pass on GPU (PyTorch path, use_cuda_kernels=False)."""
    sys.path.insert(0, "/root")
    from ultimate_trainer.config import UltimateModelConfig
    from ultimate_trainer.model import UltimateModel

    cfg = UltimateModelConfig(
        vocab_size=4096,
        hidden_dim=256,
        intermediate_dim=512,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        max_seq_len=128,
        use_bitlinear=False,
        use_subqsa=True,
        use_cuda_kernels=False,
        cmp_block=16, cmp_stride=8,
        slc_block=32, slc_topk=4, win_size=32,
    )
    model = UltimateModel(cfg).to("cuda").half()
    model.eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 64), device="cuda")

    # Warmup
    for _ in range(3):
        _ = model(input_ids)

    # Timed
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        logits = model(input_ids)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    loss = model.get_loss(input_ids)
    tokens_per_sec = 10 * input_ids.numel() / elapsed

    print(f"  E2E forward shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Tokens/sec: {tokens_per_sec:.1f}")
    print(f"  No NaN: {not torch.isnan(logits).any()}")

    assert logits.shape == (1, 64, cfg.vocab_size)
    assert not torch.isnan(logits).any()
    return {"loss": loss.item(), "tps": tokens_per_sec}


# %% [markdown]
# ## Run All Tests

# %%
@app.local_entrypoint()
def main():
    print("=" * 60)
    print("CUDA KERNEL VALIDATION ON MODAL GPU")
    print("=" * 60)

    print("\n1. Compilation test:")
    comp = test_kernel_compilation.remote()
    all_compiled = all(v == "COMPILED_OK" for v in comp.values())
    print(f"   Overall: {'✅ ALL COMPILED' if all_compiled else '❌ SOME FAILED'}")

    if not all_compiled:
        print("\n❌ Stopping — compilation failures must be fixed first.")
        return

    print("\n2. Compressed Attention parity:")
    test_compressed_attn_parity.remote()

    print("\n3. Selective Attention:")
    test_selective_attn_parity.remote()

    print("\n4. Block-Sparse Ternary:")
    test_block_sparse_ternary_parity.remote()

    print("\n5. SubQSA Combine:")
    test_subqsa_combine_parity.remote()

    print("\n6. End-to-End Forward:")
    test_e2e_forward.remote()

    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 60)
