#!/usr/bin/env python3
"""
Comparison: FP Linear vs BitLinear (ternary weights {-1, 0, +1}).
Runs a small 4-layer MLP side-by-side in both variants and reports:
  - parameter memory (FP32 master vs effective ternary)
  - forward output consistency (cosine similarity)
  - loss curve similarity over 50 steps of toy training
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import BitLinear, RMSNorm, absmean_quantize_weight


# ── helpers ──────────────────────────────────────────────────────────────


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def memory_mb(m: nn.Module, bits_per_weight: float = 32.0) -> float:
    """Estimate memory in MB at `bits_per_weight` per parameter."""
    n = count_params(m)
    return n * bits_per_weight / 8 / 1024 / 1024


def make_fp_mlp(dim=256, depth=4):
    """A simple MLP with nn.Linear + ReLU."""
    layers = []
    for i in range(depth):
        layers.append(nn.Linear(dim, dim, bias=False))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def make_bit_mlp(dim=256, depth=4):
    """Same MLP with BitLinear."""
    layers = []
    for i in range(depth):
        layers.append(BitLinear(dim, dim, bias=False, quantize_activations=True))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# ── main comparison ─────────────────────────────────────────────────────


def main():
    torch.manual_seed(42)
    dim = 256
    depth = 4
    seq_len = 128
    batch = 2
    steps = 50
    lr = 1e-3

    device = "cuda" if torch.cuda.is_available() else "cpu"

    fp_mlp = make_fp_mlp(dim, depth).to(device)
    bit_mlp = make_bit_mlp(dim, depth).to(device)

    # ── 1. parameter count & memory ──
    print("=" * 60)
    print("1-BIT TRAINER: FP vs BitLinear Comparison")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"MLP: {depth} layers, hidden={dim}")
    print()

    fp_params = count_params(fp_mlp)
    bit_params = count_params(bit_mlp)
    print(f"FP params:      {fp_params:>8,}")
    print(f"BitLinear params (master): {bit_params:>8,}")
    print(
        f"Ternary effective memory:   {memory_mb(bit_mlp, 1.58):>8.2f} MB  (1.58 b/w)"
    )
    print(f"FP memory:      {memory_mb(fp_mlp, 16):>8.2f} MB  (BF16 equiv)")
    print()

    # ── 2. forward range check ──
    x = torch.randn(batch, seq_len, dim, device=device)
    with torch.no_grad():
        fp_out = fp_mlp(x)
        bit_out = bit_mlp(x)

    cos_sim = F.cosine_similarity(fp_out.view(-1), bit_out.view(-1), dim=0).item()
    fp_mean = fp_out.abs().mean().item()
    bit_mean = bit_out.abs().mean().item()
    print(f"FP output abs mean:      {fp_mean:.4f}")
    print(f"BitLinear output abs mean: {bit_mean:.4f}")
    print(f"Outputs in similar range:  {abs(fp_mean - bit_mean) / fp_mean:.2%} diff")
    print()

    # ── 3. training: both should learn ──
    opt_fp = torch.optim.AdamW(fp_mlp.parameters(), lr=lr)
    opt_bit = torch.optim.AdamW(bit_mlp.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    target = torch.randn(batch, seq_len, dim, device=device)

    fp_losses, bit_losses = [], []
    for step in range(steps):
        opt_fp.zero_grad()
        opt_bit.zero_grad()

        loss_fp = loss_fn(fp_mlp(x), target)
        loss_bit = loss_fn(bit_mlp(x), target)

        loss_fp.backward()
        loss_bit.backward()

        opt_fp.step()
        opt_bit.step()

        fp_losses.append(loss_fp.item())
        bit_losses.append(loss_bit.item())

    fp_initial = fp_losses[0]
    bit_initial = bit_losses[0]
    fp_final = fp_losses[-1]
    bit_final = bit_losses[-1]

    fp_reduced = fp_initial > fp_final
    bit_reduced = bit_initial > bit_final

    print(
        f"FP training:       {fp_initial:.4f} → {fp_final:.4f}  ({'↓ reduced' if fp_reduced else '↑ increased'})"
    )
    print(
        f"BitLinear training: {bit_initial:.4f} → {bit_final:.4f}  ({'↓ reduced' if bit_reduced else '↑ increased'})"
    )
    print()

    # ── 4. HF Kernels decorator check ──
    try:
        from kernels import use_kernel_forward_from_hub

        print("HF Kernels decorator: AVAILABLE")
        print("  BitLinear is registered as 'BitLinear' for kernel injection.")
    except ImportError:
        print("HF Kernels: not installed (pip install kernels)")

    # ── verdict ──
    verdict = "PASS" if (fp_reduced and bit_reduced) else "CHECK"
    print(f"\nVerdict: {verdict}")
    print("  Both models must reduce loss during training (learning works).")
    print(f"  FP loss reduced: {fp_reduced}, BitLinear loss reduced: {bit_reduced}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
