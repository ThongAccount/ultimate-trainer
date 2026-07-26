"""Larger model convergence tests for packed ternary discrete optimizer.

Stage 1: 768 → 2048 → 768 MLP
Stage 2: Single transformer block (attention + FFN)
Stage 3: Small GPT (6 layers, 128-dim)

Usage:
    python test_convergence_large.py
"""

import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from kernels.packed_ternary.packed_linear import PackedTernaryLinear


def test_stage1():
    """Stage 1: 768 → 2048 → 768 MLP"""
    print("=" * 60)
    print("  Stage 1: 768 → 2048 → 768 MLP")
    print("=" * 60)

    mlp = nn.Sequential(
        PackedTernaryLinear(768, 2048),
        nn.GELU(),
        PackedTernaryLinear(2048, 768)
    ).cuda()

    x = torch.randn(32, 768, dtype=torch.float16, device="cuda")
    target = torch.randn(32, 768, dtype=torch.float16, device="cuda")

    losses = []
    for step in range(100):
        mlp.zero_grad()
        out = mlp(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        losses.append(loss.item())
        if step % 20 == 0:
            print(f"  Step {step:3d}: loss={loss.item():.4f}")

    # Check monotonic decrease (allow some noise)
    improved = sum(1 for i in range(1, len(losses)) if losses[i] < losses[i-1])
    total = len(losses) - 1
    print(f"  Steps with improvement: {improved}/{total}")
    print(f"  Final loss: {losses[-1]:.4f} (started: {losses[0]:.4f})")
    converged = losses[-1] < losses[0] * 0.5  # at least 50% reduction
    print(f"  Converged: {'YES ✅' if converged else 'NO ❌'}")
    return converged


class TransformerBlockTernary(nn.Module):
    """Stage 2: Single transformer block (attention + FFN)"""
    def __init__(self, d_model=128, nhead=4, threshold=8):
        super().__init__()
        self.qkv = PackedTernaryLinear(d_model, d_model * 3, threshold=threshold)
        self.proj = PackedTernaryLinear(d_model, d_model, threshold=threshold)
        self.ff1 = PackedTernaryLinear(d_model, d_model * 4, threshold=threshold)
        self.ff2 = PackedTernaryLinear(d_model * 4, d_model, threshold=threshold)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.nhead = nhead
        self.d_model = d_model

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.split(self.d_model, dim=2)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(attn)
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x)), approximate="tanh"))
        return x


def test_stage2():
    """Stage 2: Single transformer block (d_model=128, seq_len=32)"""
    print("\n" + "=" * 60)
    print("  Stage 2: Transformer Block (d=128, heads=4, seq=32)")
    print("=" * 60)

    block = TransformerBlockTernary(d_model=128, nhead=4).cuda()
    x = torch.randn(8, 32, 128, dtype=torch.float16, device="cuda")
    target = torch.randn(8, 32, 128, dtype=torch.float16, device="cuda")

    losses = []
    for step in range(100):
        block.zero_grad()
        out = block(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        losses.append(loss.item())
        if step % 20 == 0:
            print(f"  Step {step:3d}: loss={loss.item():.4f}")

    improved = sum(1 for i in range(1, len(losses)) if losses[i] < losses[i-1])
    total = len(losses) - 1
    print(f"  Steps with improvement: {improved}/{total}")
    print(f"  Final loss: {losses[-1]:.4f} (started: {losses[0]:.4f})")
    converged = losses[-1] < losses[0] * 0.5
    print(f"  Converged: {'YES ✅' if converged else 'NO ❌'}")
    return converged


class MiniGPT(nn.Module):
    """Stage 3: Small GPT (6 layers, d_model=128)"""
    def __init__(self, d_model=128, nhead=4, n_layers=6, vocab_size=256, threshold=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlockTernary(d_model, nhead, threshold)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = PackedTernaryLinear(d_model, vocab_size, threshold=threshold)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def test_stage3():
    """Stage 3: Mini GPT (6 layers, d=128, seq=64, vocab=256)"""
    print("\n" + "=" * 60)
    print("  Stage 3: MiniGPT (6 layers, d=128, seq=64)")
    print("=" * 60)

    model = MiniGPT(d_model=128, nhead=4, n_layers=6, vocab_size=256).cuda()
    x = torch.randint(0, 256, (4, 64), device="cuda")
    target = torch.randint(0, 256, (4, 64), device="cuda")

    losses = []
    for step in range(50):
        model.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, 256), target.view(-1))
        loss.backward()
        losses.append(loss.item())
        if step % 10 == 0:
            print(f"  Step {step:3d}: loss={loss.item():.4f}")

    improved = sum(1 for i in range(1, len(losses)) if losses[i] < losses[i-1])
    total = len(losses) - 1
    print(f"  Steps with improvement: {improved}/{total}")
    print(f"  Final loss: {losses[-1]:.4f} (started: {losses[0]:.4f})")
    converged = losses[-1] < losses[0] * 0.8  # 20% reduction for harder task
    print(f"  Converged: {'YES ✅' if converged else 'NO ❌'}")
    return converged


if __name__ == "__main__":
    results = {}

    try:
        results["Stage 1 (MLP 768→2048→768)"] = test_stage1()
    except Exception as e:
        print(f"  Stage 1 FAILED: {e}")
        results["Stage 1"] = False

    try:
        results["Stage 2 (Transformer Block)"] = test_stage2()
    except Exception as e:
        print(f"  Stage 2 FAILED: {e}")
        results["Stage 2"] = False

    try:
        results["Stage 3 (MiniGPT 6-layer)"] = test_stage3()
    except Exception as e:
        print(f"  Stage 3 FAILED: {e}")
        results["Stage 3"] = False

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name}: {'✅' if passed else '❌'}")
    print("=" * 60)
