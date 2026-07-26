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

    # Scale inputs small — ternary weights amplify magnitude
    # (sqrt(in_features * 0.5) ≈ 20× amplification)
    x = torch.randn(32, 768, dtype=torch.float16, device="cuda") * 0.01
    target = torch.randn(32, 768, dtype=torch.float16, device="cuda") * 0.01

    losses = []
    for step in range(100):
        mlp.zero_grad()
        out = mlp(x)
        loss = F.mse_loss(out.float(), target.float())  # compute in FP32
        if torch.isfinite(loss):
            loss.backward()
            losses.append(loss.item())
        else:
            losses.append(float('inf'))
        if step % 20 == 0:
            print(f"  Step {step:3d}: loss={losses[-1]:.6f}")

    finite_losses = [l for l in losses if l < float('inf')]
    if len(finite_losses) > 2:
        improved = sum(1 for i in range(1, len(finite_losses)) if finite_losses[i] < finite_losses[i-1])
        print(f"  Steps with improvement: {improved}/{len(finite_losses)-1}")
        print(f"  Final loss: {finite_losses[-1]:.6f} (started: {finite_losses[0]:.6f})")
        converged = finite_losses[-1] < finite_losses[0] * 0.5
    else:
        print(f"  Only {len(finite_losses)} finite losses")
        converged = False
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
        # LayerNorm outputs float32 — cast to half for packed ternary
        h = self.ln1(x).half()
        qkv = self.qkv(h)
        q, k, v = qkv.split(self.d_model, dim=2)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(attn)
        # Cast LayerNorm output to half
        h2 = self.ln2(x).half()
        x = x + self.ff2(F.gelu(self.ff1(h2), approximate="tanh"))
        return x


def test_stage2():
    """Stage 2: Single transformer block (d_model=128, seq_len=32)"""
    print("\n" + "=" * 60)
    print("  Stage 2: Transformer Block (d=128, heads=4, seq=32)")
    print("=" * 60)

    block = TransformerBlockTernary(d_model=128, nhead=4).cuda()
    x = torch.randn(8, 32, 128, dtype=torch.float16, device="cuda") * 0.1
    target = torch.randn(8, 32, 128, dtype=torch.float16, device="cuda") * 0.1

    losses = []
    for step in range(100):
        block.zero_grad()
        out = block(x)
        loss = F.mse_loss(out.float(), target.float())
        if torch.isfinite(loss):
            loss.backward()
            losses.append(loss.item())
        else:
            losses.append(float('inf'))
        if step % 20 == 0:
            print(f"  Step {step:3d}: loss={losses[-1]:.6f}")

    finite_losses = [l for l in losses if l < float('inf')]
    if len(finite_losses) > 2:
        improved = sum(1 for i in range(1, len(finite_losses)) if finite_losses[i] < finite_losses[i-1])
        print(f"  Steps with improvement: {improved}/{len(finite_losses)-1}")
        print(f"  Final loss: {finite_losses[-1]:.6f} (started: {finite_losses[0]:.6f})")
        converged = finite_losses[-1] < finite_losses[0] * 0.5
    else:
        print(f"  Only {len(finite_losses)} finite losses")
        converged = False
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
        x = self.ln_f(x).half()
        return self.head(x)


def test_stage3():
    """Stage 3: Mini GPT (6 layers, d_model=128, seq=64, vocab=256)"""
    print("\n" + "=" * 60)
    print("  Stage 3: MiniGPT (6 layers, d=128, seq=64)")
    print("=" * 60)

    model = MiniGPT(d_model=128, nhead=4, n_layers=6, vocab_size=256).cuda()
    x = torch.randint(0, 256, (4, 64), device="cuda")
    target = torch.randint(0, 256, (4, 64), device="cuda")

    losses = []
    for step in range(50):
        model.zero_grad()
        logits = model(x).float()  # cross_entropy needs float
        loss = F.cross_entropy(logits.view(-1, 256), target.view(-1))
        if torch.isfinite(loss):
            loss.backward()
            losses.append(loss.item())
        else:
            losses.append(float('inf'))
        if step % 10 == 0:
            print(f"  Step {step:3d}: loss={losses[-1]:.4f}")

    finite_losses = [l for l in losses if l < float('inf')]
    if len(finite_losses) > 2:
        improved = sum(1 for i in range(1, len(finite_losses)) if finite_losses[i] < finite_losses[i-1])
        print(f"  Steps with improvement: {improved}/{len(finite_losses)-1}")
        print(f"  Final loss: {finite_losses[-1]:.4f} (started: {finite_losses[0]:.4f})")
        converged = finite_losses[-1] < finite_losses[0] * 0.8
    else:
        print(f"  Only {len(finite_losses)} finite losses")
        converged = False
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
        import traceback; traceback.print_exc()
        results["Stage 2"] = False

    try:
        results["Stage 3 (MiniGPT 6-layer)"] = test_stage3()
    except Exception as e:
        print(f"  Stage 3 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["Stage 3"] = False

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name}: {'✅' if passed else '❌'}")
    print("=" * 60)
