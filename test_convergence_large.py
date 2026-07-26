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


class ScaledTernaryLinear(nn.Module):
    """PackedTernaryLinear with output scaling to prevent magnitude explosion.

    Ternary weights amplify output by sqrt(in_features * 0.5).
    Scaling by 1/sqrt(in_features) keeps output magnitude stable.

    Handles 2D [B, K] and 3D [B, T, K] inputs by reshaping internally.
    """
    def __init__(self, in_features, out_features, threshold=8):
        super().__init__()
        self.linear = PackedTernaryLinear(in_features, out_features, threshold=threshold)
        self.scale = 1.0 / (in_features ** 0.5)

    def forward(self, x):
        orig_shape = x.shape
        if x.dim() == 3:
            B, T, K = x.shape
            x = x.reshape(B * T, K)
        y = self.linear(x) * self.scale
        if len(orig_shape) == 3:
            y = y.reshape(orig_shape[0], orig_shape[1], -1)
        return y


def train_with_accumulation(model, x, target, steps=200, accum_steps=4, threshold=8, print_every=20):
    """Train with gradient accumulation to reduce noise.

    Accumulates gradients over accum_steps forward passes before
    triggering one counter update. Reduces gradient noise by sqrt(accum_steps).
    """
    # Set threshold on all PackedTernaryLinear layers
    for m in model.modules():
        if isinstance(m, PackedTernaryLinear):
            m.threshold = threshold

    losses = []
    for step in range(steps):
        total_loss = 0.0
        for accum in range(accum_steps):
            # Use different random inputs each accumulation step
            x_acc = x + torch.randn_like(x) * 0.1  # small noise for diversity
            out = model(x_acc)
            loss = F.mse_loss(out.float(), target.float())
            if torch.isfinite(loss):
                loss.backward()  # accumulates gradients in counter
                total_loss += loss.item()

        avg_loss = total_loss / accum_steps
        losses.append(avg_loss)
        if step % print_every == 0:
            print(f"  Step {step:3d}: loss={avg_loss:.6f}")

    finite_losses = [l for l in losses if l < float('inf')]
    if len(finite_losses) > 2:
        improved = sum(1 for i in range(1, len(finite_losses)) if finite_losses[i] < finite_losses[i-1])
        print(f"  Steps with improvement: {improved}/{len(finite_losses)-1}")
        print(f"  Final loss: {finite_losses[-1]:.6f} (started: {finite_losses[0]:.6f})")
        converged = finite_losses[-1] < finite_losses[0] * 0.5
    else:
        print(f"  Only {len(finite_losses)} finite losses")
        converged = False
    return converged


def test_stage1():
    """Stage 1: 768 → 2048 → 768 MLP with output scaling"""
    print("=" * 60)
    print("  Stage 1: 768 → 2048 → 768 MLP (scaled, accum=4)")
    print("=" * 60)

    mlp = nn.Sequential(
        ScaledTernaryLinear(768, 2048),
        nn.GELU(),
        ScaledTernaryLinear(2048, 768)
    ).cuda()

    x = torch.randn(32, 768, dtype=torch.float16, device="cuda")
    target = torch.randn(32, 768, dtype=torch.float16, device="cuda")

    converged = train_with_accumulation(mlp, x, target, steps=200, accum_steps=4, threshold=32)
    print(f"  Converged: {'YES ✅' if converged else 'NO ❌'}")
    return converged


class TransformerBlockTernary(nn.Module):
    """Stage 2: Single transformer block (attention + FFN)"""
    def __init__(self, d_model=128, nhead=4, threshold=8):
        super().__init__()
        self.qkv = ScaledTernaryLinear(d_model, d_model * 3, threshold=threshold)
        self.proj = ScaledTernaryLinear(d_model, d_model, threshold=threshold)
        self.ff1 = ScaledTernaryLinear(d_model, d_model * 4, threshold=threshold)
        self.ff2 = ScaledTernaryLinear(d_model * 4, d_model, threshold=threshold)
        self.ln1 = nn.LayerNorm(d_model).half()
        self.ln2 = nn.LayerNorm(d_model).half()
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
        h2 = self.ln2(x)
        x = x + self.ff2(F.gelu(self.ff1(h2), approximate="tanh"))
        return x


def test_stage2():
    """Stage 2: Single transformer block (d_model=128, seq_len=32)"""
    print("\n" + "=" * 60)
    print("  Stage 2: Transformer Block (d=128, heads=4, seq=32, accum=4)")
    print("=" * 60)

    block = TransformerBlockTernary(d_model=128, nhead=4).cuda()
    x = torch.randn(8, 32, 128, dtype=torch.float16, device="cuda")
    target = torch.randn(8, 32, 128, dtype=torch.float16, device="cuda")

    converged = train_with_accumulation(block, x, target, steps=200, accum_steps=4, threshold=16)
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
        self.ln_f = nn.LayerNorm(d_model).half()
        self.head = ScaledTernaryLinear(d_model, vocab_size, threshold=threshold)

    def forward(self, x):
        x = self.embed(x).half()
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def train_gpt_with_accumulation(model, steps=100, accum_steps=4, threshold=8, print_every=10):
    """Train GPT with gradient accumulation."""
    for m in model.modules():
        if isinstance(m, PackedTernaryLinear):
            m.threshold = threshold

    losses = []
    for step in range(steps):
        total_loss = 0.0
        for accum in range(accum_steps):
            x = torch.randint(0, 256, (4, 64), device="cuda")
            target = torch.randint(0, 256, (4, 64), device="cuda")
            logits = model(x).float()
            loss = F.cross_entropy(logits.view(-1, 256), target.view(-1))
            if torch.isfinite(loss):
                loss.backward()
                total_loss += loss.item()

        avg_loss = total_loss / accum_steps
        losses.append(avg_loss)
        if step % print_every == 0:
            print(f"  Step {step:3d}: loss={avg_loss:.4f}")

    finite_losses = [l for l in losses if l < float('inf')]
    if len(finite_losses) > 2:
        improved = sum(1 for i in range(1, len(finite_losses)) if finite_losses[i] < finite_losses[i-1])
        print(f"  Steps with improvement: {improved}/{len(finite_losses)-1}")
        print(f"  Final loss: {finite_losses[-1]:.4f} (started: {finite_losses[0]:.4f})")
        converged = finite_losses[-1] < finite_losses[0] * 0.8
    else:
        print(f"  Only {len(finite_losses)} finite losses")
        converged = False
    return converged


def test_stage3():
    """Stage 3: Mini GPT (6 layers, d_model=128, seq=64, vocab=256)"""
    print("\n" + "=" * 60)
    print("  Stage 3: MiniGPT (6 layers, d=128, seq=64, accum=4)")
    print("=" * 60)

    model = MiniGPT(d_model=128, nhead=4, n_layers=6, vocab_size=256).cuda()

    converged = train_gpt_with_accumulation(model, steps=100, accum_steps=4, threshold=16)
    print(f"  Converged: {'YES ✅' if converged else 'NO ❌'}")
    return converged


if __name__ == "__main__":
    results = {}

    try:
        results["Stage 1 (MLP 768→2048→768)"] = test_stage1()
    except Exception as e:
        print(f"  Stage 1 FAILED: {e}")
        import traceback; traceback.print_exc()
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
