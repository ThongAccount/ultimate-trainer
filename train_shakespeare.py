"""Character-level LM on Shakespeare — real data convergence test.

Trains a small GPT on Shakespeare text using the discrete ternary
optimizer. Compares loss curve and throughput vs AdamW baseline.

Usage:
    python train_shakespeare.py              # full run
    python train_shakespeare.py --quick      # 100 steps only
    python train_shakespeare.py --baseline   # AdamW baseline only

Metrics:
    - Loss curve (should decrease monotonically)
    - Tokens/second (throughput)
    - Peak memory (should be ~5× less than AdamW)
    - Final perplexity vs AdamW baseline
"""

import sys, os, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from kernels.packed_ternary.packed_linear import PackedTernaryLinear

# ═══════════════════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════════════════

VOCAB_SIZE = 256  # byte-level
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 6
BLOCK_SIZE = 128   # sequence length
BATCH_SIZE = 16
ACCUM_STEPS = 4    # gradient accumulation for discrete optimizer
THRESHOLD = 32     # counter threshold for discrete optimizer
LR_ADAMW = 3e-4    # learning rate for AdamW baseline
STEPS = 500
PRINT_EVERY = 10
SAVE_EVERY = 100


# ═══════════════════════════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════════════════════════

def get_shakespeare():
    """Download Shakespeare text or use cached version."""
    data_path = os.path.join(os.path.dirname(__file__), "shakespeare.txt")
    if not os.path.exists(data_path):
        print("Downloading Shakespeare...")
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)
        print(f"  Saved to {data_path}")

    with open(data_path, "r") as f:
        text = f.read()
    print(f"  Shakespeare: {len(text):,} characters")
    return text


def make_dataset(text, block_size, batch_size, device="cuda"):
    """Create byte-level dataset from text."""
    data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long, device=device)
    n_batches = len(data) // (block_size * batch_size)
    data = data[:n_batches * block_size * batch_size]
    return data


def get_batch(data, block_size, batch_size, step):
    """Get a batch of (x, y) pairs for next-token prediction."""
    idx = (step * batch_size * block_size) % (len(data) - block_size * batch_size)
    chunk = data[idx:idx + batch_size * block_size + 1]
    x = chunk[:-1].view(batch_size, block_size)
    y = chunk[1:].view(batch_size, block_size)
    return x, y


# ═══════════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════════

class ScaledTernaryLinear(nn.Module):
    """PackedTernaryLinear with output scaling."""
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


class TernaryTransformerBlock(nn.Module):
    """Transformer block with ternary linear layers."""
    def __init__(self, d_model, nhead, threshold=8):
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


class MiniGPT(nn.Module):
    """Small GPT for character-level LM."""
    def __init__(self, vocab_size=256, d_model=128, nhead=4, n_layers=6,
                 block_size=128, threshold=8, use_ternary=True):
        super().__init__()
        self.block_size = block_size
        self.use_ternary = use_ternary
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(block_size, d_model)

        if use_ternary:
            self.blocks = nn.ModuleList([
                TernaryTransformerBlock(d_model, nhead, threshold)
                for _ in range(n_layers)
            ])
            self.ln_f = nn.LayerNorm(d_model).half()
            self.head = ScaledTernaryLinear(d_model, vocab_size, threshold=threshold)
        else:
            # AdamW baseline uses standard nn.Linear
            self.blocks = nn.ModuleList([
                StandardTransformerBlock(d_model, nhead)
                for _ in range(n_layers)
            ])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.embed(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embed(pos)

        if self.use_ternary:
            x = (tok_emb + pos_emb).half()
        else:
            x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.head(x)


class StandardTransformerBlock(nn.Module):
    """Standard transformer block for AdamW baseline."""
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_discrete(model, data, steps, accum_steps, threshold, print_every):
    """Train with discrete ternary optimizer (counter-based)."""
    # Set threshold on all ternary layers
    for m in model.modules():
        if isinstance(m, PackedTernaryLinear):
            m.threshold = threshold

    losses = []
    tokens_per_sec = []
    start_time = time.time()

    for step in range(steps):
        total_loss = 0.0
        for accum in range(accum_steps):
            x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE, step * accum_steps + accum)
            logits = model(x).float()
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            total_loss += loss.item()

        avg_loss = total_loss / accum_steps
        losses.append(avg_loss)

        # Tokens processed this step
        tokens = BATCH_SIZE * BLOCK_SIZE * accum_steps
        elapsed = time.time() - start_time
        tps = (step + 1) * tokens / elapsed if elapsed > 0 else 0
        tokens_per_sec.append(tps)

        if step % print_every == 0:
            perplexity = 2 ** avg_loss if avg_loss < 20 else float('inf')
            print(f"  Step {step:4d}: loss={avg_loss:.4f}  ppl={perplexity:.1f}  "
                  f"tok/s={tps:.0f}")

    return losses, tokens_per_sec


def train_adamw(model, data, steps, lr, print_every):
    """Train with AdamW baseline."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    tokens_per_sec = []
    start_time = time.time()

    for step in range(steps):
        x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE, step)
        logits = model(x).float()
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

        tokens = BATCH_SIZE * BLOCK_SIZE
        elapsed = time.time() - start_time
        tps = (step + 1) * tokens / elapsed if elapsed > 0 else 0
        tokens_per_sec.append(tps)

        if step % print_every == 0:
            perplexity = 2 ** loss.item() if loss.item() < 20 else float('inf')
            print(f"  Step {step:4d}: loss={loss.item():.4f}  ppl={perplexity:.1f}  "
                  f"tok/s={tps:.0f}")

    return losses, tokens_per_sec


def peak_memory_mb():
    """Get peak GPU memory in MB."""
    return torch.cuda.max_memory_allocated() / 1024 / 1024


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="100 steps only")
    parser.add_argument("--baseline", action="store_true", help="AdamW baseline only")
    parser.add_argument("--discrete", action="store_true", help="Discrete optimizer only")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--accum", type=int, default=ACCUM_STEPS)
    parser.add_argument("--threshold", type=int, default=THRESHOLD)
    args = parser.parse_args()

    steps = 100 if args.quick else args.steps

    print("=" * 60)
    print("  Shakespeare Character-Level LM")
    print("=" * 60)
    print(f"  Config: d={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}")
    print(f"  Block={BLOCK_SIZE}, batch={BATCH_SIZE}, vocab={VOCAB_SIZE}")
    print()

    # Load data
    text = get_shakespeare()
    data_discrete = make_dataset(text, BLOCK_SIZE, BATCH_SIZE)
    data_adamw = make_dataset(text, BLOCK_SIZE, BATCH_SIZE)

    results = {}

    # ── Discrete optimizer ──────────────────────────────────────────
    if not args.baseline:
        print(f"\n{'─' * 60}")
        print(f"  Discrete Ternary Optimizer (threshold={args.threshold}, accum={args.accum})")
        print(f"{'─' * 60}")

        torch.cuda.reset_peak_memory_stats()
        model_d = MiniGPT(
            d_model=D_MODEL, nhead=N_HEADS, n_layers=N_LAYERS,
            block_size=BLOCK_SIZE, threshold=args.threshold, use_ternary=True
        ).cuda()

        losses_d, tps_d = train_discrete(
            model_d, data_discrete, steps, args.accum, args.threshold, PRINT_EVERY
        )
        mem_d = peak_memory_mb()
        results["discrete"] = {
            "losses": losses_d, "tps": tps_d, "memory_mb": mem_d,
            "final_loss": losses_d[-1] if losses_d else float('inf'),
        }
        print(f"\n  Final loss: {losses_d[-1]:.4f}")
        print(f"  Peak memory: {mem_d:.0f} MB")
        print(f"  Avg tok/s: {sum(tps_d)/len(tps_d):.0f}")

        del model_d
        torch.cuda.empty_cache()

    # ── AdamW baseline ──────────────────────────────────────────────
    if not args.discrete:
        print(f"\n{'─' * 60}")
        print(f"  AdamW Baseline (lr={LR_ADAMW})")
        print(f"{'─' * 60}")

        torch.cuda.reset_peak_memory_stats()
        model_a = MiniGPT(
            d_model=D_MODEL, nhead=N_HEADS, n_layers=N_LAYERS,
            block_size=BLOCK_SIZE, use_ternary=False
        ).cuda()

        losses_a, tps_a = train_adamw(
            model_a, data_adamw, steps, LR_ADAMW, PRINT_EVERY
        )
        mem_a = peak_memory_mb()
        results["adamw"] = {
            "losses": losses_a, "tps": tps_a, "memory_mb": mem_a,
            "final_loss": losses_a[-1] if losses_a else float('inf'),
        }
        print(f"\n  Final loss: {losses_a[-1]:.4f}")
        print(f"  Peak memory: {mem_a:.0f} MB")
        print(f"  Avg tok/s: {sum(tps_a)/len(tps_a):.0f}")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  Summary")
    print(f"{'═' * 60}")

    if "discrete" in results:
        d = results["discrete"]
        print(f"  Discrete:  loss={d['final_loss']:.4f}  "
              f"mem={d['memory_mb']:.0f}MB  "
              f"tok/s={sum(d['tps'])/len(d['tps']):.0f}")

    if "adamw" in results:
        a = results["adamw"]
        print(f"  AdamW:     loss={a['final_loss']:.4f}  "
              f"mem={a['memory_mb']:.0f}MB  "
              f"tok/s={sum(a['tps'])/len(a['tps']):.0f}")

    if "discrete" in results and "adamw" in results:
        d = results["discrete"]
        a = results["adamw"]
        mem_ratio = a["memory_mb"] / d["memory_mb"] if d["memory_mb"] > 0 else 0
        speed_ratio = (sum(d["tps"])/len(d["tps"])) / (sum(a["tps"])/len(a["tps"])) if sum(a["tps"]) > 0 else 0
        print(f"\n  Memory ratio:  {mem_ratio:.1f}× (AdamW / Discrete)")
        print(f"  Speed ratio:   {speed_ratio:.2f}× (Discrete / AdamW)")

    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
