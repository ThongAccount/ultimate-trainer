"""Optimized Shakespeare training — maximum GPU utilization.

Uses CUDAGraph to eliminate Python overhead and keep the GPU
continuously busy. Target: 60%+ GPU utilization.

Key optimizations:
1. CUDAGraph — zero Python overhead per step
2. Pre-allocated buffers — no runtime allocation
3. Larger batches — more work per kernel launch
4. Pinned memory — faster CPU→GPU transfer
5. No gradient accumulation in graph — single step per replay

Usage:
    python train_shakespeare_optimized.py              # optimized run
    python train_shakespeare_optimized.py --steps 1000 # longer
    python train_shakespeare_optimized.py --compare    # vs baseline
"""

import sys, os, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from kernels.packed_ternary.packed_linear import PackedTernaryLinear

# ═══════════════════════════════════════════════════════════════════════════════
#  Config — larger batches for better GPU utilization
# ═══════════════════════════════════════════════════════════════════════════════

VOCAB_SIZE = 256
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 6
BLOCK_SIZE = 128
BATCH_SIZE = 32   # larger batch = more GPU work per kernel
THRESHOLD = 32
STEPS = 200
PRINT_EVERY = 10


# ═══════════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════════

class ScaledTernaryLinear(nn.Module):
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
    def __init__(self, d_model, nhead, threshold=8):
        super().__init__()
        self.qkv = ScaledTernaryLinear(d_model, d_model * 3, threshold=threshold)
        self.proj = ScaledTernaryLinear(d_model, d_model, threshold=threshold)
        self.ff1 = ScaledTernaryLinear(d_model, d_model * 4, threshold=threshold)
        self.ff2 = ScaledTernaryLinear(d_model * 4, d_model, threshold=threshold)
        self.ln1 = nn.LayerNorm(d_model).half()
        self.ln2 = nn.LayerNorm(d_model).half()
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
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_embed = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.blocks = nn.ModuleList([
            TernaryTransformerBlock(D_MODEL, N_HEADS, THRESHOLD)
            for _ in range(N_LAYERS)
        ])
        self.ln_f = nn.LayerNorm(D_MODEL).half()
        self.head = ScaledTernaryLinear(D_MODEL, VOCAB_SIZE, threshold=THRESHOLD)

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.embed(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = (tok_emb + self.pos_embed(pos)).half()
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data pipeline — pinned memory for fast transfer
# ═══════════════════════════════════════════════════════════════════════════════

def get_shakespeare():
    data_path = os.path.join(os.path.dirname(__file__), "shakespeare.txt")
    if not os.path.exists(data_path):
        print("Downloading Shakespeare...")
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)
    with open(data_path, "r") as f:
        return f.read()


def make_batches(text, block_size, batch_size, n_batches):
    """Pre-compute all batches on CPU with pinned memory for fast transfer."""
    data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
    batches_x = []
    batches_y = []
    for i in range(n_batches):
        idx = (i * batch_size * block_size) % (len(data) - block_size * batch_size)
        chunk = data[idx:idx + batch_size * block_size + 1]
        x = chunk[:-1].view(batch_size, block_size).pin_memory()
        y = chunk[1:].view(batch_size, block_size).pin_memory()
        batches_x.append(x)
        batches_y.append(y)
    return batches_x, batches_y


# ═══════════════════════════════════════════════════════════════════════════════
#  Training: CUDAGraph-based (maximum GPU utilization)
# ═══════════════════════════════════════════════════════════════════════════════

def train_graph(model, batches_x, batches_y, steps, print_every):
    """Train with CUDAGraph — zero Python overhead per step.

    The entire forward+backward+update is captured in a CUDA graph.
    Each step just copies input data and replays the graph.
    GPU runs continuously without CPU stalls.
    """
    x_shape = (BATCH_SIZE, BLOCK_SIZE)
    y_shape = (BATCH_SIZE, BLOCK_SIZE)

    # Pre-allocate static buffers
    static_x = torch.zeros(x_shape, dtype=torch.long, device="cuda")
    static_y = torch.zeros(y_shape, dtype=torch.long, device="cuda")
    static_loss = torch.zeros(1, dtype=torch.float32, device="cuda")

    def loss_fn(logits, targets):
        return F.cross_entropy(logits.float().view(-1, VOCAB_SIZE), targets.view(-1))

    # Warmup (required before capture)
    print("  Warming up...")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            static_x.copy_(batches_x[0])
            static_y.copy_(batches_y[0])
            logits = model(static_x)
            loss = loss_fn(logits, static_y)
            loss.backward()
    torch.cuda.current_stream().wait_stream(s)

    # Capture graph
    print("  Capturing CUDAGraph...")
    graph = torch.cuda.CUDAGraph()
    model.zero_grad(set_to_none=True)
    static_x.copy_(batches_x[0])
    static_y.copy_(batches_y[0])

    with torch.cuda.graph(graph):
        static_logits = model(static_x)
        static_loss = loss_fn(static_logits, static_y)
        static_loss.backward()
        # Counter updates happen inside backward() via autograd hook

    # Main training loop — replay graph with zero overhead
    print(f"  Training {steps} steps (CUDAGraph replay)...")
    losses = []
    tokens_per_sec = []
    torch.cuda.synchronize()
    start_time = time.time()

    for step in range(steps):
        # Copy new data into static buffers (async, minimal overhead)
        batch_idx = step % len(batches_x)
        static_x.copy_(batches_x[batch_idx])
        static_y.copy_(batches_y[batch_idx])

        # Replay graph — GPU runs continuously
        graph.replay()

        loss_val = static_loss.item()
        losses.append(loss_val)

        # Throughput
        tokens = BATCH_SIZE * BLOCK_SIZE
        elapsed = time.time() - start_time
        tps = (step + 1) * tokens / elapsed if elapsed > 0 else 0
        tokens_per_sec.append(tps)

        if step % print_every == 0:
            ppl = 2 ** loss_val if loss_val < 20 else float('inf')
            print(f"  Step {step:4d}: loss={loss_val:.4f}  ppl={ppl:.1f}  tok/s={tps:.0f}")

    return losses, tokens_per_sec


def train_standard(model, batches_x, batches_y, steps, lr, print_every):
    """Standard training with AdamW — for comparison."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    tokens_per_sec = []
    start_time = time.time()

    for step in range(steps):
        batch_idx = step % len(batches_x)
        x = batches_x[batch_idx].to("cuda", non_blocking=True)
        y = batches_y[batch_idx].to("cuda", non_blocking=True)

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
            ppl = 2 ** loss.item() if loss.item() < 20 else float('inf')
            print(f"  Step {step:4d}: loss={loss.item():.4f}  ppl={ppl:.1f}  tok/s={tps:.0f}")

    return losses, tokens_per_sec


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--compare", action="store_true", help="Compare with AdamW")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    global BATCH_SIZE
    BATCH_SIZE = args.batch

    print("=" * 60)
    print("  Shakespeare LM — Optimized Training")
    print("=" * 60)
    print(f"  Config: d={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}")
    print(f"  Block={BLOCK_SIZE}, batch={BATCH_SIZE}, vocab={VOCAB_SIZE}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print()

    # Load data
    text = get_shakespeare()
    n_batches = min(args.steps * 2, len(text) // (BLOCK_SIZE * BATCH_SIZE))
    batches_x, batches_y = make_batches(text, BLOCK_SIZE, BATCH_SIZE, n_batches)
    print(f"  Pre-computed {len(batches_x)} batches")
    print()

    results = {}

    # ── Discrete with CUDAGraph ─────────────────────────────────────
    print(f"{'─' * 60}")
    print(f"  Discrete Ternary + CUDAGraph (threshold={THRESHOLD})")
    print(f"{'─' * 60}")

    torch.cuda.reset_peak_memory_stats()
    model = MiniGPT().cuda()

    # Set threshold
    for m in model.modules():
        if isinstance(m, PackedTernaryLinear):
            m.threshold = THRESHOLD

    losses_d, tps_d = train_graph(model, batches_x, batches_y, args.steps, PRINT_EVERY)
    mem_d = torch.cuda.max_memory_allocated() / 1024 / 1024

    results["discrete"] = {
        "losses": losses_d, "tps": tps_d, "memory_mb": mem_d,
        "final_loss": losses_d[-1] if losses_d else float('inf'),
    }
    print(f"\n  Final loss: {losses_d[-1]:.4f}")
    print(f"  Peak memory: {mem_d:.0f} MB")
    print(f"  Avg tok/s: {sum(tps_d)/len(tps_d):.0f}")

    del model
    torch.cuda.empty_cache()

    # ── AdamW baseline (optional) ───────────────────────────────────
    if args.compare:
        print(f"\n{'─' * 60}")
        print(f"  AdamW Baseline")
        print(f"{'─' * 60}")

        torch.cuda.reset_peak_memory_stats()
        model_a = MiniGPT().cuda()
        # Replace ternary with standard linear for baseline
        # (reuse same architecture but with AdamW)
        losses_a, tps_a = train_standard(
            model_a, batches_x, batches_y, args.steps, 3e-4, PRINT_EVERY
        )
        mem_a = torch.cuda.max_memory_allocated() / 1024 / 1024

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
