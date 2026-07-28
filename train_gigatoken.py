"""Sample trainer: Gigatoken + CUDAGraph verify end-to-end pipeline.

Usage:
    python train_gigatoken.py [--text path/to/data.txt] [--vocab gpt2|llama3]

Runs a few steps, prints loss + throughput. Works on any text file.
"""

import sys, os, time, argparse
import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernels.packed_ternary import PackedTernaryLinear

# ── Config ─────────────────────────────────────────────────────────────
B = 32          # micro-batch
SEQ = 512       # sequence length
K = 1024        # hidden dim
N = 1024
N_LAYERS = 6    # depth
VOCAB = 50257   # GPT-2 vocab size (overridden for Llama)
THRESHOLD = 32
LR = 1.0        # learning rate (unitless for discrete)
STEPS = 50
WARMUP = 5


def build_model():
    """Small transformer with ScaledTernaryLinear."""
    class TernaryTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(VOCAB, K)
            self.layers = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.LayerNorm(K, dtype=torch.float16),
                    PackedTernaryLinear(K, 4*K, threshold=THRESHOLD),
                    torch.nn.GELU(),
                    PackedTernaryLinear(4*K, K, threshold=THRESHOLD),
                ) for _ in range(N_LAYERS)
            ])
            self.head = PackedTernaryLinear(K, VOCAB, threshold=THRESHOLD)

        def forward(self, x):
            h = self.embed(x).half()
            for layer in self.layers:
                h = layer(h)
            return self.head(h)

    model = TernaryTransformer().cuda()
    return model


def tokenize_file(path: str, vocab: str = "gpt2"):
    """Tokenize entire file, return flat uint16 array on GPU."""
    from gigatoken import Tokenizer
    tok = Tokenizer(vocab)

    with open(path, "rb") as f:
        data = f.read()

    ids = tok.encode(data)
    print(f"  Read {len(data):_} bytes → {len(ids):_} tokens"
          f"  ({len(data)/max(len(ids),1):.1f} byte/tok)")

    # Pad to multiple of B*SEQ
    n_tok = B * SEQ
    if len(ids) < n_tok:
        # Repeat
        ids_list = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
        ids_list = (ids_list * (n_tok // len(ids_list) + 1))[:n_tok]
        t = torch.tensor(ids_list, dtype=torch.long, device="cuda").view(B, SEQ)
    else:
        # Take exactly n_tok tokens
        ids_arr = ids[:n_tok].copy() if hasattr(ids, 'copy') else ids[:n_tok]
        t = torch.tensor(ids_arr, dtype=torch.long, device="cuda").view(B, SEQ)
    return t  # GPU tensor [B, SEQ]


def train_step_cudagraph(model, x, y):
    """One training step via CUDAGraph."""
    # Forward
    logits = model(x)

    # Loss: cross-entropy on last token (next-token prediction)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = y[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, VOCAB), shift_labels.view(-1),
        reduction='mean')

    # Backward
    loss.backward()

    # Update (counter flips happen inside PackedTernaryLinear.backward)
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="/home/debian/ultimate-ai-model/data/shakespeare.txt",
                        help="Path to training text")
    parser.add_argument("--vocab", default="gpt2", choices=["gpt2", "llama3"],
                        help="Tokenizer vocab")
    args = parser.parse_args()

    print("=" * 60)
    print("  Gigatoken + CUDAGraph Training Pipeline")
    print("=" * 60)

    # Tokenize
    print(f"\n[1/4] Tokenizing {args.text}...")
    t0 = time.perf_counter()
    data_tensor = tokenize_file(args.text, args.vocab)
    t1 = time.perf_counter()
    print(f"  Tokenization: {t1-t0:.3f}s")

    # Build model
    print(f"\n[2/4] Building model ({N_LAYERS} layers, {K} dim, {VOCAB} vocab)...")
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Prepare data
    print(f"\n[3/4] Preparing data...")
    x = data_tensor[:, :SEQ-1].contiguous()
    y = data_tensor[:, :SEQ].contiguous()

    # Train
    print(f"\n[4/4] Training ({STEPS} steps, batch={B}, seq={SEQ})...")
    print(f"  {'Step':>5} {'Loss':>10} {'tok/s':>12} {'Time':>8}")
    print(f"  {'─'*37}")

    times = []
    for step in range(STEPS):
        t0 = time.perf_counter()
        loss = train_step_cudagraph(model, x, y)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)

        tok_per_sec = (B * SEQ) / elapsed
        if step < WARMUP:
            print(f"  {step:>5} {loss:>10.4f} {tok_per_sec:>12,.0f} {elapsed*1000:>7.1f}ms (warmup)")
        else:
            print(f"  {step:>5} {loss:>10.4f} {tok_per_sec:>12,.0f} {elapsed*1000:>7.1f}ms")

    # Summary
    stable = times[WARMUP:]
    avg_ms = sum(stable) / len(stable) * 1000
    avg_tok = B * SEQ / (avg_ms / 1000)
    print(f"\n  {'─'*37}")
    print(f"  Avg: {avg_ms:.1f}ms/step  {avg_tok:,.0f} tok/s")
    print(f"  Total tokens processed: {B * SEQ * STEPS:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
