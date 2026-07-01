"""Benchmark: run all trainers with TPS/FLOPs monitoring.

Usage:
    uv run python3 benchmark.py                          # all trainers
    uv run python3 benchmark.py --trainer 1bit           # one trainer
    uv run python3 benchmark.py --trainer ultimate --steps 50
"""

import math, time, sys, os, logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("benchmark")

# ── FLOPs estimator ───────────────────────────────────────────────────


def estimate_flops_per_step(
    n_layers: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch: int,
    use_bitlinear: bool = False,
    use_subqsa: bool = False,
    subqsa_cmp_stride: int = 16,
    subqsa_win_size: int = 512,
    subqsa_slc_topk: int = 16,
    subqsa_slc_block: int = 64,
    vocab_size: int = 4096,
) -> dict:
    """Estimate FLOPs per training step (forward + backward ≈ 3× forward)."""
    B, T, H = batch, seq_len, hidden_dim

    # Embedding: vocab → hidden (lookup, negligible matmul) → ignore

    # Attention projections: QKV + O = 4 linear layers
    # Each linear: 2 * B * T * in_features * out_features
    qkv_flops = 2 * B * T * H * (num_heads * head_dim) * 3  # Q,K,V
    o_flops = 2 * B * T * (num_heads * head_dim) * H  # O
    proj_flops = qkv_flops + o_flops  # per layer

    if use_subqsa:
        # Compression: MLP φ_k and φ_v: each 2 * B * T_cmp * (head_dim*block_len) * head_dim*2
        n_cmp = max(1, (T - 32) // subqsa_cmp_stride)
        cmp_phi_flops = (
            2 * B * num_heads * n_cmp * (head_dim * 32) * (head_dim * 2) * 2
        )  # *2 for k+v
        cmp_phi_flops += (
            2 * B * num_heads * n_cmp * (head_dim * 2) * head_dim * 2
        )  # second layer *2

        # Compression attention: B*num_heads * (T * n_cmp * head_dim * 2)
        cmp_attn_flops = 2 * B * num_heads * T * n_cmp * head_dim

        # Selection: full attention on top-k blocks
        n_sel = num_heads * subqsa_slc_topk * subqsa_slc_block
        sel_flops = 2 * B * num_heads * T * n_sel * head_dim

        # Sliding window
        win = min(subqsa_win_size, T)
        win_flops = 2 * B * num_heads * T * win * head_dim

        attn_flops = cmp_phi_flops + cmp_attn_flops + sel_flops + win_flops
    else:
        # Dense attention: QK^T + PV = 2 * (B * num_heads * T^2 * head_dim)
        attn_flops = 4 * B * num_heads * T * T * head_dim

    # GQA KV head expansion (negligible)
    # GQA gather for KV: no FLOPs (just view/expand)

    # FFN: gate + up + down = 3 linear layers
    ffn_flops = 2 * B * T * H * intermediate_dim * 3  # gate, up, down

    # LM head: hidden → vocab
    lm_flops = 2 * B * T * H * vocab_size

    per_layer = proj_flops + attn_flops + ffn_flops
    total_forward = per_layer * n_layers + lm_flops

    # Backward ≈ 2× forward (gradients for weights + activations)
    total_step = total_forward * 3

    return {
        "projections_per_layer": _fmt(proj_flops),
        "attention_per_layer": _fmt(attn_flops),
        "ffn_per_layer": _fmt(ffn_flops),
        "per_layer": _fmt(per_layer),
        "total_forward": _fmt(total_forward),
        "total_step_est": _fmt(total_step),
        "total_step_est_gflops": f"{total_step / 1e9:.1f} GFLOPs",
    }


def _fmt(n: float) -> str:
    if n > 1e12:
        return f"{n / 1e12:.2f} TFLOPs"
    if n > 1e9:
        return f"{n / 1e9:.1f} GFLOPs"
    return f"{n / 1e6:.0f} MFLOPs"


# ── Benchmark runner ──────────────────────────────────────────────────


@dataclass
class BenchResult:
    name: str
    steps: int
    avg_step_ms: float
    tps: float
    flops: dict
    loss_curve: str
    device: str


def run_benchmark(
    model: nn.Module,
    input_ids: torch.LongTensor,
    labels: torch.LongTensor,
    optimizer: torch.optim.Optimizer,
    name: str,
    n_steps: int = 30,
    warmup: int = 3,
    flops_estimate: Optional[dict] = None,
) -> BenchResult:
    """Run training steps with timing."""
    device = next(model.parameters()).device
    times = []
    losses = []

    for step in range(n_steps + warmup):
        if step >= warmup:
            t0 = time.perf_counter()

        optimizer.zero_grad()
        loss = (
            model.__class__.get_loss(model, input_ids, labels=labels)
            if hasattr(model, "get_loss")
            else nn.functional.cross_entropy(
                model(input_ids).view(-1, model(input_ids).size(-1)), labels.view(-1)
            )
        )
        loss.backward()
        optimizer.step()

        if step >= warmup:
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            losses.append(loss.item())

    avg_ms = sum(times) / len(times)
    tokens_per_step = input_ids.numel()
    tps = tokens_per_step / (avg_ms / 1000)

    # Loss trajectory summary
    if losses:
        l_init = losses[0]
        l_final = losses[-1]
        trend = "↓" if l_final < l_init else "↑"
        loss_str = f"{l_init:.2f} → {l_final:.2f} {trend}  (min={min(losses):.2f})"
    else:
        loss_str = "N/A"

    return BenchResult(
        name=name,
        steps=n_steps,
        avg_step_ms=avg_ms,
        tps=tps,
        flops=flops_estimate or {},
        loss_curve=loss_str,
        device=str(device),
    )


def print_results(results: list[BenchResult]):
    """Pretty-print benchmark results."""
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"{'BENCHMARK RESULTS':^72}")
    print(sep)
    for r in results:
        print(f"\n  {r.name}  ({r.device}, {r.steps} steps)")
        print(f"  {'─' * 50}")
        print(f"  Avg step time:     {r.avg_step_ms:>8.1f} ms")
        print(f"  Tokens/sec:        {r.tps:>8,.0f}")
        if r.flops:
            f = r.flops
            print(f"  Forward FLOPs:     {f.get('total_forward', '?'):>16}")
            print(f"  Step FLOPs (est):  {f.get('total_step_est', '?'):>16}")
        print(f"  Loss:              {r.loss_curve}")
    print(f"\n{sep}\n")


# ── Individual benchmarks ────────────────────────────────────────────


def bench_1bit(seq_len=128, batch=2, steps=30):
    import importlib.util

    _root = os.path.dirname(os.path.abspath(__file__))

    def _load(mod, file):
        spec = importlib.util.spec_from_file_location(mod, os.path.join(_root, file))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    _cfg = _load("1bit_config", "1bit-trainer/config.py")
    _mod = _load("1bit_model", "1bit-trainer/model.py")

    mc = _cfg.ModelConfig(
        vocab_size=4096,
        hidden_dim=256,
        intermediate_dim=512,
        num_layers=2,
        num_attention_heads=4,
        max_seq_len=seq_len,
    )
    tc = _cfg.TrainingConfig(max_steps=steps, learning_rate=1e-3)
    model = _mod.BitNetModel(mc)
    ids = torch.randint(0, 4096, (batch, seq_len))
    labels = ids.clone()

    flops = estimate_flops_per_step(
        n_layers=mc.num_layers,
        hidden_dim=mc.hidden_dim,
        intermediate_dim=mc.intermediate_dim,
        num_heads=mc.num_attention_heads,
        num_kv_heads=mc.num_kv_heads or mc.num_attention_heads,
        head_dim=mc.head_dim,
        seq_len=seq_len,
        batch=batch,
        use_bitlinear=True,
        vocab_size=mc.vocab_size,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return run_benchmark(
        model, ids, labels, opt, "1bit-trainer BitLinear", steps, flops_estimate=flops
    )


def bench_1bit_fp(seq_len=128, batch=2, steps=30):
    """FP baseline for 1bit comparison."""
    import importlib.util

    _root = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "1bit_comp", os.path.join(_root, "1bit-trainer/comparison.py")
    )
    _comp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_comp)
    make_fp_mlp = _comp.make_fp_mlp
    dim, depth = 256, 4
    model = make_fp_mlp(dim, depth)
    ids = torch.randn(batch, seq_len, dim)
    labels = torch.randn(batch, seq_len, dim)
    loss_fn = nn.MSELoss()

    def forward_loss(x, y):
        return loss_fn(model(x), y)

    model.get_loss = lambda x, y=None: forward_loss(x, y or x)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return run_benchmark(model, ids, labels, opt, "1bit-trainer FP (MLP)", steps)


def bench_subqsa(seq_len=128, batch=2, steps=20):
    _root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _root)
    from subqsa_trainer.config import ModelConfig, SubQSAConfig
    from subqsa_trainer.model import SubQSAModel

    mc = ModelConfig(
        vocab_size=4096,
        hidden_dim=256,
        intermediate_dim=512,
        num_layers=2,
        num_attention_heads=4,
        max_seq_len=seq_len,
    )
    mc.subqsa = SubQSAConfig(
        cmp_block=16, cmp_stride=8, slc_block=32, slc_topk=4, win_size=32
    )
    model = SubQSAModel(mc)
    ids = torch.randint(0, 4096, (batch, seq_len))
    labels = ids.clone()

    flops = estimate_flops_per_step(
        n_layers=mc.num_layers,
        hidden_dim=mc.hidden_dim,
        intermediate_dim=mc.intermediate_dim,
        num_heads=mc.num_attention_heads,
        num_kv_heads=mc.num_kv_heads or mc.num_attention_heads,
        head_dim=mc.head_dim,
        seq_len=seq_len,
        batch=batch,
        use_subqsa=True,
        subqsa_cmp_stride=mc.subqsa.cmp_stride,
        subqsa_slc_topk=mc.subqsa.slc_topk,
        subqsa_slc_block=mc.subqsa.slc_block,
        subqsa_win_size=mc.subqsa.win_size,
        vocab_size=mc.vocab_size,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return run_benchmark(
        model, ids, labels, opt, "subqsa-trainer SubQSA", steps, flops_estimate=flops
    )


def bench_ultimate(seq_len=128, batch=2, steps=20):
    _root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _root)
    from ultimate_trainer.config import UltimateModelConfig
    from ultimate_trainer.model import UltimateModel

    mc = UltimateModelConfig(
        vocab_size=4096,
        hidden_dim=256,
        intermediate_dim=512,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        max_seq_len=seq_len,
        use_bitlinear=True,
        cmp_block=16,
        cmp_stride=8,
        slc_block=32,
        slc_topk=4,
        win_size=32,
    )
    model = UltimateModel(mc)
    ids = torch.randint(0, 4096, (batch, seq_len))
    labels = ids.clone()

    flops = estimate_flops_per_step(
        n_layers=mc.num_layers,
        hidden_dim=mc.hidden_dim,
        intermediate_dim=mc.intermediate_dim,
        num_heads=mc.num_attention_heads,
        num_kv_heads=mc.num_kv_heads or mc.num_attention_heads,
        head_dim=mc.head_dim,
        seq_len=seq_len,
        batch=batch,
        use_bitlinear=True,
        use_subqsa=True,
        subqsa_cmp_stride=mc.cmp_stride,
        subqsa_slc_topk=mc.slc_topk,
        subqsa_slc_block=mc.slc_block,
        subqsa_win_size=mc.win_size,
        vocab_size=mc.vocab_size,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return run_benchmark(
        model,
        ids,
        labels,
        opt,
        "ultimate-trainer (BitLinear + SubQSA)",
        steps,
        flops_estimate=flops,
    )


def bench_ultimate_fp(seq_len=128, batch=2, steps=20):
    """FP-only baseline for ultimate comparison."""
    _root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _root)
    from ultimate_trainer.config import UltimateModelConfig
    from ultimate_trainer.model import UltimateModel

    mc = UltimateModelConfig(
        vocab_size=4096,
        hidden_dim=256,
        intermediate_dim=512,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        max_seq_len=seq_len,
        use_bitlinear=False,  # FP
        cmp_block=16,
        cmp_stride=8,
        slc_block=32,
        slc_topk=4,
        win_size=32,
    )
    model = UltimateModel(mc)
    ids = torch.randint(0, 4096, (batch, seq_len))
    labels = ids.clone()

    flops = estimate_flops_per_step(
        n_layers=mc.num_layers,
        hidden_dim=mc.hidden_dim,
        intermediate_dim=mc.intermediate_dim,
        num_heads=mc.num_attention_heads,
        num_kv_heads=mc.num_kv_heads or mc.num_attention_heads,
        head_dim=mc.head_dim,
        seq_len=seq_len,
        batch=batch,
        use_bitlinear=False,
        use_subqsa=True,
        subqsa_cmp_stride=mc.cmp_stride,
        subqsa_slc_topk=mc.slc_topk,
        subqsa_slc_block=mc.slc_block,
        subqsa_win_size=mc.win_size,
        vocab_size=mc.vocab_size,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return run_benchmark(
        model,
        ids,
        labels,
        opt,
        "ultimate-trainer FP-only (SubQSA)",
        steps,
        flops_estimate=flops,
    )


# ── Main ──────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark all trainers with TPS/FLOPs"
    )
    parser.add_argument(
        "--trainer", choices=["1bit", "subqsa", "ultimate", "all"], default="all"
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    # Set up import paths
    _root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _root)
    sys.path.insert(0, os.path.join(_root, "1bit-trainer"))
    sys.path.insert(0, os.path.join(_root, "subqsa-trainer"))
    sys.path.insert(0, os.path.join(_root, "ultimate-trainer"))

    results = []
    label = f"seq_len={args.seq_len}, batch={args.batch}, {args.steps} steps"
    print(f"Running benchmarks: {label}\n")

    t_all = time.perf_counter()

    if args.trainer in ("1bit", "all"):
        r = bench_1bit(args.seq_len, args.batch, args.steps)
        results.append(r)

    if args.trainer in ("subqsa", "all"):
        r = bench_subqsa(args.seq_len, args.batch, args.steps)
        results.append(r)

    if args.trainer in ("ultimate", "all"):
        r = bench_ultimate(args.seq_len, args.batch, args.steps)
        results.append(r)
        r2 = bench_ultimate_fp(args.seq_len, args.batch, args.steps)
        results.append(r2)

    elapsed = time.perf_counter() - t_all

    print_results(results)
    print(f"  Total benchmark time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
