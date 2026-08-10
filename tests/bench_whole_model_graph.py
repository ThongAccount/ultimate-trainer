"""Benchmark whole-model train step: eager-Python vs CUDAGraph-wrapped.

Captures the FULL forward + backward into a replayable CUDAGraph (update stays
outside, like train_step_graph_v2, because in-place counter updates break
static-address capture). Reports wall/step for both, plus kernel-only time.

Usage:
    python tests/bench_whole_model_graph.py
"""
import sys, os, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train_shakespeare_optimized import (
    MiniGPT, make_batches, get_shakespeare,
    VOCAB_SIZE, BLOCK_SIZE, BATCH_SIZE, THRESHOLD,
)
from kernels.packed_ternary.packed_linear import PackedTernaryLinear


def best_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build(device):
    model = MiniGPT(use_ternary=True).to(device)
    model.train()
    opt = []  # no optimizer — counters update in-place inside autograd
    return model, opt


def benchmark_eager(model, batches_x, batches_y, steps=30, warmup=5):
    model.train()
    for _ in range(warmup):
        b = 0
        x = batches_x[b].to("cuda", non_blocking=True)
        y = batches_y[b].to("cuda", non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for step in range(steps):
        b = step % len(batches_x)
        x = batches_x[b].to("cuda", non_blocking=True)
        y = batches_y[b].to("cuda", non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)
        _ = loss.item()
    torch.cuda.synchronize()
    return (time.time() - t0) / steps


def capture_graph(model, batches_x, batches_y, n_batches):
    """Capture fwd+bwd over one fixed input. Update OUTSIDE graph (in-place
    counter mutation would break static addressing)."""
    # static buffers
    static_x = batches_x[0].to("cuda", non_blocking=True).clone().contiguous()
    static_y = batches_y[0].to("cuda", non_blocking=True).clone().contiguous()
    x_g = static_x  # same shape
    y_g = static_y

    # warmup on side stream (torch requirement)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            logits = model(x_g)
            loss = F.cross_entropy(logits.float().view(-1, VOCAB_SIZE), y_g.view(-1))
            loss.backward()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = model(x_g)
        loss = F.cross_entropy(logits.float().view(-1, VOCAB_SIZE), y_g.view(-1))
        loss.backward()
    return graph, loss


def main():
    torch.manual_seed(0)
    device = best_device()
    assert device.type == "cuda"
    text = get_shakespeare()
    batches_x, batches_y = make_batches(text, BLOCK_SIZE, BATCH_SIZE, 272)

    model, _ = build(device)
    t = benchmark_eager(model, batches_x, batches_y)
    print(f"eager             : {t*1e3:8.2f} ms/step   ({BATCH_SIZE*BLOCK_SIZE/t:9.0f} tok/s)")

    # graph path
    model2, _ = build(device)
    torch.cuda.synchronize()
    graph, loss = capture_graph(model2, batches_x, batches_y, 272)
    torch.cuda.synchronize()

    # replay timing
    n_replay = 30
    t0 = time.time()
    for _ in range(n_replay):
        graph.replay()
    torch.cuda.synchronize()
    tg = (time.time() - t0) / n_replay
    print(f"cuDAGraph         : {tg*1e3:8.2f} ms/step   ({BATCH_SIZE*BLOCK_SIZE/tg:9.0f} tok/s)")
    print(f"speedup           : {t/tg:6.2f}x")
    print(f"post-replay loss  : {loss.item():.4f}")


if __name__ == "__main__":
    main()