"""Same-instance A/B: eager training step vs CUDA-graph-captured whole step.

Captures the ENTIRE gigatoken step (13 PackedTernaryLinear fwd+bwd+update,
norms, gelu, CE loss, autograd backward) into a CUDAGraph with static input
tensor, then replays per step. Compares per-step latency in-process.

Update is in-place on W/counter (static addresses) so it can live in the graph
(atomicCAS inside kernel is fine for capture).

Runs on Modal T4, mounts local tree.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
import torch

from train_gigatoken import build_model, B, SEQ, VOCAB, K, N_LAYERS

def eager_step(model, x, y):
    scale = K ** -0.5
    Bt, T = x.shape
    h = model.embed(x).half().view(Bt * T, K)
    for i in range(N_LAYERS):
        h = model.norms[i](h.float()).half()
        h = model.fc1s[i](h) * scale
        h = torch.nn.functional.gelu(h)
        h = model.fc2s[i](h) * scale
    h = model.head(h) * scale
    logits = h.view(Bt, T, VOCAB)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, VOCAB), y.reshape(-1), reduction="mean")
    loss.backward()
    return loss.detach()

def main():
    torch.manual_seed(0)
    dev = "cuda"
    print("building model...", flush=True)
    model = build_model().cuda()
    # warmup builds JIT kernels
    x0 = torch.randint(0, VOCAB, (B, SEQ), device=dev)
    y0 = x0.clone()
    for _ in range(2):
        eager_step(model, x0, y0)
    torch.cuda.synchronize()

    # Static buffers for graph
    x_static = torch.randint(0, VOCAB, (B, SEQ), device=dev)
    y_static = x_static.clone()

    # Capture graph (needs warmup on side stream)
    print("capturing graph...", flush=True)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            eager_step(model, x_static, y_static)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        eager_step(model, x_static, y_static)
    print("graph captured", flush=True)

    def run_graph():
        g.replay()

    # Time eager
    te = []
    for _ in range(3):
        x = torch.randint(0, VOCAB, (B, SEQ), device=dev)
        y = x.clone()
        t0 = time.perf_counter()
        eager_step(model, x, y)
        torch.cuda.synchronize()
        te.append((time.perf_counter() - t0) * 1000)

    # Time graph replay
    tg = []
    for _ in range(3):
        x_static.copy_(torch.randint(0, VOCAB, (B, SEQ), device=dev))
        t0 = time.perf_counter()
        run_graph()
        torch.cuda.synchronize()
        tg.append((time.perf_counter() - t0) * 1000)

    me, mg = min(te), min(tg)
    print(f"EAGER: {te}", flush=True)
    print(f"GRAPH: {tg}", flush=True)
    print(f"EAGER avg {sum(te)/len(te):.1f}ms  GRAPH avg {sum(tg)/len(tg):.1f}ms  delta {100.0*(sum(tg)/len(tg)-sum(te)/len(te))/(sum(te)/len(te)):+.2f}%", flush=True)

if __name__ == "__main__":
    main()