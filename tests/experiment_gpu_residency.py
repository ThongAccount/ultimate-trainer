"""GPU-residency experiment (session 9).

Measures whether host-side transfers/sync/dispatch limit the training step.

Config A (BASELINE): current trainer behavior — sync + loss.item() every step.
Config B (RESIDENT):  fully async stepping; loss read back every K steps only;
                      no per-step sync at all. All tensors already on GPU.

Also records: cudaMemcpy count, cudaDeviceSynchronize count, accumulated CPU
op time via torch.profiler, and peak GPU memory.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch

VOCAB = 5152 if os.environ.get("SMALLVOCAB") else 50272

B, SEQ, K_, NL = 32, 512, 1024, 6
BATCH = B * SEQ

from kernels.packed_ternary import PackedTernaryLinear
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, K_)
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(K_) for _ in range(NL)])
        self.fc1s = torch.nn.ModuleList([PackedTernaryLinear(K_, 4 * K_) for _ in range(NL)])
        self.fc2s = torch.nn.ModuleList([PackedTernaryLinear(4 * K_, K_) for _ in range(NL)])
        self.head = PackedTernaryLinear(K_, VOCAB)
    def forward(self, x):
        Bt, T = x.shape
        h = self.embed(x).half().view(Bt * T, K_)
        scale = K_ ** -0.5
        for i in range(NL):
            h = self.norms[i](h.float()).half()
            h = self.fc1s[i](h) * scale
            h = F.gelu(h)
            h = self.fc2s[i](h) * scale
        h = self.head(h) * scale
        return h.view(Bt, T, VOCAB)


def run_config(steps, per_step_sync, item_every, label):
    torch.manual_seed(0)
    model = Model().cuda().train()
    x = torch.randint(0, VOCAB, (B, SEQ), device="cuda")
    y = x.clone()
    loss_fn = torch.nn.CrossEntropyLoss()

    # warmup (compiles kernels, no sync semantics here since first steps
    # synchronise implicitly via autograd allocator paths)
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model(x).view(-1, VOCAB), y.view(-1))
        loss.backward()
    torch.cuda.synchronize()

    losses = []
    t0 = time.perf_counter()
    for i in range(steps):
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model(x).view(-1, VOCAB), y.view(-1))
        loss.backward()
        if per_step_sync:
            torch.cuda.synchronize()
        if (i % item_every) == 0:
            losses.append(float(loss))  # forces a sync at readback
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    toks = steps * BATCH
    print(f"  [{label}] {steps} steps: {dt*1000/steps:.1f} ms/step, "
          f"{toks/dt:,.0f} tok/s  (last loss {losses[-1]:.4f})", flush=True)
    return dt / steps

def count_runtime_events():
    """One profiled step: count memcpy / sync / kernel-launch runtime calls."""
    torch.manual_seed(0)
    model = Model().cuda().train()
    x = torch.randint(0, VOCAB, (B, SEQ), device="cuda")
    y = x.clone()
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model(x).view(-1, VOCAB), y.view(-1))
        loss.backward()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model(x).view(-1, VOCAB), y.view(-1))
        loss.backward()
        torch.cuda.synchronize()

    counts = {}
    for e in prof.events():
        n = e.name
        for key in ("cudaMemcpy", "cudaMemset", "cudaDeviceSynchronize",
                    "cudaLaunchKernel", "cudaStreamSynchronize",
                    "cudaEventRecord", "cudaHostAlloc", "Memset"):
            if key in n:
                counts[key] = counts.get(key, 0) + 1
    print("  --- runtime API event counts (1 step) ---", flush=True)
    for k, v in sorted(counts.items()):
        print(f"  {k:28s} {v}")
    print(f"  peak mem: alloc={torch.cuda.max_memory_allocated()/2**30:.2f} GiB "
          f"reserved={torch.cuda.max_memory_reserved()/2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    count_runtime_events()
    a = run_config(10, per_step_sync=True,  item_every=1,  label="A baseline (sync+item/step)")
    b = run_config(10, per_step_sync=False, item_every=20, label="B resident (no sync, item @20)")
    c = run_config(10, per_step_sync=False, item_every=1,  label="C no sync, item every step")
    print(f"\n  B/A = {b/a:.3f}  (resident speedup: {(1-b/a)*100:+.1f}%)")
    print(f"  C/A = {c/a:.3f}  (item-only cost: {(c-b)/b*100:+.1f}% over B)")
