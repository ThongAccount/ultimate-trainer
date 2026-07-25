"""Convergence test for the discrete optimizer.

Tests whether PackedTernaryLinear's counter-based optimizer can actually
learn a task by reducing loss over training steps.

The discrete optimizer has NO learning rate — it's a purely sign-based
counter system:
  dW > 0 → counter-- → when |counter| > T: flip weight, reset counter

Convergence here means "the counter mechanism finds weights that
reduce loss", not "matches AdamW accuracy".

Test structure:
  1. Create a small 3-layer MLP with PackedTernaryLinear
  2. Train on a random embedding task (learn to map index → output)
  3. Track loss at each step
  4. Compare loss trajectory to a frozen-random baseline
"""

from __future__ import annotations

import sys, os, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernels.packed_ternary import PackedTernaryLinear

if not torch.cuda.is_available():
    print("CUDA not available")
    sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════
#  Tiny MLP with PackedTernaryLinear
# ═══════════════════════════════════════════════════════════════════════

class DiscreteMLP(nn.Module):
    """Small 3-layer MLP with discrete ternary weights."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, threshold: int = 64):
        super().__init__()
        self.fc1 = PackedTernaryLinear(d_in, d_hidden, threshold=threshold)
        self.fc2 = PackedTernaryLinear(d_hidden, d_hidden, threshold=threshold)
        self.fc3 = PackedTernaryLinear(d_hidden, d_out, threshold=threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class FloatMLP(nn.Module):
    """Same architecture with standard float weights + AdamW baseline."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_hidden, bias=False)
        self.fc3 = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


# ═══════════════════════════════════════════════════════════════════════
#  Synthetic task: learn a fixed random linear transformation
# ═══════════════════════════════════════════════════════════════════════

def make_data(batch: int, d_in: int, d_out: int, num_batches: int, seed: int = 0):
    """Generate synthetic regression task: learn random linear map.

    Returns generator of (x, y) batches.
    """
    torch.manual_seed(seed)
    W_true = torch.randn(d_out, d_in) / math.sqrt(d_in)
    for _ in range(num_batches):
        x = torch.randn(batch, d_in)
        y = x @ W_true.T  # (batch, d_out)
        yield x, y


# ═══════════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════════

def train_discrete(
    d_in: int, d_hidden: int, d_out: int,
    threshold: int,
    steps: int = 200,
    batch_size: int = 32,
    lr: float = 0.0,  # unused — discrete has no learning rate
) -> list[float]:
    """Train the discrete MLP, return loss history."""
    model = DiscreteMLP(d_in, d_hidden, d_out, threshold=threshold).cuda().half()
    losses = []

    for step, (x, y) in enumerate(make_data(batch_size, d_in, d_out, steps, seed=42)):
        if step >= steps:
            break
        x = x.cuda().half()
        y = y.cuda().half()

        y_pred = model(x)
        loss = F.mse_loss(y_pred, y)

        # Diagnostic init (before first backward to capture initial state)
        if step == 0:
            W_initial = model.fc1.W_packed.clone()
            counter_before = model.fc1.counter.clone()
            print(f"  counter contiguous: {model.fc1.counter.is_contiguous()}")
            print(f"  counter device: {model.fc1.counter.device}")
            print(f"  counter dtype: {model.fc1.counter.dtype}")

        # The backward() triggers PackedTernaryLinearFn.backward(),
        # which computes dX and applies the fused counter update.
        loss.backward()
        torch.cuda.synchronize()
        # Check for any CUDA error that might indicate kernel failure
        err = torch.cuda.get_last_error_string() if hasattr(torch.cuda, 'get_last_error_string') else "N/A"
        if step == 0:
            cuda_err = torch.cuda.current_stream().query()  # False if error pending
            print(f"  cuda stream ok: {cuda_err}")

        # Check counter immediately after first backward
        if step == 0:
            for name in ['fc1', 'fc2', 'fc3']:
                layer = getattr(model, name)
                c_min = layer.counter.min().item()
                c_max = layer.counter.max().item()
                c_nonzero = (layer.counter != 0).sum().item()
                print(f"  after step 0 backward: {name}.counter=[{c_min},{c_max}], nonzero={c_nonzero}/{layer.counter.numel()}")
                print(f"    {name}.counter.data_ptr()={layer.counter.data_ptr():#x}")

        losses.append(loss.item())
        if step == steps - 1:
            for name in ['fc1', 'fc2', 'fc3']:
                layer = getattr(model, name)
                print(f"  {name}.counter range=[{layer.counter.min().item()},{layer.counter.max().item()}] nonzero={(layer.counter != 0).sum().item()}/{layer.counter.numel()}")

        if step % 20 == 0 or step == steps - 1:
            print(f"  step {step:4d}: loss={loss.item():.6f}")

    return losses


def train_float(
    d_in: int, d_hidden: int, d_out: int,
    steps: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> list[float]:
    """Train the float MLP with AdamW, return loss history."""
    model = FloatMLP(d_in, d_hidden, d_out).cuda().half()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []

    for step, (x, y) in enumerate(make_data(batch_size, d_in, d_out, steps, seed=42)):
        if step >= steps:
            break
        x = x.cuda().half()
        y = y.cuda().half()

        y_pred = model(x)
        loss = F.mse_loss(y_pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())

        if step % 20 == 0 or step == steps - 1:
            print(f"  step {step:4d}: loss={loss.item():.6f}")

    return losses


def baseline_loss(
    d_in: int, d_hidden: int, d_out: int,
    batch_size: int = 32,
    num_batches: int = 10,
) -> float:
    """Loss of a frozen random discrete model (no training)."""
    model = DiscreteMLP(d_in, d_hidden, d_out, threshold=9999).cuda().half()
    total_loss = 0.0
    count = 0
    for x, y in make_data(batch_size, d_in, d_out, num_batches, seed=42):
        x = x.cuda().half()
        y = y.cuda().half()
        with torch.no_grad():
            y_pred = model(x)
            loss = F.mse_loss(y_pred, y)
        total_loss += loss.item()
        count += 1
    return total_loss / count


# ═══════════════════════════════════════════════════════════════════════
#  Run
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    d_in, d_hidden, d_out = 16, 32, 8
    steps = 200
    batch_size = 32

    print(f"Model: {d_in}→{d_hidden}→{d_out}, {steps} steps, batch={batch_size}")
    print()

    # Direct update() test — bypass autograd, call update() directly
    print("Direct update() test (bypass autograd)...")
    model = DiscreteMLP(d_in, d_hidden, d_out, threshold=8).cuda().half()
    x = torch.randn(batch_size, d_in, dtype=torch.float16, device="cuda")
    y = torch.randn(batch_size, d_out, dtype=torch.float16, device="cuda")
    y_pred = model(x)
    dY = 2 * (y_pred - y) / (batch_size * d_out)
    from kernels.packed_ternary.pack_update import update
    # X for fc3 = activation after fc2+ReLU (before fc3)
    with torch.no_grad():
        fc3_in = torch.relu(model.fc2(torch.relu(model.fc1(x))))
    c_before = model.fc3.counter.clone()
    update(model.fc3.W_packed, model.fc3.counter, fc3_in, dY, threshold=8)
    torch.cuda.synchronize()
    c_after = model.fc3.counter
    changed = not torch.equal(c_before, c_after)
    print(f"  Direct update() fc3 counter changed: {changed}")
    if changed:
        nonzero = (c_after != 0).sum().item()
        print(f"  fc3 counter nonzero: {nonzero}/{c_after.numel()}")
    else:
        print(f"  fc3 counter still [0,0] — update() kernel does NOT modify counter when called directly")
    print()

    # Baseline: no training
    print("Computing baseline (frozen random weights)...")
    bl = baseline_loss(d_in, d_hidden, d_out, batch_size)
    print(f"  Baseline loss (no train): {bl:.6f}")
    print()

    # Discrete optimizer
    print("Training discrete MLP (counter-based optimizer, threshold=8)...")
    discrete_losses = train_discrete(d_in, d_hidden, d_out, threshold=8, steps=steps)
    final_discrete = discrete_losses[-1]
    improved = final_discrete < bl
    print(f"  Final discrete loss: {final_discrete:.6f} (baseline: {bl:.6f})")
    print(f"  Converged: {'YES ✅' if improved else 'NO ❌'}")
    print()

    # Float baseline
    print("Training float MLP (AdamW, lr=1e-3)...")
    float_losses = train_float(d_in, d_hidden, d_out, lr=1e-3, steps=steps)
    final_float = float_losses[-1]
    print(f"  Final float loss: {final_float:.6f} (baseline: {bl:.6f})")
    print()

    # Summary
    print("═" * 50)
    print("Summary")
    print("═" * 50)
    print(f"  Baseline (no training):        {bl:.6f}")
    print(f"  Discrete (counter-based):      {final_discrete:.6f}")
    print(f"  Float (AdamW):                 {final_float:.6f}")
    if improved:
        redux = (bl - final_discrete) / bl * 100
        print(f"  Discrete loss reduction:       {redux:.1f}%")
    else:
        print(f"  Discrete did NOT converge")
