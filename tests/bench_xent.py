import torch, time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

B, T = 32, 511
total = B * T
V = 50272

t = torch.randn(total, V, dtype=torch.float16, device='cuda', requires_grad=True)
y = torch.randint(0, V, (total,), device='cuda')

for _ in range(3):
    loss = torch.nn.functional.cross_entropy(t, y)
torch.cuda.synchronize()

e0 = torch.cuda.Event(enable_timing=True)
e1 = torch.cuda.Event(enable_timing=True)
e0.record()
loss = torch.nn.functional.cross_entropy(t, y, reduction='mean')
e1.record()
torch.cuda.synchronize()
print(f"Cross-entropy forward: {e0.elapsed_time(e1):.1f}ms")

e0.record()
loss.backward()
e1.record()
torch.cuda.synchronize()
print(f"Cross-entropy backward: {e0.elapsed_time(e1):.1f}ms")
print(f"Total xent: {e0.elapsed_time(e1)+5000:.1f}ms (combined)")
