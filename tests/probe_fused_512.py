"""GPU probe: fused kernel vs half-input eager at full config (D_out=512)."""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, T, H, D = 8, 512, 8, 64
D_out = H * D

dev = "cuda"
x = torch.randn(B, T, D_out, device=dev)
o_cmp = torch.randn(B, H, T, D, device=dev)
o_slc = torch.randn(B, H, T, D, device=dev)
o_win = torch.randn(B, H, T, D, device=dev)
gate_w1 = torch.randn(64, D_out, device=dev) * 0.1
gate_w2 = torch.randn(3 * H, 64, device=dev) * 0.1
out_norm_weight = torch.randn(D_out, device=dev)
o_proj_weight = torch.randn(D_out, D_out, device=dev) * 0.05
gamma = 0.1

from kernels.subqsa_combine.subqsa_combine import SubQSACombineFn, _HAS_SUBQSA_COMBINE
print(f"_HAS_SUBQSA_COMBINE={_HAS_SUBQSA_COMBINE}", flush=True)
y_kern = SubQSACombineFn.apply(x, o_cmp, o_slc, o_win, gate_w1, gate_w2,
                               out_norm_weight, o_proj_weight, gamma, None)
torch.cuda.synchronize()

# half-input eager (mirror of wrapper conversion)
xh = x.half().float(); oc = o_cmp.half().float(); os = o_slc.half().float(); ow = o_win.half().float()
g1 = gate_w1.half().float(); g2 = gate_w2.half().float(); onw = out_norm_weight.half().float()
g = F.linear(xh, g1)
g = F.silu(g)
g = F.linear(g, g2).view(B, T, 3, H).permute(0, 3, 1, 2).sigmoid()
g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)
o = (g[..., 0:1] * oc + g[..., 1:2] * os + g[..., 2:3] * ow).transpose(1, 2).reshape(B, T, -1)
rms = o.pow(2).mean(-1, keepdim=True).sqrt()
o = o / (rms + 1e-5) * onw
w_q = torch.clamp(torch.round(o_proj_weight / gamma), -1, 1) * gamma
y_ref = F.linear(o.float(), w_q)

d = (y_kern - y_ref).abs()
print(f"kernel vs half-eager: max={d.max().item():.5f}  #>0.1={(d>0.1).sum().item()}", flush=True)
idx = d.argmax().item()
bb = idx // (T * D_out); rr = (idx % (T * D_out)) // D_out; cc = idx % D_out
print(f"max at b={bb} t={rr} o={cc}: kern={y_kern.flatten()[idx].item():.5f} ref={y_ref.flatten()[idx].item():.5f}", flush=True)

# check a row's blend before O-proj: reconstruct kernel blend (half inputs, eager rms) and compare gates
# gate values from eager
print(f"gates[0,0,0,:] = {g[0,0,0,:].tolist()}", flush=True)
print(f"gates[0,0,1,:] = {g[0,0,1,:].tolist()}", flush=True)
print(f"gates[0,0,2,:] = {g[0,0,2,:].tolist()}", flush=True)

# Reconstruct kernel blend (half inputs, eager gates) and compare with kernel output
# via O-proj: if blend correct but final wrong -> phase 8 bug; else blend bug.
blend = (g[..., 0:1] * oc + g[..., 1:2] * os + g[..., 2:3] * ow).transpose(1, 2).reshape(B, T, -1)
# kernel blend should equal this. We can't read kernel shared mem, but we CAN
# test phase 8 in isolation: feed a known blend through the kernel? Not directly.
# Instead compare: y_ref = F.linear(normalized blend, w_q). If kernel's O-proj
# used a DIFFERENT blend, the diff pattern would be column-structured.
# Check column structure: mean |diff| per output column
md = d.mean(dim=(0, 1))  # (D_out,)
print(f"mean|diff| per column: min={md.min().item():.5f} max={md.max().item():.5f}", flush=True)
print(f"top columns: {md.topk(5).indices.tolist()}", flush=True)
print(f"bottom columns: {md.topk(5, largest=False).indices.tolist()}", flush=True)
# check per-token structure
mt = d.mean(dim=(0, 2))
print(f"mean|diff| per t: min={mt.min().item():.5f} max={mt.max().item():.5f}", flush=True)
# check head structure: group output cols by head
for h in range(H):
    seg = d[:, :, h * D:(h + 1) * D]
    print(f"head {h}: mean|diff|={seg.mean().item():.5f} max={seg.max().item():.5f}", flush=True)

# per-token correlation: print diff and blend norm for a few bad tokens
mt2 = d.mean(dim=2)  # (B,T)
for b in range(B):
    bad = mt2[b].topk(3).indices.tolist()
    good = mt2[b].topk(3, largest=False).indices.tolist()
    for t in bad + good:
        # blend row norm (after rms) and gate row
        seg = d[b, t].mean().item()
        print(f"  b={b} t={t}: mean|d|={seg:.5f} kern[0]={y_kern[b,t,0].item():.4f} ref[0]={y_ref[b,t,0].item():.4f}", flush=True)
