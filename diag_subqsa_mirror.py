"""CPU mirror of subqsa_combine_kernel — find where it diverges from eager."""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, T, H, D_head = 2, 64, 2, 32
D = H * D_head

x = torch.randn(B, T, D)
o_cmp = torch.randn(B, H, T, D_head)
o_slc = torch.randn(B, H, T, D_head)
o_win = torch.randn(B, H, T, D_head)
gate_w1 = torch.randn(64, D) * 0.1
gate_w2 = torch.randn(3 * H, 64) * 0.1
out_norm_weight = torch.randn(D)
o_proj_weight = torch.randn(D, D) * 0.05
gamma = 0.1

# ---------- EAGER ----------
g = F.linear(x, gate_w1)
g = F.silu(g)
g = F.linear(g, gate_w2).view(B, T, 3, H).permute(0, 3, 1, 2)
g = g.sigmoid()
g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)

o = (g[..., 0:1] * o_cmp + g[..., 1:2] * o_slc + g[..., 2:3] * o_win).to(dtype=x.dtype)
o = o.transpose(1, 2).reshape(B, T, -1)

rms = o.pow(2).mean(-1, keepdim=True).sqrt()
o = o / (rms + 1e-5) * out_norm_weight

w_q = torch.clamp(torch.round(o_proj_weight / gamma), -1, 1) * gamma
y_eager = F.linear(o.float(), w_q)

# ---------- KERNEL MIRROR ----------
# Phase 1-3: gate MLP, per-token
y_kern = torch.zeros(B, T, D)
for b in range(B):
    for t in range(T):
        # phase 2: s_hidden
        s_hidden = torch.zeros(64)
        for i in range(64):
            s = sum(gate_w1[i, j] * x[b, t, j] for j in range(D))
            s_hidden[i] = s * (1.0 / (1.0 + torch.exp(-s)))
        # phase 3: s_gate
        s_gate = torch.zeros(3 * H)
        for i in range(3 * H):
            s_gate[i] = sum(gate_w2[i, j] * s_hidden[j] for j in range(64))
        # phase 4: sigmoid + L1 norm
        for h in range(H):
            g0 = torch.sigmoid(s_gate[h * 3 + 0])
            g1 = torch.sigmoid(s_gate[h * 3 + 1])
            g2 = torch.sigmoid(s_gate[h * 3 + 2])
            ssum = g0 + g1 + g2 + 1e-8
            s_gate[h * 3 + 0] = g0 / ssum
            s_gate[h * 3 + 1] = g1 / ssum
            s_gate[h * 3 + 2] = g2 / ssum
        # phase 5: blend
        s_blended = torch.zeros(D)
        for i in range(D):
            h = i // D_head
            dh = i % D_head
            v_cmp = o_cmp[b, h, t, dh]
            v_slc = o_slc[b, h, t, dh]
            v_win = o_win[b, h, t, dh]
            s_blended[i] = s_gate[h * 3 + 0] * v_cmp + s_gate[h * 3 + 1] * v_slc + s_gate[h * 3 + 2] * v_win
        # phase 6-7: RMSNorm (eager-style: /(rms+1e-5))
        rms = torch.sqrt((s_blended ** 2).mean())
        s_blended = s_blended / (rms + 1e-5) * out_norm_weight
        # phase 8: O proj
        for oi in range(D):
            s = 0.0
            for ii in range(D):
                w = o_proj_weight[oi, ii]
                wq = torch.clamp(torch.round(w / gamma), -1, 1) * gamma
                s += s_blended[ii] * wq
            y_kern[b, t, oi] = s

diff = (y_kern - y_eager).abs()
print(f"max diff = {diff.max().item():.6f}")
idx = diff.argmax().item()
b = idx // (T * D); rem = idx % (T * D); t = rem // D; oi = rem % D
print(f"max at b={b} t={t} out={oi}: kern={y_kern[b,t,oi].item():.6f} eager={y_eager[b,t,oi].item():.6f}")

# per-phase diff
g_eager = g  # (B,H,T,3)
blend_eager = (g_eager[..., 0:1] * o_cmp + g_eager[..., 1:2] * o_slc + g_eager[..., 2:3] * o_win).transpose(1,2).reshape(B,T,-1)
blend_kern = torch.zeros(B,T,D)
for b in range(B):
    for t in range(T):
        for i in range(D):
            h = i // D_head; dh = i % D_head
            g0 = g_eager[b, h, t, 0]; g1 = g_eager[b, h, t, 1]; g2 = g_eager[b, h, t, 2]
            blend_kern[b,t,i] = g0 * o_cmp[b,h,t,dh] + g1 * o_slc[b,h,t,dh] + g2 * o_win[b,h,t,dh]
bdiff = (blend_kern - blend_eager).abs().max().item()
print(f"blend diff = {bdiff:.6f}")

# gate diff
print(f"gate eager shape {g_eager.shape}")
