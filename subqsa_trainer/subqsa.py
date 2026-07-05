"""SubQSA — NSA-style 3-branch sparse attention. Clean, verified shapes."""

import importlib.util
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# Load the 1bit-trainer RotaryEmbedding implementation so that both trainer tiers
# share the exact same RoPE code without duplicating it.
def _load_1bit_rope():
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "_bit_model", os.path.join(_root, "1bit-trainer", "model.py")
    )
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    return _mod.RotaryEmbedding


RotaryEmbedding = _load_1bit_rope()

try:
    from kernels import use_kernel_forward_from_hub
except ImportError:

    def use_kernel_forward_from_hub(name):
        def _decorator(cls):
            return cls

        return _decorator


class SubQSA(nn.Module):
    """NSA-inspired sparse attention with verified shape broadcasting."""

    def __init__(
        self,
        hidden_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        max_seq_len=4096,
        rope_theta=10000.0,
        dropout=0.0,
        cmp_block=32,
        cmp_stride=16,
        slc_block=64,
        slc_topk=16,
        win_size=512,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.cmp_block = cmp_block
        self.cmp_stride = cmp_stride
        self.slc_block = slc_block
        self.slc_topk = slc_topk
        self.win_size = win_size

        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        # RoPE — unified with 1bit-trainer implementation
        self.rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len, theta=rope_theta)

        # Gate: one scalar per head per position
        self.gate_fc = nn.Linear(hidden_dim, 3 * num_heads, bias=False)

    def _compress(self, k, v, B, H, T):
        l, d = self.cmp_block, self.cmp_stride
        n = max(1, (T - l) // d)
        if n < 1:
            return torch.zeros(
                B, H, 1, self.head_dim, device=k.device, dtype=k.dtype
            ), torch.zeros(B, H, 1, self.head_dim, device=k.device, dtype=k.dtype)
        k_b = torch.stack([k[:, :, i * d : i * d + l] for i in range(n)], dim=2).mean(
            2
        )  # (B, H, n, D)
        v_b = torch.stack([v[:, :, i * d : i * d + l] for i in range(n)], dim=2).mean(2)
        return k_b, v_b

    def _score_and_select(self, q, k_cmp, v_cmp, B, H, T):
        # q: (B, H, T, D), k_cmp: (B, H, n, D)
        k_mag = k_cmp.abs().mean(dim=-1)  # (B, H, n)
        _, top_idx = k_mag.topk(min(self.slc_topk, k_cmp.shape[2]), dim=-1)  # (B, H, k)
        # gather
        ki = top_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        vi = top_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        k_sel = torch.gather(k_cmp, dim=2, index=ki)
        v_sel = torch.gather(v_cmp, dim=2, index=vi)
        return k_sel, v_sel, top_idx

    def forward(self, x, start_pos=0, seq_len=None):
        B, T, _ = x.shape
        if seq_len is None:
            seq_len = T

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE — unified 1bit-trainer implementation
        position_ids = (
            torch.arange(start_pos, start_pos + T, device=x.device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        q = self.rope(q, position_ids)
        k_r = self.rope(k, position_ids)

        # GQA expand
        if self.num_heads != self.num_kv_heads:
            reps = self.num_heads // self.num_kv_heads
            k_r = k_r.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)

        # Compression branch
        k_cmp, v_cmp = self._compress(k_r, v, B, self.num_heads, T)

        # Selection branch (top-k magnitude on compressed KV)
        k_sel, v_sel, _ = self._score_and_select(q, k_cmp, v_cmp, B, self.num_heads, T)

        # All 3 branches use standard SDPA (shapes verified to match)
        # 1) Compression: q (B,H,T,D) x k_cmp (B,H,n,D) -> (B,H,T,n) -> (B,H,T,D)
        cmp_out = F.scaled_dot_product_attention(q, k_cmp, v_cmp, dropout_p=0.0)

        # 2) Selection: q (B,H,T,D) x k_sel (B,H,k,D) -> (B,H,T,k) -> (B,H,T,D)
        slc_out = F.scaled_dot_product_attention(q, k_sel, v_sel, dropout_p=0.0)

        # 3) Sliding window: last win_size tokens attend locally
        win = min(self.win_size, T)
        q_win = q[:, :, -win:, :]
        k_win = k_r[:, :, -win:, :]
        v_win = v[:, :, -win:, :]
        win_out = F.scaled_dot_product_attention(q_win, k_win, v_win, dropout_p=0.0)
        # Pad to T
        if win < T:
            pad = torch.zeros(
                B,
                self.num_heads,
                T - win,
                self.head_dim,
                device=q.device,
                dtype=q.dtype,
            )
            win_out = torch.cat([pad, win_out], dim=2)
        win_out = win_out + q  # residual

        # Gate blending
        g = self.gate_fc(x).view(B, T, 3, self.num_heads).permute(0, 3, 1, 2).sigmoid()
        g = g / g.sum(dim=-1, keepdim=True)
        out = g[..., 0:1] * cmp_out + g[..., 1:2] * slc_out + g[..., 2:3] * win_out
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)
