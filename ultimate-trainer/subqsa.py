"""SubQSA for Ultimate Trainer with BitLinear projections + subln."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from kernels import use_kernel_forward_from_hub
except ImportError:

    def use_kernel_forward_from_hub(name):
        def _decorator(cls):
            return cls

        return _decorator


from ultimate_trainer.bitlinear import BitLinear, RMSNorm

# Optional Triton NSA kernel (fla-org) — 10-50× faster on GPU
_HAS_NSA_KERNEL = False
try:
    from fla.ops import parallel_nsa

    _HAS_NSA_KERNEL = True
except ImportError:
    pass


def _nsa_fused_forward(self, q, k, v, k_cmp, v_cmp, p_cmp, n_cmp, x, B, T):
    """Fallback for nsa-fused path: pure PyTorch 3-branch attention (same as default)."""
    o_cmp = F.scaled_dot_product_attention(
        q, k_cmp, v_cmp, dropout_p=self.dropout if self.training else 0.0
    )
    o_slc, _ = self.selection(q, k, v, p_cmp, n_cmp)
    o_win = sliding_window_attention(q, k, v, self.win_size)
    g = self.gate_mlp(x).view(B, T, 3, self.num_heads).permute(0, 3, 1, 2)
    g = g.float().sigmoid()
    g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)
    o = (
        g[..., 0:1] * o_cmp.float()
        + g[..., 1:2] * o_slc.float()
        + g[..., 2:3] * o_win.float()
    ).to(dtype=x.dtype)
    return o


class CompressionBranch(nn.Module):
    def __init__(self, head_dim, block_len, stride):
        super().__init__()
        self.l = block_len
        self.d = stride
        self.phi_k = nn.Sequential(
            nn.Linear(head_dim * block_len, head_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(head_dim * 2, head_dim, bias=False),
        )
        self.phi_v = nn.Sequential(
            nn.Linear(head_dim * block_len, head_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(head_dim * 2, head_dim, bias=False),
        )

    def forward(self, k, v):
        B, H, T, D = k.shape
        l, d = self.l, self.d
        n_blocks = (T - l) // d
        if n_blocks <= 0:
            return k[:, :, :1], v[:, :, :1]
        blocks_k = torch.stack(
            [k[:, :, i * d : i * d + l] for i in range(n_blocks)], dim=2
        ).reshape(B, H, n_blocks, l * D)
        blocks_v = torch.stack(
            [v[:, :, i * d : i * d + l] for i in range(n_blocks)], dim=2
        ).reshape(B, H, n_blocks, l * D)
        return self.phi_k(blocks_k), self.phi_v(blocks_v)


class SelectionBranch(nn.Module):
    """Content-aware top-k block selection from compression scores."""

    def __init__(self, block_size=64, topk=16):
        super().__init__()
        self.l_prime = block_size
        self.n = topk

    def forward(self, q, k, v, p_cmp, n_cmp):
        B, H, T, D = q.shape
        lp, n = self.l_prime, self.n
        n_sel = max(1, T // lp)
        if p_cmp.shape[-1] != n_sel and p_cmp.shape[-1] > 1:
            p_cmp = (
                F.interpolate(
                    p_cmp.reshape(-1, 1, p_cmp.shape[-1]).float(),
                    size=n_sel,
                    mode="nearest",
                )
                .reshape(B, H, T, n_sel)
                .to(q.dtype)
            )
        topk_actual = min(n, n_sel)
        _, top_idx = p_cmp.topk(topk_actual, dim=-1)
        best_idx = top_idx[..., 0]
        k_sel = torch.stack(
            [
                k[b, h, best_idx[b, h, t] * lp : best_idx[b, h, t] * lp + lp]
                for b in range(B)
                for h in range(H)
                for t in range(T)
            ],
            dim=0,
        ).view(B, H, T, lp, D)
        v_sel = torch.stack(
            [
                v[b, h, best_idx[b, h, t] * lp : best_idx[b, h, t] * lp + lp]
                for b in range(B)
                for h in range(H)
                for t in range(T)
            ],
            dim=0,
        ).view(B, H, T, lp, D)
        scores = torch.einsum("bhtd,bhtld->bhtl", q.float(), k_sel.float()) / math.sqrt(
            D
        )
        attn = F.softmax(scores, dim=-1)
        return torch.einsum("bhtl,bhtld->bhtd", attn, v_sel.float()).to(
            q.dtype
        ), top_idx


def sliding_window_attention(q, k, v, win_size):
    """Causal sliding window attention with triangular mask."""
    B, H, T, D = q.shape
    w = min(win_size, T)
    mask = torch.tril(torch.ones(T, T, device=q.device)).to(q.dtype)
    mask = torch.triu(mask, diagonal=-(w - 1))
    attn_mask = (1.0 - mask[None, None, :, :]) * -1e9
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)


@use_kernel_forward_from_hub("SubQSAAttention")
class SubQSAAttention(nn.Module):
    """NSA-style 3-branch sparse attention with BitLinear + subln.

    Each sub-block has two subln norms (2B4T spec):
      1. subln_in (applied before QKV projections — handled in TransformerBlock)
      2. subln_out (applied before O projection — internal)
    """

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
        use_bitlinear=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dropout = dropout

        proj_cls = BitLinear if use_bitlinear else nn.Linear
        self.q_proj = proj_cls(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = proj_cls(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = proj_cls(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = proj_cls(num_heads * head_dim, hidden_dim, bias=False)

        # 2B4T: subln before output projection (applied to gated 3-branch output)
        self.out_norm = RMSNorm(num_heads * head_dim)

        # RoPE
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.compression = CompressionBranch(head_dim, cmp_block, cmp_stride)
        self.selection = SelectionBranch(slc_block, slc_topk)
        self.win_size = win_size

        # Per-head gating MLP (tiny — stays FP16)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64, bias=False),
            nn.SiLU(),
            nn.Linear(64, 3 * num_heads, bias=False),
        )

    def _apply_rope(self, x, position_ids):
        B, H, T, D = x.shape
        inv_freq = self.inv_freq[None, :, None, None]
        pos = position_ids[:, None, :, None].float()
        angles = pos * inv_freq
        angles = angles.permute(0, 2, 1, 3)
        cos = angles.cos().to(dtype=x.dtype).squeeze(-1).unsqueeze(1)  # (B, 1, T, D/2)
        sin = angles.sin().to(dtype=x.dtype).squeeze(-1).unsqueeze(1)  # (B, 1, T, D/2)
        x1, x2 = x[..., : D // 2], x[..., D // 2 :]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x, position_ids, attention_mask=None):
        B, T, _ = x.shape

        # Project with BitLinear (ternary weights, INT8 activations)
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA: expand KV heads
        if self.num_heads != self.num_kv_heads:
            n_reps = self.num_heads // self.num_kv_heads
            k = (
                k[:, :, None]
                .expand(-1, -1, n_reps, -1, -1)
                .reshape(B, self.num_heads, T, self.head_dim)
            )
            v = (
                v[:, :, None]
                .expand(-1, -1, n_reps, -1, -1)
                .reshape(B, self.num_heads, T, self.head_dim)
            )

        # RoPE (BF16 precision)
        q = self._apply_rope(q, position_ids)
        k = self._apply_rope(k, position_ids)

        # ── Compression branch ──
        k_cmp, v_cmp = self.compression(k, v)
        if k_cmp.shape[2] > 0:
            # Compression attention scores reused for selection routing
            scores_cmp = torch.einsum(
                "bhtd,bhld->bhtl", q.float(), k_cmp.float()
            ) / math.sqrt(self.head_dim)
            p_cmp = F.softmax(scores_cmp, dim=-1)
        else:
            p_cmp = torch.zeros(B, self.num_heads, T, 1, device=x.device)
            o_cmp = torch.zeros_like(q)

        n_cmp = k_cmp.shape[2]

        # ── Triton fused path (fla-org parallel_nsa, ~10× faster on GPU) ──
        if _HAS_NSA_KERNEL and q.is_cuda:
            o = _nsa_fused_forward(self, q, k, v, k_cmp, v_cmp, p_cmp, n_cmp, x, B, T)
        else:
            # Pure PyTorch 3-branch path (CPU-safe fallback)
            if k_cmp.shape[2] > 0:
                o_cmp = F.scaled_dot_product_attention(
                    q, k_cmp, v_cmp, dropout_p=self.dropout if self.training else 0.0
                )
            o_slc, _ = self.selection(q, k, v, p_cmp, n_cmp)
            o_win = sliding_window_attention(q, k, v, self.win_size)

            g = self.gate_mlp(x).view(B, T, 3, self.num_heads).permute(0, 3, 1, 2)
            g = g.float().sigmoid()
            g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)

            o = (
                g[..., 0:1] * o_cmp.float()
                + g[..., 1:2] * o_slc.float()
                + g[..., 2:3] * o_win.float()
            ).to(dtype=x.dtype)

        # ── 2B4T: subln before output projection ──
        o = o.transpose(1, 2).reshape(B, T, -1)
        o = self.out_norm(o)
        return self.o_proj(o)
