"""SubQSA for Ultimate Trainer with BitLinear projections + subln."""

import importlib.util
import math
import os
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


from ultimate_trainer.bitlinear import BitLinear, RMSNorm


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
            return k.mean(dim=2, keepdim=True), v.mean(dim=2, keepdim=True)
        blocks_k = k.unfold(2, l, d)[:, :, :n_blocks].transpose(-1, -2).reshape(B, H, n_blocks, l * D)
        blocks_v = v.unfold(2, l, d)[:, :, :n_blocks].transpose(-1, -2).reshape(B, H, n_blocks, l * D)
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
        if p_cmp.shape[-1] != n_sel:
            n_c = p_cmp.shape[-1]
            if n_c > n_sel:
                stride = n_c // n_sel
                p_cmp = p_cmp[..., :stride * n_sel].reshape(B, H, T, n_sel, stride).sum(dim=-1)
            else:
                repeats = n_sel - n_c
                p_cmp = torch.cat([p_cmp, p_cmp[..., -1:].expand(-1, -1, -1, repeats)], dim=-1)
            p_cmp = p_cmp / p_cmp.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        topk_actual = min(n, n_sel)
        _, top_idx = p_cmp.topk(topk_actual, dim=-1)  # (B, H, T, topk_actual)

        # Gather full l_prime-token blocks from the first n_sel * l_prime positions.
        k_sliced = k[:, :, : n_sel * lp, :]
        v_sliced = v[:, :, : n_sel * lp, :]
        k_blocks = k_sliced.reshape(B, H, n_sel, lp, D)
        v_blocks = v_sliced.reshape(B, H, n_sel, lp, D)

        b_idx = torch.arange(B, device=q.device).view(B, 1, 1, 1)
        h_idx = torch.arange(H, device=q.device).view(1, H, 1, 1)
        k_sel = k_blocks[b_idx, h_idx, top_idx]  # (B, H, T, topk_actual, lp, D)
        v_sel = v_blocks[b_idx, h_idx, top_idx]

        # Concatenate selected blocks along the key/value length dimension.
        k_sel = k_sel.reshape(B, H, T, topk_actual * lp, D)
        v_sel = v_sel.reshape(B, H, T, topk_actual * lp, D)

        L = k_sel.shape[-2]  # topk_actual * lp
        q_flat = q.reshape(B * H * T, 1, D)
        k_flat = k_sel.reshape(B * H * T, L, D)
        v_flat = v_sel.reshape(B * H * T, L, D)
        out = F.scaled_dot_product_attention(q_flat, k_flat, v_flat)
        return out.reshape(B, H, T, D).to(q.dtype), top_idx


def sliding_window_attention(q, k, v, win_size, cache=None):
    """Causal sliding-window attention over the last ``win_size`` tokens.

    Slices ``k`` and ``v`` to their trailing window, then applies
    ``F.scaled_dot_product_attention`` with a ``(T, w)`` additive mask so
    query position ``t`` only attends to key positions
    ``[max(0, t - w + 1), ..., t]`` within the sliced window.
    Positions with no valid causal keys inside the window are zeroed out.

    When ``cache`` is a dict it is indexed by ``(T, w)``; if the mask has
    already been built for a given ``(T, w)`` pair it is reused instead of
    recomputed.
    """
    B, H, T, D = q.shape
    w = min(win_size, T)
    k_win = k[..., -w:, :]
    v_win = v[..., -w:, :]

    # Use or build a (T, w) additive causal mask.
    if cache is not None and (T, w) in cache:
        attn_mask = cache[(T, w)]
        valid = None
    else:
        t_idx = torch.arange(T, device=q.device).unsqueeze(1)
        j_idx = torch.arange(w, device=q.device).unsqueeze(0)
        valid = (T - w + j_idx) <= t_idx  # (T, w)
        attn_mask = torch.where(
            valid,
            torch.zeros((), device=q.device, dtype=q.dtype),
            torch.full((), -1e9, device=q.device, dtype=q.dtype),
        )
        if cache is not None:
            cache[(T, w)] = attn_mask

    out = F.scaled_dot_product_attention(q, k_win, v_win, attn_mask=attn_mask)

    if valid is None:
        # Recompute validity to zero queries with no causal keys inside window.
        t_idx = torch.arange(T, device=q.device).unsqueeze(1)
        j_idx = torch.arange(w, device=q.device).unsqueeze(0)
        valid = (T - w + j_idx) <= t_idx
    # Zero queries that have no causal keys inside the window.
    has_valid = valid.any(dim=-1).view(1, 1, T, 1)
    out = torch.where(has_valid, out, torch.zeros_like(out))
    return out


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

        # RoPE — unified with 1bit-trainer implementation
        self.rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len, theta=rope_theta)

        self.compression = CompressionBranch(head_dim, cmp_block, cmp_stride)
        self.selection = SelectionBranch(slc_block, slc_topk)
        self.win_size = win_size
        self._sw_mask_cache = {}

        # Per-head gating MLP (tiny — stays FP16)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64, bias=False),
            nn.SiLU(),
            nn.Linear(64, 3 * num_heads, bias=False),
        )

    def forward(self, x, position_ids, attention_mask=None):
        B, T, _ = x.shape

        # Project with BitLinear (ternary weights, INT8 activations)
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE — unified 1bit-trainer implementation
        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        # ── Compression branch ──
        # Compress at num_kv_heads resolution (GQA-aware); expand compressed KV
        # only when attending with query heads.
        k_cmp, v_cmp = self.compression(k, v)

        def repeat_kv(t, n_rep):
            B_, H_, T_, D_ = t.shape
            return (
                t[:, :, None]
                .expand(-1, -1, n_rep, -1, -1)
                .reshape(B_, H_ * n_rep, T_, D_)
            )

        n_reps = self.num_heads // self.num_kv_heads if self.num_kv_heads else 1
        if k_cmp.shape[2] > 0 and n_reps > 1:
            # GQA-aware compression attention: process each Q-head group with its
            # KV head directly, avoiding 4× materialization of k_cmp/v_cmp.
            # q: (B, 20, T, D), k_cmp: (B, 5, n_cmp, D)
            # Split 20 Q heads into n_reps=4 groups of 5, each attending to the
            # corresponding KV head's compressed representation.
            o_cmp_parts = []
            p_cmp_parts = []
            n_kv = self.num_kv_heads
            for gi in range(n_reps):
                # Every n_reps-th query head maps to KV head j:
                # group 0: heads [0, 4,  8, 12, 16]; group 1: heads [1, 5,  9, 13, 17]; etc.
                q_g = q[:, gi::n_reps, :, :]  # (B, 5, T, D)
                scores_g = torch.einsum(
                    "bhtd,bhld->bhtl", q_g.float(), k_cmp.float()
                ) / math.sqrt(self.head_dim)
                p_cmp_g = F.softmax(scores_g, dim=-1)
                p_cmp_parts.append(p_cmp_g)
                # SDPA at KV-head resolution
                o_g = F.scaled_dot_product_attention(
                    q_g, k_cmp, v_cmp,
                    dropout_p=self.dropout if self.training else 0.0
                )
                o_cmp_parts.append(o_g)
            # Interleave groups back to original head order
            n_cmp = k_cmp.shape[2]
            o_cmp = torch.stack(o_cmp_parts, dim=2).reshape(B, self.num_heads, T, self.head_dim)
            p_cmp = torch.stack(p_cmp_parts, dim=2).reshape(B, self.num_heads, T, n_cmp)
        elif k_cmp.shape[2] > 0:
            # No GQA or n_reps==1 — full attention at query-head resolution
            scores_cmp = torch.einsum(
                "bhtd,bhld->bhtl", q.float(), k_cmp.float()
            ) / math.sqrt(self.head_dim)
            p_cmp = F.softmax(scores_cmp, dim=-1)
            o_cmp = F.scaled_dot_product_attention(
                q, k_cmp, v_cmp, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            p_cmp = torch.zeros(B, self.num_heads, T, 1, device=x.device)
            o_cmp = torch.zeros_like(q)

        n_cmp = k_cmp.shape[2]
        # Expand full KV to num_heads for selection and sliding-window branches.
        k_h = repeat_kv(k, n_reps) if n_reps > 1 else k
        v_h = repeat_kv(v, n_reps) if n_reps > 1 else v
        o_slc, _ = self.selection(q, k_h, v_h, p_cmp, n_cmp)
        o_win = sliding_window_attention(q, k_h, v_h, self.win_size, self._sw_mask_cache)

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
