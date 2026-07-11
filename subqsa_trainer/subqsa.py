"""SubQSA — NSA-style 3-branch sparse attention. Clean, verified shapes."""

import importlib.util
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# Load the 1bit_trainer RotaryEmbedding implementation so that both trainer tiers
# share the exact same RoPE code without duplicating it.
def _load_1bit_rope():
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "_bit_model", os.path.join(_root, "1bit_trainer", "model.py")
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

        # RoPE — unified with 1bit_trainer implementation
        self.rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len, theta=rope_theta)

        # Gate: one scalar per head per position
        self.gate_fc = nn.Linear(hidden_dim, 3 * num_heads, bias=False)

        # Learned compression: MLP-based block pooling (instead of mean-pool)
        self._cmp_phi_k = nn.Sequential(
            nn.Linear(head_dim * cmp_block, head_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(head_dim * 2, head_dim, bias=False),
        )
        self._cmp_phi_v = nn.Sequential(
            nn.Linear(head_dim * cmp_block, head_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(head_dim * 2, head_dim, bias=False),
        )

    def _compress(self, k, v, B, H, T):
        l, d = self.cmp_block, self.cmp_stride
        n = max(1, (T - l) // d)
        if n < 1:
            return torch.zeros(
                B, H, 1, self.head_dim, device=k.device, dtype=k.dtype
            ), torch.zeros(B, H, 1, self.head_dim, device=k.device, dtype=k.dtype)
        # Use learned MLP compressors instead of mean-pool
        # Stack blocks for MLP processing
        k_blocks = torch.stack([k[:, :, i * d : i * d + l] for i in range(n)], dim=2)
        v_blocks = torch.stack([v[:, :, i * d : i * d + l] for i in range(n)], dim=2)
        # Flatten block tokens for MLP input
        k_flat = k_blocks.reshape(B, H, n, l * self.head_dim)
        v_flat = v_blocks.reshape(B, H, n, l * self.head_dim)
        return self._cmp_phi_k(k_flat), self._cmp_phi_v(v_flat)

    def _score_and_select(self, q, k, v, raw_scores_agg, n_sel):
        """Select top-k blocks via raw attention scores (pre-softmax).

        Uses raw scores (not softmax probabilities) so the absolute
        importance signal is preserved. Max-pool across query positions
        captures the most important query per block.

        Args:
            q: (B, H, T, D) — queries (unused, kept for interface compat).
            k: (B, H, T, D) — full RoPE'd GQA-expanded keys.
            v: (B, H, T, D) — full GQA-expanded values.
            raw_scores_agg: (B, H, n_sel) — per-block raw scores aggregated
                across query positions, already resampled to selection grid.
            n_sel: int — number of selection grid blocks.

        Returns:
            k_sel: (B, H, k_actual * l_prime, D) — selected key blocks flattened.
            v_sel: (B, H, k_actual * l_prime, D) — selected value blocks flattened.
            top_idx: (B, H, k_actual) — indices of selected blocks.
        """
        B, H, T, D = k.shape
        l_prime = self.slc_block

        k_blocks = k[:, :, : n_sel * l_prime].reshape(B, H, n_sel, l_prime, D)
        v_blocks = v[:, :, : n_sel * l_prime].reshape(B, H, n_sel, l_prime, D)

        k_actual = min(self.slc_topk, n_sel)
        _, top_idx = raw_scores_agg.topk(k_actual, dim=-1)

        bi = top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, l_prime, D)
        k_sel = torch.gather(k_blocks, dim=2, index=bi)
        v_sel = torch.gather(v_blocks, dim=2, index=bi)
        k_sel = k_sel.reshape(B, H, k_actual * l_prime, D)
        v_sel = v_sel.reshape(B, H, k_actual * l_prime, D)

        return k_sel, v_sel, top_idx

    def forward(self, x, start_pos=0, seq_len=None):
        B, T, _ = x.shape
        if seq_len is None:
            seq_len = T

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE — unified 1bit_trainer implementation
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

        # Compression attention SCORES (pre-softmax) as selection signal.
        # Using raw scores preserves absolute importance — softmax normalizes
        # across blocks and loses the relative ranking signal.
        raw_scores = (q @ k_cmp.transpose(-2, -1)) * (self.head_dim**-0.5)
        p_cmp = F.softmax(raw_scores, dim=-1)  # for compressed attention output
        # Aggregate raw scores: max-pool across query positions captures peak importance
        raw_scores_agg = raw_scores.max(dim=2).values  # (B, H, n_cmp)

        # Resample raw_scores_agg to selection grid if shapes differ.
        n_cmp = k_cmp.shape[2]
        n_sel = max(1, T // self.slc_block)
        if n_cmp != n_sel:
            if n_cmp > n_sel:
                stride = n_cmp // n_sel
                raw_scores_agg = (
                    raw_scores_agg[..., : stride * n_sel]
                    .reshape(B, self.num_heads, n_sel, stride)
                    .max(dim=-1).values
                )
            else:
                repeats = n_sel - n_cmp
                raw_scores_agg = torch.cat(
                    [raw_scores_agg, raw_scores_agg[..., -1:].expand(-1, -1, repeats)], dim=-1
                )

        # Selection branch (top-k from raw compression scores)
        k_sel, v_sel, _ = self._score_and_select(q, k_r, v, raw_scores_agg, n_sel)

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

        # Gate blending
        g = self.gate_fc(x).view(B, T, 3, self.num_heads).permute(0, 3, 1, 2).sigmoid()
        g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)
        out = g[..., 0:1] * cmp_out + g[..., 1:2] * slc_out + g[..., 2:3] * win_out
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)
