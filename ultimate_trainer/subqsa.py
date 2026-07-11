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
        blocks_k = (
            k.unfold(2, l, d)[:, :, :n_blocks]
            .transpose(-1, -2)
            .reshape(B, H, n_blocks, l * D)
        )
        blocks_v = (
            v.unfold(2, l, d)[:, :, :n_blocks]
            .transpose(-1, -2)
            .reshape(B, H, n_blocks, l * D)
        )
        return self.phi_k(blocks_k), self.phi_v(blocks_v)


class SelectionBranch(nn.Module):
    """Content-aware top-k block selection from compression scores.

    Aggregates compression attention scores across query positions and selects
    the same top-k blocks for all queries.  This produces standard 4D key/value
    tensors ``(B, H, K, D)`` that are eligible for FlashAttention / memory-
    efficient SDPA kernels, unlike the previous per-query formulation which
    materialized 6D ``(B, H, T, topk, lp, D)`` tensors.
    """

    def __init__(self, block_size=64, topk=16):
        super().__init__()
        self.l_prime = block_size
        self.n = topk

    def forward(self, q, k, v, raw_scores_cmp, n_cmp):
        B, H, T, D = q.shape
        lp, n = self.l_prime, self.n
        n_sel = max(1, T // lp)

        # ── Aggregate compression SCORES (pre-softmax) across query positions ──
        # Using raw scores (not softmax probabilities) preserves absolute
        # importance information — softmax normalizes across blocks and Loses
        # the relative ranking signal.  Max-pool captures the most important
        # query position per block.
        # raw_scores_cmp: (B, H, T, n_cmp) → (B, H, n_cmp)
        scores_agg = raw_scores_cmp.max(dim=2).values

        # Resample to selection grid if needed
        if scores_agg.shape[-1] != n_sel:
            n_c = scores_agg.shape[-1]
            if n_c > n_sel:
                stride = n_c // n_sel
                scores_agg = (
                    scores_agg[..., : stride * n_sel]
                    .reshape(B, H, n_sel, stride)
                    .max(dim=-1).values
                )
            else:
                repeats = n_sel - n_c
                scores_agg = torch.cat(
                    [scores_agg, scores_agg[..., -1:].expand(-1, -1, repeats)], dim=-1
                )

        topk_actual = min(n, n_sel)
        _, top_idx = scores_agg.topk(topk_actual, dim=-1)  # (B, H, topk_actual)

        # Gather full l_prime-token blocks — same blocks for all queries.
        k_sliced = k[:, :, : n_sel * lp, :]
        v_sliced = v[:, :, : n_sel * lp, :]
        k_blocks = k_sliced.reshape(B, H, n_sel, lp, D)
        v_blocks = v_sliced.reshape(B, H, n_sel, lp, D)

        # Gather: top_idx (B, H, topk_actual) → expand for (lp, D)
        bi = top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, lp, D)
        k_sel = torch.gather(k_blocks, dim=2, index=bi)  # (B, H, topk_actual, lp, D)
        v_sel = torch.gather(v_blocks, dim=2, index=bi)
        k_sel = k_sel.reshape(B, H, topk_actual * lp, D)  # standard 4D
        v_sel = v_sel.reshape(B, H, topk_actual * lp, D)

        # ── Causal Masking for Selected Blocks ──
        # Reconstruct the original sequence positions of the selected key tokens
        # top_idx: (B, H, K). Each block has lp tokens.
        # Original block start positions: top_idx * lp
        orig_block_starts = top_idx * lp  # (B, H, K)
        # Offsets within block: 0, 1, ..., lp-1
        offsets = torch.arange(lp, device=q.device)  # (lp,)
        # Original token positions: (B, H, K, lp) -> flatten to (B, H, K*lp)
        orig_k_pos = (orig_block_starts.unsqueeze(-1) + offsets).reshape(B, H, topk_actual * lp)
        
        # Queries can only attend to keys whose original position is <= query position
        q_pos = torch.arange(T, device=q.device).view(1, 1, T, 1)  # (1, 1, T, 1)
        orig_k_pos_exp = orig_k_pos.unsqueeze(2)  # (B, H, 1, K*lp)
        
        valid = orig_k_pos_exp <= q_pos  # (B, H, T, K*lp)
        attn_mask = torch.where(
            valid,
            torch.zeros((), device=q.device, dtype=q.dtype),
            torch.full((), float("-inf"), device=q.device, dtype=q.dtype),
        )
        
        # Standard batched SDPA: (B, H, T, D) × (B, H, K, D) with causal mask
        out = F.scaled_dot_product_attention(q, k_sel, v_sel, attn_mask=attn_mask)
        return out.to(q.dtype), top_idx


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
        attn_mask, valid = cache[(T, w)]
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
            cache[(T, w)] = (attn_mask, valid)

    out = F.scaled_dot_product_attention(q, k_win, v_win, attn_mask=attn_mask)

    # Use cached valid mask; never recompute on cache hit.
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
        use_cuda_kernels=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.use_cuda_kernels = use_cuda_kernels

        proj_cls = BitLinear if use_bitlinear else nn.Linear
        self.q_proj = proj_cls(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = proj_cls(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = proj_cls(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = proj_cls(num_heads * head_dim, hidden_dim, bias=False)

        # FP routing projection for better compression branch scores.
        # Attends at KV-head resolution, always FP (not BitLinear) so routing
        # decisions get rich gradient signals independent of ternary quantization.
        self.routing_k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)

        # 2B4T: subln before output projection (applied to gated 3-branch output)
        self.out_norm = RMSNorm(num_heads * head_dim)

        # RoPE — unified with 1bit_trainer implementation
        self.rope = RotaryEmbedding(head_dim, max_seq_len=max_seq_len, theta=rope_theta)

        self.compression = CompressionBranch(head_dim, cmp_block, cmp_stride)
        self.selection = SelectionBranch(slc_block, slc_topk)
        self.win_size = win_size

        # Per-head gating MLP (tiny — stays FP16)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64, bias=False),
            nn.SiLU(),
            nn.Linear(64, 3 * num_heads, bias=False),
        )
        # Initialize gate biases so window branch dominates early (~0.90),
        # compression and selection start low (~0.05 each). This gives the
        # model a safe fallback while the routing branches learn.
        with torch.no_grad():
            last_layer = self.gate_mlp[-1]
            # last_layer.weight is (3*num_heads, 64) — keep near-zero init
            nn.init.zeros_(last_layer.weight)
            # Add bias per head per branch: cmp(−2.5), slc(−2.5), win(+2.2)
            last_layer.bias = nn.Parameter(torch.zeros(3 * num_heads))
            last_layer.bias.data[0::3] = -2.5  # compression  ≈ sigmoid(−2.5) ≈ 0.076
            last_layer.bias.data[1::3] = -2.5  # selection    ≈ sigmoid(−2.5) ≈ 0.076
            last_layer.bias.data[2::3] = 2.2   # sliding wind ≈ sigmoid(2.2) ≈ 0.90

    def _forward_cuda(self, x, q, k, v, k_routing, B, T):
        """CUDA-accelerated sub-paths: compression, selection, and fused combine."""
        from kernels.compressed_attn.compressed_attn import compressed_attn_forward
        from kernels.selective_attn.selective_attn import selective_attn_forward
        from kernels.subqsa_combine.subqsa_combine import subqsa_combine_forward
        from kernels.block_sparse_ternary.block_sparse_ternary import block_sparse_ternary_matmul
        _ = block_sparse_ternary_matmul  # imported for external use

        n_reps = self.num_heads // self.num_kv_heads if self.num_kv_heads else 1

        # ── Compression branch (fused CUDA kernel) ──
        phi_k = (
            self.compression.phi_k[0].weight,
            None,
            self.compression.phi_k[2].weight,
            None,
        )
        phi_v = (
            self.compression.phi_v[0].weight,
            None,
            self.compression.phi_v[2].weight,
            None,
        )
        k_cmp, v_cmp = compressed_attn_forward(
            k_routing, v, phi_k, phi_v,
            self.compression.l, self.compression.d,
        )
        n_cmp = k_cmp.shape[2]

        # ── Compression attention scores + output ──
        if n_cmp > 0 and n_reps > 1:
            q_re = q.reshape(B, self.num_kv_heads, n_reps, T, self.head_dim)
            scores_cmp = torch.einsum(
                "bhrtd,bhld->bhrtl", q_re.float(), k_cmp.float()
            ) / math.sqrt(self.head_dim)
            p_cmp = F.softmax(scores_cmp, dim=-1).reshape(
                B, self.num_heads, T, n_cmp
            )
            k_cmp_exp = (
                k_cmp[:, :, None]
                .expand(-1, -1, n_reps, -1, -1)
                .reshape(B, self.num_heads, n_cmp, self.head_dim)
            )
            v_cmp_exp = (
                v_cmp[:, :, None]
                .expand(-1, -1, n_reps, -1, -1)
                .reshape(B, self.num_heads, n_cmp, self.head_dim)
            )
            o_cmp = F.scaled_dot_product_attention(
                q, k_cmp_exp, v_cmp_exp,
                dropout_p=self.dropout if self.training else 0.0,
            )
        elif n_cmp > 0:
            scores_cmp = torch.einsum(
                "bhtd,bhld->bhtl", q.float(), k_cmp.float()
            ) / math.sqrt(self.head_dim)
            p_cmp = F.softmax(scores_cmp, dim=-1)
            o_cmp = F.scaled_dot_product_attention(
                q, k_cmp, v_cmp,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            p_cmp = torch.zeros(B, self.num_heads, T, 1, device=x.device)
            o_cmp = torch.zeros_like(q)

        # Expand KV to num_heads for selection and sliding-window branches
        k_h = (
            k[:, :, None]
            .expand(-1, -1, n_reps, -1, -1)
            .reshape(B, self.num_heads, T, self.head_dim)
            if n_reps > 1
            else k
        )
        v_h = (
            v[:, :, None]
            .expand(-1, -1, n_reps, -1, -1)
            .reshape(B, self.num_heads, T, self.head_dim)
            if n_reps > 1
            else v
        )

        # ── Selection branch (fused CUDA kernel) ──
        if n_cmp > 0:
            scores_cmp_for_slc = (
                scores_cmp.reshape(B, self.num_heads, T, n_cmp)
                if n_reps > 1
                else scores_cmp
            )
            scores_agg = scores_cmp_for_slc.max(dim=2).values
        else:
            scores_agg = torch.zeros(
                B, self.num_heads, 1, device=x.device
            )
        o_slc = selective_attn_forward(
            q, k_h, v_h, scores_agg,
            self.selection.n, self.selection.l_prime,
        )

        # ── Sliding window branch (PyTorch) ──
        o_win = sliding_window_attention(q, k_h, v_h, self.win_size)

        # ── Fused combine: gate + 3-way blend + subln + O projection ──
        gamma = getattr(self.o_proj, 'gamma', 1.0)
        o = subqsa_combine_forward(
            x,
            o_cmp, o_slc, o_win,
            self.gate_mlp[0].weight,
            self.gate_mlp[2].weight,
            self.out_norm.weight,
            self.o_proj.weight,
            gamma,
        )
        return o

    def forward(self, x, position_ids, attention_mask=None):
        B, T, _ = x.shape

        # Cast x to the model's working dtype so plain nn.Linear modules
        # (routing_k_proj, gate_mlp) match their weights.  During training
        # autocast handles this; without autocast (e.g. checkpoint
        # verification) the explicit cast prevents float != BFloat16 errors
        # when embed is float32 but weights are bfloat16.
        x = x.to(self.routing_k_proj.weight.dtype)

        # Project with BitLinear (ternary weights, INT8 activations)
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE — unified 1bit_trainer implementation
        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        # ── FP routing projection for better compression scores ──
        # Compute compressed keys from FP-projected K so the compression
        # attention scores are not degraded by ternary quantization.
        k_routing = self.routing_k_proj(x).view(
            B, T, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        k_routing = self.rope(k_routing, position_ids)

        # Early dispatch to CUDA-accelerated path when kernels are enabled
        if self.use_cuda_kernels:
            return self._forward_cuda(x, q, k, v, k_routing, B, T)

        # ── Compression branch ──
        # Compress at num_kv_heads resolution (GQA-aware); expand compressed KV
        # only when attending with query heads.
        # k_cmp uses FP routing K for better scores; v_cmp uses ternary V.
        k_cmp, v_cmp = self.compression(k_routing, v)

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
            n_cmp = k_cmp.shape[2]
            # Compute p_cmp ONCE at KV-head resolution via single einsum/softmax.
            # q: (B, num_heads, T, D) → (B, num_kv_heads, n_reps, T, D)
            # einsum w/ k_cmp (B, num_kv_heads, n_cmp, D) → (B, num_kv_heads, n_reps, T, n_cmp)
            q_re = q.reshape(B, self.num_kv_heads, n_reps, T, self.head_dim)
            scores_cmp = torch.einsum(
                "bhrtd,bhld->bhrtl", q_re.float(), k_cmp.float()
            ) / math.sqrt(self.head_dim)
            p_cmp = F.softmax(scores_cmp, dim=-1)  # (B, num_kv_heads, n_reps, T, n_cmp)
            p_cmp = p_cmp.reshape(
                B, self.num_heads, T, n_cmp
            )  # (B, num_heads, T, n_cmp)

            # GQA-aware SDPA: expand compressed KV once and use a single call.
            # k_cmp/v_cmp: (B, num_kv_heads, n_cmp, D) → (B, num_heads, n_cmp, D)
            k_cmp_exp = (
                k_cmp[:, :, None]
                .expand(-1, -1, n_reps, -1, -1)
                .reshape(B, self.num_heads, n_cmp, self.head_dim)
            )
            v_cmp_exp = (
                v_cmp[:, :, None]
                .expand(-1, -1, n_reps, -1, -1)
                .reshape(B, self.num_heads, n_cmp, self.head_dim)
            )
            o_cmp = F.scaled_dot_product_attention(
                q,
                k_cmp_exp,
                v_cmp_exp,
                dropout_p=self.dropout if self.training else 0.0,
            )
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

        # ── Selection branch: per-query top-k blocks (standard 4D SDPA) ──
        # Pass raw scores (pre-softmax) so SelectionBranch can use absolute
        # importance values instead of normalized probabilities.
        # scores_cmp: (B, H, T, n_cmp) or (B, num_kv_h, n_reps, T, n_cmp)
        if k_cmp.shape[2] > 0 and n_reps > 1:
            # Reshape GQA scores to (B, H, T, n_cmp) for SelectionBranch
            scores_cmp_for_slc = scores_cmp.reshape(B, self.num_heads, T, n_cmp)
        elif k_cmp.shape[2] > 0:
            scores_cmp_for_slc = scores_cmp
        else:
            scores_cmp_for_slc = torch.zeros(B, self.num_heads, T, 1, device=x.device)
        o_slc, _ = self.selection(q, k_h, v_h, scores_cmp_for_slc, n_cmp)

        # ── Sliding window branch: standard 4D causal SDPA ──
        # Uses the existing sliding_window_attention helper which builds a
        # (T, w) additive causal mask — no per-query 5D expansion needed.
        # This keeps tensors in (B, H, T, D) / (B, H, w, D) shapes that
        # are eligible for FlashAttention / memory-efficient SDPA kernels.
        o_win = sliding_window_attention(q, k_h, v_h, self.win_size)

        # ── 3-way gating (independent gradients per branch) ──
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
