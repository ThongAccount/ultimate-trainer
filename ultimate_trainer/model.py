"""Ultimate Trainer model: BitLinear + SubQSA (2B4T spec).

Architecture per transformer block (2B4T subln pattern):
  x → subln_in → SubQSAAttention(BitLinear Q,K,V → attn → subln_out → BitLinear O)
    → + → subln_in → ReLU²(BitLinear gate) * BitLinear up → subln_out → BitLinear down → + → out
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultimate_trainer.bitlinear import BitLinear, RMSNorm
from ultimate_trainer.subqsa import SubQSAAttention


class TransformerBlock(nn.Module):
    """Transformer block with 2B4T spec: subln + SubQSA + ReLU² FFN."""

    def __init__(self, cfg):
        super().__init__()
        # Attention sub-block: outer subln before QKV; inner subln inside SubQSAAttention before O
        self.attn_norm = RMSNorm(cfg.hidden_dim, cfg.norm_eps)
        self.attn = SubQSAAttention(
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            max_seq_len=cfg.max_seq_len,
            cmp_block=cfg.cmp_block,
            cmp_stride=cfg.cmp_stride,
            slc_block=cfg.slc_block,
            slc_topk=cfg.slc_topk,
            win_size=cfg.win_size,
            use_bitlinear=cfg.use_bitlinear,
        )

        # ReLU² FFN sub-block: outer subln before gate/up; inner subln before down
        self.ffn_norm = RMSNorm(cfg.hidden_dim, cfg.norm_eps)
        self.ffn_out_norm = RMSNorm(cfg.intermediate_dim, cfg.norm_eps)
        self.ffn_gate = BitLinear(cfg.hidden_dim, cfg.intermediate_dim, bias=False)
        self.ffn_up = BitLinear(cfg.hidden_dim, cfg.intermediate_dim, bias=False)
        self.ffn_down = BitLinear(cfg.intermediate_dim, cfg.hidden_dim, bias=False)

    def forward(self, x, start_pos=0):
        B, T = x.shape[:2]
        position_ids = (
            torch.arange(start_pos, start_pos + T, device=x.device)
            .unsqueeze(0)
            .expand(B, -1)
        )

        # ── Attention sub-block (subln → SubQSA → +) ──
        # SubQSAAttention internally applies subln_out before O projection
        r = x
        x = self.attn_norm(x)
        x = self.attn(x, position_ids)
        x = r + x

        # ── ReLU² FFN sub-block (subln → gate/up → ReLU² → subln → down → +) ──
        r = x
        x = self.ffn_norm(x)
        gate = F.relu(self.ffn_gate(x)).pow(2)  # ReLU²
        up = self.ffn_up(x)
        hidden = gate * up
        hidden = self.ffn_out_norm(hidden)  # subln before down projection
        x = self.ffn_down(hidden)
        x = r + x
        return x


class UltimateModel(nn.Module):
    """Ultimate merged model: BitNet b1.58 2B4T + SubQSA sparse attention.

    Architecture:
      FP16 embedding → N × [subln → SubQSAAttention(BitLinear QKV/O) → subln → ReLU² FFN(BitLinear)]
        → RMSNorm → BitLinear LM head (tied)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim, padding_idx=0)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.num_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_dim, cfg.norm_eps)
        self.lm_head = BitLinear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tie embeddings
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids, start_pos=0):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, start_pos=start_pos)
        return self.lm_head(self.norm(x))

    def get_loss(self, input_ids, labels=None):
        logits = self(input_ids)
        if labels is None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
        else:
            shift_logits = logits.contiguous()
            shift_labels = labels.contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=0,
        )
