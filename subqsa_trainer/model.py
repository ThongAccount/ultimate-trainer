"""SubQSA full transformer model. Clean, verified shapes."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from subqsa_trainer.subqsa import SubQSA as SubQSAAttention


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        return x / (rms + self.eps) * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = SubQSAAttention(
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_kv_heads or cfg.num_attention_heads,
            head_dim=cfg.head_dim,
            max_seq_len=cfg.max_seq_len,
            cmp_block=cfg.subqsa.cmp_block,
            cmp_stride=cfg.subqsa.cmp_stride,
            slc_block=cfg.subqsa.slc_block,
            slc_topk=cfg.subqsa.slc_topk,
            win_size=cfg.subqsa.win_size,
        )
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.intermediate_dim, bias=False),
            nn.GELU(),
            nn.Linear(cfg.intermediate_dim, cfg.hidden_dim, bias=False),
        )
        self.norm1 = RMSNorm(cfg.hidden_dim, cfg.norm_eps)
        self.norm2 = RMSNorm(cfg.hidden_dim, cfg.norm_eps)
        self.drop = nn.Dropout(cfg.hidden_dropout)

    def forward(self, x, start_pos=0):
        r = x
        x = self.norm1(x)
        x = self.attn(x, start_pos=start_pos, seq_len=x.shape[1])
        x = self.drop(x) + r
        r = x
        x = self.norm2(x)
        x = self.mlp(x)
        return self.drop(x) + r


class SubQSAModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.num_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m):
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
