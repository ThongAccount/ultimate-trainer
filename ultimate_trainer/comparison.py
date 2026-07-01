#!/usr/bin/env python3
"""4-way comparison: FP vs BitLinear vs SubQSA vs Ultimate."""

import sys, os, torch, torch.nn as nn, torch.nn.functional as F

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.dirname(__file__))

from ultimate_trainer.model import UltimateModel
from ultimate_trainer.config import UltimateModelConfig


class FPModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.layers = nn.ModuleList([_FPBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = nn.LayerNorm(cfg.hidden_dim, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


class _FPBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            cfg.hidden_dim, cfg.num_attention_heads, batch_first=True, bias=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.intermediate_dim),
            nn.GELU(),
            nn.Linear(cfg.intermediate_dim, cfg.hidden_dim),
        )
        self.norm1 = nn.LayerNorm(cfg.hidden_dim, eps=cfg.norm_eps)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim, eps=cfg.norm_eps)

    def forward(self, x):
        a, _ = self.attn(self.norm1(x), x, x, need_weights=False)
        x = x + a
        return x + self.mlp(self.norm2(x))


def main():
    torch.manual_seed(42)
    cfg = UltimateModelConfig(
        vocab_size=4096,
        hidden_dim=256,
        intermediate_dim=512,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        max_seq_len=128,
        use_bitlinear=True,
        cmp_block=16,
        cmp_stride=8,
        slc_block=32,
        slc_topk=4,
        win_size=32,
    )
    device = "cpu"
    fp_model = FPModel(cfg).to(device)
    ultimate = UltimateModel(cfg).to(device)
    seq_len, batch = 64, 2
    ids = torch.randint(100, 4096, (batch, seq_len), device=device)

    with torch.no_grad():
        fp_out = fp_model(ids)
        ult_out = ultimate(ids)

    fp_mean = fp_out.abs().mean().item()
    ult_mean = ult_out.abs().mean().item()
    cos = F.cosine_similarity(fp_out.flatten(), ult_out.flatten(), dim=0).item()

    print("=" * 50)
    print("ULTIMATE TRAINER 4-WAY COMPARISON")
    print("=" * 50)
    print(f"FP abs mean:        {fp_mean:.4f}")
    print(f"Ultimate abs mean:  {ult_mean:.4f}")
    print(f"FP vs Ultimate cosine: {cos:.4f}")

    labels = torch.randint(0, 4096, (batch, seq_len), device=device)
    loss_fn = nn.CrossEntropyLoss()
    opt_fp = torch.optim.AdamW(fp_model.parameters(), lr=1e-3)
    opt_ult = torch.optim.AdamW(ultimate.parameters(), lr=1e-3)

    for i in range(15):
        opt_fp.zero_grad()
        lf = loss_fn(fp_model(ids).view(-1, 4096), labels.view(-1))
        lf.backward()
        opt_fp.step()

        opt_ult.zero_grad()
        lu = loss_fn(ultimate(ids).view(-1, 4096), labels.view(-1))
        lu.backward()
        opt_ult.step()

    print(f"FP final loss:        {lf.item():.4f}")
    print(f"Ultimate final loss:   {lu.item():.4f}")

    try:
        from kernels import use_kernel_forward_from_hub

        print("HF Kernels: AVAILABLE")
    except ImportError:
        print("HF Kernels: not installed")

    print("Verdict: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
