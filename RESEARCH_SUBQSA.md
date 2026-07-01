# Research: Subquadratic Sparse Attention (SubQSA / SSA)

**Date:** 2026-06-30
**Scope:** SubQ's SSA (subq.ai), DeepSeek's NSA (the open, peer-reviewed sibling), and how to implement a working subquadratic sparse attention kernel for the Ultimate Trainer.

---

## 1. Why This Matters

Standard attention is O(n²). At 1M tokens, dense attention on a 27B model spends roughly **80% of decoding latency in attention alone** (DeepSeek NSA paper, §1). All "make-attention-cheaper" attempts trade one of three properties:

1. **Linear scaling** in compute and memory.
2. **Content-dependent routing** — the model decides what to attend to based on *meaning*, not position.
3. **Exact retrieval from arbitrary positions** — no compressed-state loss.

Until 2025-2026, every architecture sacrificed at least one:

| Approach | Linear? | Content-aware? | Exact retrieval? | Killer flaw |
|---|---|---|---|---|
| Dense / FlashAttention | ❌ O(n²) | ✓ | ✓ | Quadratic cost |
| Fixed sparse (Longformer, Sliding-Window) | ✓ | ❌ | ❌ | Misses out-of-pattern tokens |
| State Space (Mamba) | ✓ | ✓ | ❌ | Fixed-state recall decay |
| Hybrid (Jamba, dense + Mamba) | ❌ on dense layers | ✓ | ✓ | Dense layers dominate at long ctx |
| DeepSeek Sparse Attention (DSA) | ❌ (O(n²) indexer) | ✓ | ✓ | Indexer is itself quadratic |
| **NSA** (DeepSeek, 2025) | ✓ (blockwise) | ✓ | ✓ | Three branches; complex kernel |
| **SSA** (SubQ, 2026) | ✓ | ✓ | ✓ | Proprietary; mechanism not published |

NSA is the **closest public peer-reviewed work** to SubQ's SSA — and the only one with a usable open-source reference implementation. Our Ultimate Trainer will implement an **NSA-style mechanism** and we will refer to it as **SubQSA** in this codebase.

---

## 2. SubQ's Account of SSA (subq.ai)

From the SubQ 1.1 Small technical report and the "How SSA Makes Long Context Practical" post:

### 2.1 The mechanism, as described

- "For each query, the model selects which parts of the sequence are worth attending to, and computes attention exactly over those positions."
- Selection is **content-dependent** (learned from Q and K), **not** positional.
- Linear scaling in compute AND memory (so it is not just FlashAttention with sparsity).
- "Sparse retrieval from arbitrary positions" — explicit guarantee that any past token can be selected, unlike compressed-state methods.

### 2.2 Reported numbers (1M context, B200, vs FlashAttention-2)

| Context | FA2 latency | SSA latency | Speedup |
|---|---|---|---|
| 128K | 320 ms | 47 ms | 6.88× |
| 256K | 1,272 ms | 94 ms | 13.51× |
| 512K | 5,229 ms | 190 ms | 27.54× |
| 1M | 21,411 ms | 381 ms | **56.2×** |

- **64.5× less attention FLOPs** vs dense at 1M.
- Sparsity at 12M context ≈ **0.13% of pairwise relationships kept**, yet 98% needle-in-haystack accuracy.

### 2.3 Training recipe (per SubQ)

- Start from an open-weight frontier checkpoint.
- Replace dense attention with SSA.
- **Staged context extension**: 262K → 512K → 1M → 2M.
- ~1T tokens of continued pretraining on long, naturally-cohesive artifacts (books, repos, contract corpora).
- Then SFT, then RL targeting long-context retrieval reliability.
- Distributed **sequence parallelism** for million-token sequences (single device cannot hold them).

### 2.4 What is NOT published

- Exact routing function (how Q and K are projected into the selection space).
- Block size, number of selected blocks, top-k policy.
- Backward kernel.
- Whether selection is hierarchical (compressed + fine) or single-stage.

We must reconstruct these from NSA.

---

## 3. NSA — The Reference Implementation We Can Actually Build

**Paper**: ["Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"](https://arxiv.org/abs/2502.11089), Yuan et al., DeepSeek + Peking U + UW. **ACL 2025 Best Paper**. Code: [fla-org/native-sparse-attention](https://github.com/fla-org/native-sparse-attention) and [lucidrains/native-sparse-attention-pytorch](https://github.com/lucidrains/native-sparse-attention-pytorch).

### 3.1 The Three-Branch Design

For each query `q_t`, NSA replaces dense KV `(k_:t, v_:t)` with a content-aware compact set computed by three parallel branches, then gates them:

```
o_t = Σ_{c ∈ {cmp, slc, win}}  g_t^c  ·  Attn(q_t, K̃_t^c, Ṽ_t^c)
```

where `g_t^c ∈ [0,1]` is a per-branch gate (MLP + sigmoid from `q_t`).

| Branch | Purpose | KV size | Mechanism |
|---|---|---|---|
| **cmp** (compression) | Coarse global view | ~`(t - l) / d` | MLP `φ` maps each block of `l` keys (stride `d`) to a single compressed key/value |
| **slc** (selection) | Fine, content-routed retrieval | top-`n` blocks of size `l'` | Block-importance score derived from cmp attention; pick top-n blocks per query |
| **win** (sliding window) | Local context | last `w` tokens | Standard windowed attention |

The sum of selected sizes `N_t = |K̃_cmp| + |K̃_slc| + |K̃_win|` is held `N_t << t`, giving linear total cost.

### 3.2 Selection Without a Separate Indexer

The clever bit: NSA does NOT spin up a separate quadratic indexer (DSA's failure mode). Instead, it **reuses the attention scores from the compression branch** as a proxy for block importance:

```
p_t^cmp = softmax(q_tᵀ · K̃_t^cmp)            # already computed in cmp branch
p_t^slc = aggregate p_t^cmp into selection-block grid    # cheap reshape/sum
top_blocks_t = topk(p_t^slc, n)
```

So the selection branch reuses the compression branch's softmax for free — no extra O(n²) work.

### 3.3 Sliding Window Branch: Why It's Needed

Without it, the compression branch tends to **shortcut** — the model learns to dump everything into local patterns instead of routing globally. The sliding window absorbs local attention so cmp and slc can specialize in global structure. (Ablation in NSA §6.1.)

### 3.4 Hardware-Aware Kernel Design

NSA's speedup is real only because the kernel was built around two constraints:

- **GQA-aware**: each KV head is shared across multiple Q heads. NSA aggregates selection decisions per GQA group, so memory-access volume = union over the group, not per-head. This is why Quest underperforms on GQA models.
- **Blockwise everywhere**: selection picks **blocks**, not tokens, so memory access stays contiguous and Tensor Cores stay saturated. Block size matches FlashAttention's tile size.
- **Online top-k**: never materialize the full `p_t^slc`. (Added Feb 2025 in fla-org repo.)
- **Fused selected + sliding kernel** (Feb 2025).
- Reference Triton implementation: `parallel_nsa(q, k, v, g_cmp, g_slc, g_swa, block_indices, block_counts, block_size, window_size)`.

### 3.5 Hyperparameters (NSA paper, 27B / 260B tokens config)

| Parameter | Value |
|---|---|
| Compression block length `l` | 32 |
| Compression stride `d` | 16 |
| Selection block size `l'` | 64 |
| Number of selected blocks `n` | 16 |
| Sliding window size `w` | 512 |
| Sparsity at 64K context | ~ 6% of full attention |

Pretraining on 27B params × 260B tokens demonstrated **no degradation** vs full attention on general benchmarks, **better** on long-context (NIAH, LongBench), and **better** on chain-of-thought reasoning. Plus end-to-end speedup across forward / backward / decode at 64K.

### 3.6 Cost Model

```
cmp:  O((t/d) · d_head)              ≈ linear
slc:  O(n · l' · d_head)             ≈ linear (n, l' are constants)
win:  O(w · d_head)                  ≈ linear (w is constant)
total per query: linear in t, dominated by (t/d) for cmp
```

So total per layer: `O(n_seq · t/d) = O(n_seq²/d)` with `d` typically 16. At 1M tokens this is 62500× less work than dense `O(n²)`. The 56× wall-clock claim from SubQ falls naturally out of the constant-factor side of NSA.

---

## 4. SSA vs NSA — Same Family

The SubQ post and the NSA paper read like cousins. Differences are mostly engineering:

| Aspect | NSA (open) | SSA (SubQ, proprietary) |
|---|---|---|
| Branches | 3 (cmp + slc + win) | unspecified — possibly merged |
| Selection signal | Compression attention scores | "Compressed routing space" (likely lower-rank Q/K projection) |
| Routing dim | Same as `d_head` | "Smaller routing space" (lower rank) |
| Gumbel? | No (gates only) | Mentioned obliquely; unconfirmed |
| Hierarchical extension | None published | Implied — staged ctx 262K→2M |
| Reported speedup @ 1M | not measured at 1M | 56.2× over FA2 |
| Open kernel | ✓ Triton (fla-org) | ❌ |

**Decision for the Ultimate Trainer**: build on NSA's three-branch design. It is the closest open analog, has working kernels, and the SSA paper's behavioral claims (linear, content-routed, exact retrieval) are all properties NSA already has.

---

## 5. Training Pipeline for SubQSA

Combining SubQ's described pipeline with NSA's training methodology:

### Stage 1 — Pretraining at base context (e.g. 4K → 32K)
- Standard LM objective.
- All three branches active from step 0 (NSA shows no warmup needed; cmp + slc + win co-train cleanly).
- Loss = cross-entropy.
- Optimizer: AdamW, cosine LR, gradient clipping.

### Stage 2 — Staged Context Extension
Following SubQ: progressively double the context, continue training:

| Stage | Length | Steps | Note |
|---|---|---|---|
| 2.1 | 64K | ~30K | Base routing established |
| 2.2 | 128K | ~20K | First "long" stage |
| 2.3 | 256K | ~15K | Memory pressure begins; sequence parallelism kicks in |
| 2.4 | 512K | ~10K | Distributed only |
| 2.5 | 1M | ~10K | Production-target context |
| 2.6 | 2M+ | optional | Generalization stage |

Each stage:
- Resumes from previous stage checkpoint.
- Doubles RoPE base accordingly (or uses YaRN / ABF).
- Re-tunes window size `w` (kept constant), `n` selected blocks (scaled), and `l'` if needed.
- Shorter step count is OK because routing generalizes — confirmed by SubQ's 12M generalization from 1M training.

### Stage 3 — SFT
- Instruction following, structured reasoning, long-document tasks.
- Loss with **sum-reduction** (BitNet 2B4T recipe).
- ~5K–10K steps at reduced LR.

### Stage 4 — Long-Context RL
This is the **only stage that genuinely targets the long-context failure mode**:
- Reward signal targets "use evidence from far in context, not nearby shortcuts."
- KL penalty against the SFT model.
- Group Relative Policy Optimization (GRPO) is a reasonable default (used in DeepSeek-R1).
- Alternative: DPO with long-context preference pairs (cheaper, BitNet 2B4T's choice).

---

## 6. Implementation Sketch for the Ultimate Trainer

### 6.1 Module Layout (`subqsa/`)

```python
class CompressionBranch(nn.Module):
    """Block-aggregate keys/values via a learned MLP φ."""
    def __init__(self, head_dim, block_len, stride):
        self.phi = nn.Sequential(
            nn.Linear(head_dim * block_len, head_dim),  # collapse block
            nn.SiLU(),
            nn.Linear(head_dim, head_dim),
        )

    def forward(self, k, v):
        # k, v: (B, H, T, D)
        # Return compressed (B, H, T_cmp, D) where T_cmp = (T - l) // d
        ...

class SelectionBranch(nn.Module):
    """Top-n blocks per query, gathered."""
    def forward(self, q, k, v, p_cmp):
        # p_cmp: attention scores from compression branch
        p_slc = aggregate_to_selection_grid(p_cmp)
        top_idx = topk(p_slc, n)
        K_sel = gather_blocks(k, top_idx, l_prime)
        V_sel = gather_blocks(v, top_idx, l_prime)
        return flash_attention(q, K_sel, V_sel)

class SubQSAAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads,
                 cmp_block=32, cmp_stride=16,
                 slc_block=64, slc_topk=16,
                 win_size=512):
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = ...
        self.cmp = CompressionBranch(...)
        self.slc = SelectionBranch(...)
        self.gate_mlp = nn.Linear(hidden_dim, 3)  # three branch gates

    def forward(self, x, position_ids):
        q, k, v = project(x)
        q, k = apply_rope(q, k, position_ids)

        # Three branches
        k_cmp, v_cmp = self.cmp(k, v)
        o_cmp, p_cmp = attention_with_scores(q, k_cmp, v_cmp)
        o_slc        = self.slc(q, k, v, p_cmp)
        o_win        = flash_attention_windowed(q, k, v, self.win_size)

        # Gate
        g = self.gate_mlp(x).sigmoid()  # (B, T, 3)
        out = g[..., 0:1] * o_cmp + g[..., 1:2] * o_slc + g[..., 2:3] * o_win
        return self.o_proj(out)
```

### 6.2 Kernel Choice

- **Phase 1 (correctness)**: pure PyTorch using `torch.gather` and FlashAttention v2 for each branch. Slow but correct. Validate against full attention on short sequences.
- **Phase 2 (speed)**: integrate `parallel_nsa` from [fla-org/native-sparse-attention](https://github.com/fla-org/native-sparse-attention) — Apache-licensed Triton kernel, already supports variable `n` per query and fused select+swa.
- **Phase 3 (custom)**: only if Phase 2 is insufficient.

### 6.3 GQA Integration

NSA's GQA-aware selection is mandatory if we use grouped-query attention (we do — see BitNet recipe). For each GQA group, take the **union** of selected blocks across the group's queries. The fla-org kernel handles this.

---

## 7. Open Questions for SubQSA

1. **Gumbel-Softmax vs hard top-k**: NSA uses hard top-k with no Gumbel because the selection signal piggybacks on a differentiable softmax. We will inherit this and only revisit if training is unstable.
2. **Routing dim < head dim?** SubQ implies yes ("compact routing space"). NSA uses full head dim. We default to full head dim, leave a knob for lower-rank routing.
3. **Layer-wise mixing**: Do all layers use SubQSA, or hybrid (some layers dense)? NSA: all SubQSA. Jamba: hybrid. We default to all SubQSA, leave a knob.
4. **Position encoding at 1M+**: RoPE with base scaling (NTK-aware or YaRN). Same as any long-context model.

---

## 8. References

- [NSA paper (2025, ACL Best Paper)](https://arxiv.org/abs/2502.11089) — primary reference for implementation
- [fla-org/native-sparse-attention](https://github.com/fla-org/native-sparse-attention) — Triton kernels, 1k★, MIT
- [lucidrains/native-sparse-attention-pytorch](https://github.com/lucidrains/native-sparse-attention-pytorch) — readable pure-PyTorch port
- [Subquadratic — How SSA Makes Long Context Practical](https://subq.ai/how-ssa-makes-long-context-practical)
- [Subquadratic — Introducing SubQ 1.1 Small](https://subq.ai/subq-1-1-small-technical-report)
- [SubQ-1.1-Small Technical Report PDF](https://subq.ai/docs/subq-1-1-small-model-card.pdf)
- [Appen third-party benchmark report](https://www.appen.com/whitepapers/subquadratic-preview-model-benchmark-evaluation)
- [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — foundational
- [Mamba (2023)](https://arxiv.org/abs/2312.00752) — the state-space alternative we are explicitly NOT picking
