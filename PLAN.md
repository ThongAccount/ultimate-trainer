# Ultimate AI Model — Project Plan

**Date:** 2026-06-30
**Goal:** A reference trainer that merges **native 1-bit (ternary) quantization** with **SubQSA (Subquadratic Sparse Attention)** so the same model gets both the **~10× memory reduction** of BitNet b1.58 and the **~56× attention speedup** of NSA-style sparse routing.

This plan is the synthesis of [RESEARCH_1BIT.md](RESEARCH_1BIT.md) and [RESEARCH_SUBQSA.md](RESEARCH_SUBQSA.md). Read those first for derivations.

---

## 1. The Big Idea

A normal LLM training stack has two heavy operators:

1. **Linear layers** (Q/K/V/O projections, FFN gate/up/down, LM head) — dominate memory and matmul FLOPs.
2. **Attention** — dominates latency at long context (O(n²)).

The Ultimate Trainer **replaces both at once**:

| Operator | Standard | Ultimate Trainer |
|---|---|---|
| Linear layer | `nn.Linear`, BF16 weights | **`BitLinear`** — ternary {-1, 0, +1} weights, INT8 activations |
| Attention | `F.scaled_dot_product_attention`, O(n²) | **`SubQSAAttention`** — NSA-style 3-branch sparse, O(n) |
| Embeddings | FP16 | FP16 (unchanged — only ~1% of params at scale) |
| LM head | tied with embeddings | tied, ternary BitLinear |
| Normalization | RMSNorm | **subln** (two norms per sub-block — required for ternary stability) |
| FFN activation | SwiGLU | **Squared ReLU** (`ReLU²`) — sparsity-friendly under quantization |

**Net effect**: at 1M context, a 2B-parameter model uses
- **~0.5GB** for ternary weights (vs ~4GB BF16)
- **~0.4s** prefill attention (vs ~21s FlashAttention-2 dense — see [SubQSA §2.2](RESEARCH_SUBQSA.md))
- Activations remain INT8 → cheap KV cache too.

This is not stacking two papers blindly. It works because **BitNet 2B4T already uses `ReLU²` + `subln`**, which produces naturally sparse activations — and NSA's selection branch likes sparse activations. The two methods are co-friendly, not just co-located.

---

## 2. Repo Layout (Final)

```
ultimate-ai-model/
├── PLAN.md                       (this file)
├── RESEARCH_1BIT.md              (deep research — done)
├── RESEARCH_SUBQSA.md            (deep research — done)
│
├── 1bit-trainer/                 (Task 1 — done; v1 b1.58 reference)
│   ├── config.py
│   ├── model.py
│   └── train.py
│
├── subqsa_trainer/               (Task 2 — NSA-style SubQSA over FP attention)
│   ├── config.py
│   ├── subqsa.py                 (CompressionBranch, SelectionBranch, SubQSAAttention)
│   ├── model.py                  (full transformer with SubQSA)
│   ├── train.py                  (staged ctx extension: 4K→32K→128K→1M)
│   └── comparison.py             (dense vs SubQSA comparison)
│
└── ultimate_trainer/             (Task 3 — the merged trainer; THE deliverable)
    ├── config.py                 (UltimateModelConfig + UltimateTrainingConfig)
    ├── bitlinear.py              (2B4T-spec BitLinear extracted from 1bit-trainer)
    ├── subqsa.py                 (SubQSA with BitLinear projections + subln)
    ├── model.py                  (BitLinear FFN/QKV/O + SubQSAAttention)
    ├── train.py                  (pretrain → SFT → DPO/GRPO + staged ctx extension)
    └── comparison.py             (4-way comparison)
```

Each tier is independently runnable so we can ablate "1-bit alone", "SubQSA alone", "1-bit + SubQSA" against each other.

---

## 3. Task Status

### ✅ Task 1: 1-Bit Trainer (v1 b1.58) — DONE

[1bit-trainer/config.py](1bit-trainer/config.py), [1bit-trainer/model.py](1bit-trainer/model.py), [1bit-trainer/train.py](1bit-trainer/train.py) are complete and runnable. They implement the **2024 v1 b1.58 paper** (absmean activations, SwiGLU, RMSNorm).

`1bit-trainer/` is intentionally left untouched as the v1 reference. `ultimate_trainer/bitlinear.py` implements the 2B4T spec independently, and both `subqsa_trainer/subqsa.py` and `ultimate_trainer/subqsa.py` load `RotaryEmbedding` from `1bit-trainer/model.py` via `importlib` to keep the reference intact while sharing RoPE.

### ✅ Task 2: SubQSA Trainer — IMPLEMENTED

NSA's three-branch sparse attention ([RESEARCH_SUBQSA.md §3](RESEARCH_SUBQSA.md)) is implemented with FP weights. Unit tests verify that `SelectionBranch` returns exactly `topk` contiguous blocks and `sliding_window_attention` only attends to the last `win_size` tokens.

| File | Contents |
|---|---|
| [subqsa_trainer/config.py](subqsa_trainer/config.py) | `ModelConfig` with NSA hyperparams: `cmp_block=32`, `cmp_stride=16`, `slc_block=64`, `slc_topk=16`, `win_size=512`. `TrainingConfig` with staged context extension schedule. |
| [subqsa_trainer/subqsa.py](subqsa_trainer/subqsa.py) | `CompressionBranch` (MLP φ + blockwise pooling), `SelectionBranch` (importance scoring + top-k block gather), `SubQSAAttention` (3 branches + gate MLP). Pure PyTorch fallback. |
| [subqsa_trainer/model.py](subqsa_trainer/model.py) | LLaMA-like transformer with `SubQSAAttention` replacing dense attention. RMSNorm, SwiGLU, RoPE — standard FP. |
| [subqsa_trainer/train.py](subqsa_trainer/train.py) | DDP-capable training loop with staged context extension. |

**Verification status**: The dense-vs-SubQSA comparison script runs, but **cosine similarity is ~0.012 vs the target ≥ 0.7**. The current mean-pool compression + top-k-by-magnitude selection does not yet match dense attention quality; this is a research gap, not a runtime bug.

### ✅ Task 3: Ultimate Trainer (1-bit + SubQSA) — IMPLEMENTED

The merged deliverable exists and runs end-to-end on CPU.

| File | Contents |
|---|---|
| [ultimate_trainer/config.py](ultimate_trainer/config.py) | Union of BitNet 2B4T + NSA configs. `use_bitlinear=True`, `use_subqsa=True`, all NSA hyperparams. |
| [ultimate_trainer/bitlinear.py](ultimate_trainer/bitlinear.py) | **2B4T-spec** `BitLinear`: absmean weights, **absmax** per-token 8-bit activations, STE backward. Safe fused-Triton dispatch (only when CUDA + kernel available); eager CPU fallback otherwise. Eval-mode ternary weight refresh added. |
| [ultimate_trainer/subqsa.py](ultimate_trainer/subqsa.py) | SubQSA with BitLinear projections + subln. GQA-aware compression/expansion and unified RoPE. |
| [ultimate_trainer/model.py](ultimate_trainer/model.py) | `UltimateModel`: FP16 embedding → N × `[subln → SubQSAAttention (BitLinear projections) → subln → ReLU² FFN (BitLinear gate/up/down)]` → BitLinear LM head (tied). |
| [ultimate_trainer/train.py](ultimate_trainer/train.py) | Training loop with dummy/real data and smoke mode. |

**Verification status**: The FP-vs-Ultimate comparison runs, but **cosine similarity is ~0 and Ultimate loss is ~46× the FP loss**. The architecture is trainable but has not yet aligned with the FP baseline.

---

## 4. Architecture Details (Ultimate Model)

### 4.1 Block Structure

```
x  ── subln ── SubQSAAttention(BitLinear Q,K,V,O) ─┐
   └────────────────────────────────────────────── + ── subln ── ReLU² FFN(BitLinear gate,up,down) ─ + ── x_out
                                                                                                    └──────┘
```

Two residual streams as usual. The **second subln inside each sub-block** (before the output projection) is the BitNet 2B4T trick that makes deep ternary training stable.

### 4.2 SubQSAAttention with BitLinear projections

The projections become ternary, but the **routing math stays full-precision** (or BF16). Specifically:

| Sub-op | Precision |
|---|---|
| `q = BitLinear(x)`, `k = BitLinear(x)`, `v = BitLinear(x)` | weights ternary, activations INT8 |
| RoPE on `q`, `k` | BF16 |
| Compression branch MLP `φ` | `BitLinear` (ternary) |
| Compression branch attention `softmax(qK_cmpᵀ)` | BF16 |
| Selection branch `topk(p_slc, n)` | BF16 |
| Selection / window / compression attention pass | BF16 (FlashAttention v2) |
| Gate MLP `nn.Linear → sigmoid` | small, kept FP16 — cheap |
| Output projection `o = BitLinear(gated_combine)` | weights ternary, activations INT8 |

So the **matrix multiplications that scale with hidden_dim are ternary**, while the **softmax/topk math that needs precision** stays BF16. Best of both worlds.

### 4.3 Sizing — First Run Target

| Dim | Value | Source |
|---|---|---|
| Params | 2.0B | matches BitNet 2B4T |
| Layers | 30 | 2B4T |
| Hidden | 2560 | 2B4T |
| Intermediate (FFN) | 6912 (2.7×) | 2B4T (smaller than SwiGLU's 8/3× because ReLU² doesn't need 2 gate matmuls bridging) |
| Q heads | 20 | 2B4T |
| KV heads (GQA) | 5 | 2B4T |
| Head dim | 128 | 2B4T |
| Vocab | 128 256 | LLaMA 3 BPE (2B4T choice) |
| Initial max_seq_len | 4096 | extend in stages |
| Target max_seq_len | 1 048 576 (1M) | SubQSA target |
| Tokens | 1T–4T | BitNet 2B4T used 4T; we can start at 1T |

This sizing is deliberate: the **2B4T number is the only validated open-source native-1.58-bit + ≥2B + ≥1T-token training run on record**. We anchor to it so any divergence in results is attributable to *our* SubQSA addition, not to mis-sizing.

### 4.4 Hyperparameters at a Glance

**Quantization**:
- Weight quantization: absmean → {-1, 0, +1} via STE (Eq. 1–3 of b1.58)
- Activation quantization: **absmax** per-token, 8-bit signed, [-127, 127]. *Not* absmean (2B4T deviation from v1).
- Activation quantization warmup: gradually enabled over **5000 steps** (v1 b1.58 recommendation, 2B4T does not specify; we keep it).

**SubQSA branches**:
- Compression block `l = 32`, stride `d = 16` → ~`T/16` compressed keys per layer.
- Selection block `l' = 64`, top-`n = 16` → 1024 selected tokens per query per layer regardless of context.
- Sliding window `w = 512` (fits inside 2B4T's RoPE base comfortably).
- At 1M tokens, sparsity ≈ `(T/16 + 16·64 + 512) / T ≈ 6.4%` — close to NSA's reported ~6%.

**Optimizer**:
- AdamW, β1=0.9, β2=0.95, ε=1e-8.
- LR: two-stage cosine. Stage 1 peaks at ~1.5e-3 (1-bit models tolerate higher LR than FP); Stage 2 cooldown peaks at ~3e-4.
- Weight decay: 0.1 in Stage 1, **0** in Stage 2.
- Grad clip: 1.0.

**Distributed**:
- DDP for ≤256K context.
- Sequence parallelism (Ring Attention or DeepSpeed-Ulysses) for ≥512K.
- ZeRO-2 acceptable for ≥1B params; ZeRO-3 only if needed (FSDP is cleaner with BitLinear's custom backward).

---

## 5. Training Schedule

The full schedule is ~80K–120K steps depending on batch size:

| Phase | Length | Steps | Notes |
|---|---|---|---|
| **P1** Pretrain @ 4K | 4 096 | ~30K | Stage 1 LR (high). Both BitLinear and SubQSA active from step 0. |
| **P2** Pretrain @ 32K | 32 768 | ~15K | Continue from P1. RoPE base × 8. |
| **P3** Cooldown @ 32K | 32 768 | ~10K | Stage 2 LR (cooldown), WD=0, curated + math data. |
| **P4** Extend @ 128K | 131 072 | ~8K | RoPE base × 32. Test sliding window sufficiency. |
| **P5** Extend @ 256K | 262 144 | ~5K | First long-context stage. |
| **P6** Extend @ 512K | 524 288 | ~3K | Requires sequence parallelism. |
| **P7** Extend @ 1M | 1 048 576 | ~3K | Production target. |
| **P8** SFT | up to 1M | ~5K | Instruction following + long-doc tasks. Sum-reduction loss. |
| **P9** DPO (or GRPO) | up to 1M | ~5K | Long-context preference pairs targeting retrieval reliability. |

The exact step counts will be tuned to validation loss. The bias is **front-loaded short context** (P1+P2 = 45K steps) where most language modeling happens, then thin slices at each long stage. SubQ's own training recipe matches this shape.

---

## 6. Verification & Ablations

Before declaring the trainer "done":

1. **BitLinear-only baseline** at 2B params, 4K context, 1T tokens — must match BitNet 2B4T paper numbers on PIQA, ARC, BoolQ, GSM8K within noise.
2. **SubQSA-only baseline** at 2B params, 1M context, FP weights — must match NSA paper qualitative behavior + ≥ 99% RULER @ 128K.
3. **Combined (Ultimate)** at 2B params, 1M context, ternary weights:
   - Perplexity within 5% of (1) at 4K.
   - RULER @ 128K within 2 points of (2).
   - NIAH @ 1M ≥ 95% accuracy.
   - End-to-end prefill latency at 1M: ≥ 30× faster than dense BF16 attention (NSA reports ~9× at 64K; SubQ reports 56× at 1M; we aim conservatively).
4. **No-cheat checks**: gate distribution per layer — slc gate should be non-zero at long context; if it collapses to all-window, routing failed.

---

## 7. Risks & Open Questions

1. **Combined stability**. No public paper combines native ternary with NSA-style sparse attention. Risk: gradient flow through the compression branch's MLP `φ` (ternary) may be too noisy for routing. **Mitigation**: keep `φ` as one of the few BF16 layers if stability fails. Decide empirically at small scale (300M params, 4K ctx, 10B tokens).
2. **Selection branch + ternary noise**. Top-k selection on a ternary-projected Q/K may produce volatile selections early. **Mitigation**: the selection signal flows through compression's softmax — which is BF16 — so the noise is bounded.
3. **Routing under quantized activations**. INT8 per-token activations into the compression MLP may lose information needed for routing. **Mitigation**: 2B4T's absmax-per-token is already what NSA needs (peaked scaling). Validate at small scale.
4. **Kernel maturity**. fla-org's NSA Triton kernel is ~1 year old and well-tested for FP; we use it unchanged for SubQSA branches that don't touch BitLinear. The matmul inside `BitLinear` stays standard `F.linear` (with quantized inputs/weights) — no custom kernel needed for training.
5. **Sequence parallelism + selection branch**. Selecting blocks that live on a different rank requires either all-gather of the selection indices (cheap) or sharded gather (more code). NSA paper does not address this explicitly; we'll implement all-gather first.

---

## 8. Completed Work (Summary)

- **SubQSA correctness** — 3-branch design, selection top-k blocks, causal sliding window, GQA-aware compression, unified RoPE.
- **BitLinear parity** — Safe fused/eager dispatch; eval-mode refresh; parity tests.
- **Long-context pipeline** — `train_longctx.py` runs, supports AMP dtype/autocast, saves/resumes checkpoints.
- **Tests & reporting** — 157 tests across 8 test files (~60% coverage).

### Bug Fixes Applied (2026-07-06)

| Severity | Fix | Files |
|----------|-----|-------|
| CRITICAL | Fused Triton kernel guarded with `not self.training` — autograd graph no longer severed on GPU | `ultimate_trainer/bitlinear.py` |
| HIGH | Zero-input NaN prevented in both activation quant functions | Both `bitlinear.py`, `1bit-trainer/model.py` |
| HIGH | Checkpoint resume now trains remaining steps, not extra full run | `train_longctx.py` |
| HIGH | Dataset `StopIteration` caught and iterator recreated | `train_longctx.py` |
| HIGH | Gate normalization epsilon added to prevent NaN | `subqsa_trainer/subqsa.py` |
| HIGH | Dead NSA fused kernel code removed (never called `parallel_nsa`) | `ultimate_trainer/subqsa.py` |
| MEDIUM | `_load_from_state_dict` properly strips non-persistent `_w_ternary` from `missing_keys` | Both `bitlinear.py` |
| MEDIUM | FLOP estimator formulas corrected (selection 20x overcount, compression 2x undercount, GQA overcount) | `benchmark.py` |
| MEDIUM | SelectionBranch probability interpolation fixed (sum-pool replaces `F.interpolate`) | `ultimate_trainer/subqsa.py` |
| MEDIUM | SubQSA sliding window residual fixed (removed `win_out + q`) | `subqsa_trainer/subqsa.py` |
| MEDIUM | CompressionBranch empty fallback now mean-pools all tokens | `ultimate_trainer/subqsa.py` |

### Optimizations Applied (2026-07-06)

| Area | Optimization | Files |
|------|-------------|-------|
| GPU | SelectionBranch FlashAttention (3 kernel launches → 1 fused call) | `ultimate_trainer/subqsa.py` |
| GPU | Sliding window mask caching (`_sw_mask_cache` dict) | `ultimate_trainer/subqsa.py` |
| GPU | RoPE cos/sin table caching (precomputed, indexed by position_ids) | `1bit-trainer/model.py` |
| GPU | ReLU² fusion (`.clamp(min=0).pow(2)` — single fused kernel) | `ultimate_trainer/model.py` |
| Memory | Compression attention at KV-head resolution (avoids 4x expansion) | `ultimate_trainer/subqsa.py` |
| Memory | `_w_ternary` buffers made non-persistent (halves checkpoint size) | Both `bitlinear.py` |
| Memory | Activation checkpointing support (`use_checkpoint` config flag) | `ultimate_trainer/model.py`, `config.py` |
| Memory | FineWebDataset uses `np.memmap` instead of in-memory list | `data_pipeline.py` |
| Memory | FineWebLongCtxDataset buffer uses `array('I')` instead of `list[int]` | `train_longctx.py` |
| Training | Checkpoint resume adjusts `max_steps` | `train_longctx.py` |
| Training | Dataset iterator resurrection on exhaustion | `train_longctx.py` |

## 9. Remaining Open Work

The following are research/model-quality gaps, not missing plumbing:

1. **SubQSA selection quality** — Replace mean-pool compression / magnitude-based top-k with a learned or attention-based importance score so dense-vs-SubQSA cosine reaches ≥ 0.7.
2. **Ultimate model alignment** — Debug why FP vs Ultimate cosine is near zero and Ultimate loss is ~46× FP loss. Suspects: (a) interaction between ternary weights and SubQSA routing, (b) initialization scale, (c) gate normalization under INT8 activations.
3. **Long-context scaling** — Validate multi-stage extension (4K → 1M) on GPU; current smoke test only exercises the 128-token offline path.
4. **SFT / DPO** — Not implemented; deferred until base pretraining quality is achieved.
5. **GPU kernel maturity** — Fused Triton ternary matmul is present but eval-only; needs custom `torch.autograd.Function` wrapper for training use.
6. **RMSNorm fused Triton kernel** — Would fuse pow+mean+sqrt into single kernel launch for ~2x speedup on a common operation.

## 10. Immediate Next Steps (in order)

1. Diagnose and fix SubQSA selection/importance scoring so dense-vs-SubQSA cosine ≥ 0.7.
2. Re-run Ultimate comparison; target FP-vs-Ultimate cosine ≥ 0.5 and Ultimate loss within 2× of FP loss.
3. Write a `torch.autograd.Function` wrapper for the fused Triton kernel to enable training-time use.
4. Validate `train_longctx.py` on a real GPU at 4K context with FineWeb data.
5. Add NIAH / RULER needle tests once model quality targets are met.
6. Scale to 2B / 1T tokens / staged context extension.

---

## 9. References

Primary:
- [RESEARCH_1BIT.md](RESEARCH_1BIT.md) — full BitNet lineage + 2B4T recipe
- [RESEARCH_SUBQSA.md](RESEARCH_SUBQSA.md) — full SubQSA + NSA derivation

Anchor papers:
- [BitNet b1.58 2B4T (Apr 2025)](https://arxiv.org/abs/2504.12285) — reference 1-bit trainer recipe
- [Native Sparse Attention (Feb 2025, ACL Best Paper)](https://arxiv.org/abs/2502.11089) — reference sparse attention design
- [SubQ-1.1-Small Technical Report](https://subq.ai/docs/subq-1-1-small-model-card.pdf) — long-context target behaviour

Reference implementations:
- [microsoft/BitNet](https://github.com/microsoft/BitNet) — official BitLinear, 2B4T weights, inference kernels
- [fla-org/native-sparse-attention](https://github.com/fla-org/native-sparse-attention) — Triton NSA kernel (MIT)
- [lucidrains/native-sparse-attention-pytorch](https://github.com/lucidrains/native-sparse-attention-pytorch) — readable port
