# Research: Native 1-Bit LLMs — BitNet Family

**Date:** 2026-06-30
**Scope:** Native ternary training, follow-up work through 2025-2026, scaling, kernels, and recipes for the Ultimate Trainer.

---

## 1. Lineage Overview

| Paper | Date | Contribution |
|---|---|---|
| **BitNet** (Wang et al., [arXiv:2310.11453](https://arxiv.org/abs/2310.11453)) | Oct 2023 | First native 1-bit Transformer. Binary {-1, +1} weights via `BitLinear`. |
| **BitNet b1.58** (Ma et al., [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)) | Feb 2024 | Ternary {-1, 0, +1} ≈ 1.58 bits. Matches FP16 perplexity at 3B+. |
| **1-bit AI Infra** ([arXiv:2410.16144](https://arxiv.org/abs/2410.16144)) | Oct 2024 | Lossless CPU kernels for b1.58. Foundation of `bitnet.cpp`. |
| **BitNet a4.8** ([arXiv:2411.04965](https://arxiv.org/abs/2411.04965)) | Nov 2024 | 4-bit activations on top of 1.58-bit weights. Hybrid quant+sparsification. |
| **BitNet b1.58 2B4T** ([arXiv:2504.12285](https://arxiv.org/abs/2504.12285)) | Apr 2025 | First **open-source native 1-bit LLM at 2B / 4T tokens**. Matches LLaMA 3.2 1B / Qwen2.5 1.5B. |
| **bitnet.cpp** ([arXiv:2502.11880](https://arxiv.org/abs/2502.11880)) | Feb 2025 | Edge inference kernels. 100B b1.58 on a single CPU at human-reading speed. |
| **BitNet GPU kernel** (Microsoft, May 2025) | May 2025 | First official GPU inference path for ternary weights. |
| **BitNet Distillation** ([arXiv:2510.13998](https://arxiv.org/html/2510.13998v1)) | Oct 2025 | QAT recipe to convert any FP LLM → 1.58-bit with minimal degradation. |

The **2B4T technical report** (Apr 2025) is the definitive reference recipe. It revised the original b1.58 paper's choices in several important ways. Our trainer should track 2B4T, not the 2024 paper.

---

## 2. Architecture: What 2B4T Actually Uses

The 2B4T report (Microsoft Research) commits to a specific configuration. Key deltas vs. the v1 b1.58 paper:

| Component | v1 b1.58 (2024) | **b1.58 2B4T (2025)** | Rationale |
|---|---|---|---|
| FFN activation | SwiGLU | **Squared ReLU (`ReLU²`)** | Sparsity-friendly in low-bit regime; better fit with INT8 activations |
| Normalization | RMSNorm | **subln** (Wang et al., 2022) | More stable in quantized training |
| Activation quant | absmean | **absmax**, per-token, 8-bit | Cleaner outlier handling |
| Position encoding | RoPE | RoPE | Unchanged |
| Biases | None | None | Unchanged |
| Tokenizer | — | **LLaMA 3 BPE, vocab 128 256** | Robust over text + code |
| Inference packing | n/a | **4 ternary values → 1 int8** | Bandwidth-friendly HBM layout |

### 2.1 `BitLinear` (the only layer that matters)

```
Forward:
  γ_w = mean(|W|)                       # scalar per-layer scale
  W̃   = clamp(round(W / (γ_w + ε)), -1, 1)   ∈ {-1, 0, +1}
  s_x = max(|x|, dim=-1, keepdim=True) / 127  # per-token activation scale
  x̃   = clamp(round(x / s_x), -127, 127)      ∈ INT8
  y   = (x̃ @ W̃ᵀ) · s_x · γ_w                  # matmul is ternary × INT8

Backward (STE):
  ∂L/∂W = ∂L/∂W̃
  ∂L/∂x = ∂L/∂x̃
```

Master weights remain FP32/BF16 for gradient accumulation. The quantization functions are wrapped so that forward produces the quantized output while backward passes gradients unmodified (Straight-Through Estimator).

### 2.2 Squared ReLU FFN

```
FFN(x) = down_proj( (ReLU(gate(x)))²  ⊙  up(x) )
```
- All three projections (`gate`, `up`, `down`) are `BitLinear`.
- ReLU² zeros out roughly half of activations naturally — gifting structured sparsity to BitNet a4.8 later.

### 2.3 subln (sub-layer normalization)

Two RMSNorm-like layers inside each sub-block (attention and FFN) — one before the projection and one inside, before the output projection. This dampens activation magnitude before the second BitLinear and is what makes deep ternary training stable.

---

## 3. Training Recipe (b1.58 2B4T)

### 3.1 Two-Stage Pre-training

| Stage | Steps | LR | Weight Decay | Data |
|---|---|---|---|---|
| Stage 1 (high LR) | first ~50% of tokens | cosine, **high peak** (1-bit models tolerate aggressive LR) | cosine peaking at **0.1** | bulk DCLM + general web |
| Stage 2 (cooldown) | second ~50% | abrupt decay → cosine at much lower peak | **0** (off) | FineWeb-EDU + curated + synthetic math |

Total: **4T tokens** for 2B parameters.

### 3.2 Activation-Quantization Warmup

The original paper recommends gradually enabling activation quantization over ~5000 steps. 2B4T does NOT do gradual warmup — it trains with INT8 activations from step 0, relying on subln + the cooldown to stabilize. **Our trainer can choose either**; 2B4T's approach is simpler.

### 3.3 SFT

- Sources: WildChat, LMSYS-Chat-1M, WizardLM Evol-Instruct, SlimOrca + synthetic (GLAN, MathScale).
- **Loss aggregation: SUM, not MEAN** over tokens. Empirically improves convergence for 1-bit.
- Larger LR than FP SFT.
- More epochs than FP SFT.
- Chat template: LLaMA-style `<|begin_of_text|>` / `<|eot_id|>`.

### 3.4 DPO (not RLHF, not PPO/GRPO)

- 2 epochs, LR 2e-7, β = 0.1.
- Datasets: UltraFeedback + MagPie.
- Uses **Liger Kernel** fused ops for efficiency.
- 2B4T explicitly defers PPO/GRPO to future work.

---

## 4. Inference Engineering

### 4.1 GPU Path

Custom CUDA kernel for **W1.58 × A8 matmul**:
- Pack 4 ternary values into 1 INT8 in HBM (2 bits / weight including the unused state).
- Load packed into SRAM, unpack inline, multiply by INT8 activations.
- Reference: Ladder framework + the Microsoft BitNet GPU kernel (May 2025).

### 4.2 CPU Path (`bitnet.cpp`)

- Lossless w.r.t. training math.
- Two quant variants: `i2_s` (raw INT2 storage) and `tl1` (tile-lookup).
- Speedups: 2.37×–6.17× on x86, 71.9%–82.2% energy reduction.
- Jan 2026 update: parallel kernel + configurable tiling + embedding quantization → additional 1.15×–2.1×.

### 4.3 Effective Model Size

Memory ≈ `params × 1.58 / 8 bytes` for the BitLinear portion. Embeddings remain BF16/FP16 (`vocab × hidden × 2 bytes`) and dominate at small scales. At 2B params, the 2B4T model fits in **~0.4 GB non-embedding memory** (Table 1 of the 2B4T report).

---

## 5. Extensions: a4.8 and Q-Sparse

### 5.1 BitNet a4.8 (Nov 2024)

- Continues from a W1.58A8 checkpoint, retrains briefly to **W1.58A4**.
- **Activation analysis**: attention/FFN *inputs* tolerate 4-bit; *intermediate* states (after ReLU²) have heavy outliers → keep them at INT8 but **sparsify** them (Q-Sparse style).
- Sparsify-then-quantize for the attention output projection.
- 55% activated parameters, 3-bit KV cache support.

### 5.2 Q-Sparse

A top-k magnitude mask applied to activations. Used inside a4.8 for the intermediate states. Gives "free" 2× speedup on the down-projection on modern accelerators with sparse-compute kernels.

### 5.3 BitNet Distillation (Oct 2025)

Three-stage QAT to convert an existing FP model:
1. **Refinement** — insert subln modules, swap activation, re-init the heads.
2. **Continue pre-training** — short FP→1.58-bit warm-up on the same data distribution.
3. **MiniLM-style multi-head attention distillation** — student matches teacher's attention queries/keys.

Useful escape hatch when training from scratch is too expensive. Not the focus of the Ultimate Trainer, but worth keeping the architecture compatible.

---

## 6. Scaling Properties

From the 2B4T benchmarks (Table 1 of [arXiv:2504.12285](https://arxiv.org/abs/2504.12285)):

| Model | Memory (non-emb) | CPU TPOT | Avg over 16 benchmarks |
|---|---|---|---|
| LLaMA 3.2 1B | 2 GB | 48 ms | 44.90 |
| Gemma-3 1B | 1.4 GB | 41 ms | 43.74 |
| Qwen2.5 1.5B | 2.6 GB | 65 ms | 55.23 |
| **BitNet b1.58 2B** | **0.4 GB** | **29 ms** | **54.19** |

Native 1-bit at 2B/4T effectively matches Qwen2.5 1.5B while using ~6× less memory and ~2× less latency.

vs. post-training-quantized Qwen2.5 1.5B (Table 2):
- GPTQ-int4 / AWQ-int4: 52.15 / 51.17 average.
- b1.58 2B beats both — *native training* recovers what PTQ loses.

vs. larger PTQ-to-1.58 models (Table 3): b1.58 2B (60.68 avg) beats Llama3-8B-1.58 (49.75) and Falcon3-7B-1.58bit (50.76). **Native trumps post-hoc.**

### Scaling caveat (Nielsen 2024, [arXiv:2407.09527](https://arxiv.org/html/2407.09527))

For small models (<100M), b1.58 needs **roughly 2× the hidden dimension** to match FP. The parity claim is for ≥3B parameters or sufficient hidden width. Plan accordingly when sizing the Ultimate Trainer.

---

## 7. Implementation Status (`1bit_trainer/`)

The current `1bit_trainer/` implements **the v1 b1.58 paper recipe** (absmean activations, SwiGLU, RMSNorm). For the Ultimate Trainer we will upgrade to 2B4T-spec:

- [ ] Switch FFN to squared ReLU
- [ ] Switch normalization to subln (two norms per sub-block)
- [ ] Switch activation quantization to absmax-per-token
- [ ] Two-stage LR (high → cooldown)
- [ ] Two-stage WD (0.1 → 0)
- [ ] LLaMA 3 tokenizer (vocab 128 256)
- [ ] SFT loss with sum-reduction
- [ ] DPO loop with Liger Kernel
- [ ] Pack-store-load-unpack-compute inference kernel (deferred to post-training)

---

## 8. References

- [BitNet (2023)](https://arxiv.org/abs/2310.11453) — original binary 1-bit
- [b1.58 (2024)](https://arxiv.org/abs/2402.17764) — ternary
- [b1.58 2B4T (2025)](https://arxiv.org/abs/2504.12285) — open-weight 2B reference
- [BitNet a4.8 (2024)](https://arxiv.org/abs/2411.04965) — 4-bit activations
- [bitnet.cpp (2025)](https://arxiv.org/abs/2502.11880) — CPU inference
- [BitNet Distillation (2025)](https://arxiv.org/abs/2510.13998) — QAT from FP
- [microsoft/BitNet](https://github.com/microsoft/BitNet) — reference code
- [HF model](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) — weights
- [Nielsen 2024](https://arxiv.org/abs/2407.09527) — SLM scaling, absmedian variant
- [Continual 1.58-bit Pre-training (2025)](https://aclanthology.org/anthology-files/pdf/findings/2025.findings-acl.694.pdf) — 16-bit → 1.58-bit switch point
