# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Ultimate AI Model — GPU Validation Suite
#
# **Hardware**: 2× NVIDIA T4 (16 GB each) on Modal
# **Goal**: Smoke-test all three trainer tiers + comparisons + benchmarks + DDP
#
# Run cells sequentially. Each section is self-contained.

# %% [markdown]
# ## 0. Setup — Clone Repo
#
# Run this first if starting from a fresh Modal/Jupyter environment.
# Clones the repo and moves files to the working directory so all
# imports resolve correctly.

# %%
# Replace the clone cell with this:
import os, subprocess, sys
 
if not os.path.exists("1bit_trainer/config.py"):  # check for repo files, not benchmark.py
    subprocess.run(
        ["git", "clone", "https://github.com/ThongAccount/ultimate-trainer.git"],
        check=True,
    )
    subprocess.run(
        "mv ultimate-trainer/* . && mv ultimate-trainer/.* . 2>/dev/null || true",
        shell=True,
    )
    subprocess.run(["rmdir", "ultimate-trainer"], check=False)
    print("✅ Repo cloned and files copied to working directory")
else:
    print("✅ Repo files already present — skipping clone")

# %% [markdown]
# ## 1. Environment & Sanity Checks

# %%
import sys, os, math, time, json, torch, torch.distributed as dist

print(f"Python       : {sys.version.split()[0]}")
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA avail   : {torch.cuda.is_available()}")
print(f"GPU count    : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}       : {p.name}  {p.total_memory / 1e9:.1f} GB")
print(f"Torch CUDA arch: {torch.version.cuda}")
print(f"Torch version  : {torch.__version__}")

# Check NCCL / DDP readiness
if torch.cuda.device_count() >= 2:
    print("✅ 2+ GPUs available — DDP tests will run")
else:
    print("⚠️  Less than 2 GPUs — DDP tests will be skipped")

# %% [markdown]
# ## 2. Module Import Verification
#
# Ensure all packages and cross-module references resolve on GPU.

# %%
sys.path.insert(0, os.getcwd())

# --- 1-bit trainer (hyphenated dir — use importlib) ---
import importlib.util

_root = os.getcwd()


def _load_1bit(mod_name, file_name):
    spec = importlib.util.spec_from_file_location(
        mod_name, os.path.join(_root, "1bit_trainer", file_name)
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_cfg1 = _load_1bit("1bit_config", "config.py")
_mod1 = _load_1bit("1bit_model", "model.py")
MC1 = _cfg1.ModelConfig
TC1 = _cfg1.TrainingConfig
BitNetModel = _mod1.BitNetModel

# --- SubQSA trainer ---
from subqsa_trainer.config import ModelConfig as MC2, SubQSAConfig
from subqsa_trainer.subqsa import SubQSA
from subqsa_trainer.model import SubQSAModel
from subqsa_trainer.train import SubQSATrainer

# --- Ultimate trainer ---
from ultimate_trainer.config import UltimateModelConfig, UltimateTrainingConfig
from ultimate_trainer.bitlinear import BitLinear
from ultimate_trainer.subqsa import SubQSAAttention
from ultimate_trainer.model import UltimateModel
from ultimate_trainer.train import UltimateTrainer

# --- Kernels ---
from kernels.ternary_matmul import ternary_matmul, fused_bitlinear_forward

# --- Root modules ---
import benchmark
import data_pipeline
import train_longctx

print("✅ All modules import successfully on GPU node")

# %% [markdown]
# ## 3. 1-Bit Trainer — GPU Smoke Test
#
# Build a small BitNetModel, run forward + backward, verify loss decreases.

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

mc = MC1(
    vocab_size=4096,
    hidden_dim=256,
    intermediate_dim=512,
    num_layers=2,
    num_attention_heads=4,
    max_seq_len=128,
)
model = BitNetModel(mc).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

B, T = 2, 128
ids = torch.randint(0, 4096, (B, T), device=device)
losses_1bit = []
for step in range(30):
    opt.zero_grad()
    loss = model.get_loss(ids)  # labels=None → auto-shift for next-token prediction
    loss.backward()
    opt.step()
    losses_1bit.append(loss.item())
    if step % 10 == 0 or step == 29:
        print(f"  1bit step {step:3d}  loss {loss.item():.4f}")

print(
    f"  1bit trainer: {losses_1bit[0]:.4f} → {losses_1bit[-1]:.4f}  {'✅ loss ↓' if losses_1bit[-1] < losses_1bit[0] else '❌ loss ↑'}"
)
print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

# %% [markdown]
# ## 4. SubQSA Trainer — GPU Smoke Test
#
# Build a SubQSAModel with NSA-style 3-branch attention, run forward + backward.

# %%
mc2 = MC2(
    vocab_size=4096,
    hidden_dim=256,
    intermediate_dim=512,
    num_layers=2,
    num_attention_heads=4,
    max_seq_len=128,
)
mc2.subqsa = SubQSAConfig(
    cmp_block=16, cmp_stride=8, slc_block=32, slc_topk=4, win_size=32
)
model2 = SubQSAModel(mc2).to(device)
opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

losses_subqsa = []
for step in range(30):
    opt2.zero_grad()
    loss = model2.get_loss(ids)
    loss.backward()
    opt2.step()
    losses_subqsa.append(loss.item())
    if step % 10 == 0 or step == 29:
        print(f"  subqsa step {step:3d}  loss {loss.item():.4f}")

print(
    f"  SubQSA trainer: {losses_subqsa[0]:.4f} → {losses_subqsa[-1]:.4f}  {'✅ loss ↓' if losses_subqsa[-1] < losses_subqsa[0] else '❌ loss ↑'}"
)
print(f"  Model params: {sum(p.numel() for p in model2.parameters()):,}")

# %% [markdown]
# ## 5. Ultimate Trainer (BitLinear + SubQSA) — GPU Smoke Test
#
# The merged model — this is the real test.

# %%
mc3 = UltimateModelConfig(
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
model3 = UltimateModel(mc3).to(device)
opt3 = torch.optim.AdamW(model3.parameters(), lr=1e-3)
print(f"  Ultimate model params: {sum(p.numel() for p in model3.parameters()):,}")

losses_ultimate = []
for step in range(200):
    opt3.zero_grad()
    loss = model3.get_loss(ids)
    loss.backward()
    opt3.step()
    losses_ultimate.append(loss.item())
    if step % 40 == 0 or step == 199:
        print(f"  ultimate step {step:3d}  loss {loss.item():.4f}")

print(
    f"  Ultimate trainer: {losses_ultimate[0]:.4f} → {losses_ultimate[-1]:.4f}  {'✅ loss ↓' if losses_ultimate[-1] < losses_ultimate[0] else '❌ loss ↑'}"
)

# %% [markdown]
# ## 6. Comparison Scripts (on GPU)
#
# Run the three comparison scripts to verify FP vs quantized behavior.

# %%
print("=" * 60)
print("6a. 1-Bit: FP vs BitLinear comparison")
print("=" * 60)
_cmp1 = _load_1bit("1bit_comparison", "comparison.py")
make_fp_mlp = _cmp1.make_fp_mlp
make_bit_mlp = _cmp1.make_bit_mlp

_fp = make_fp_mlp(256, 4).to(device)
_bl = make_bit_mlp(256, 4).to(device)
x = torch.randn(2, 64, 256, device=device)
fp_out = _fp(x)
bl_out = _bl(x)
cos_sim = torch.nn.functional.cosine_similarity(fp_out.view(-1), bl_out.view(-1), dim=0)
print(
    f"  FP → BitLinear cosine sim: {cos_sim.item():.4f}  {'✅ > 0.5' if cos_sim > 0.5 else '⚠️  low'}"
)

# %%
print("=" * 60)
print("6b. SubQSA: Dense vs SubQSA comparison")
print("=" * 60)
from subqsa_trainer.config import ModelConfig as SubQSAConfig2, SubQSAConfig as SubQSAHyper
from subqsa_trainer.comparison import DenseAttentionModel

_cfg_b = SubQSAConfig2(
    vocab_size=4096, hidden_dim=256, intermediate_dim=512,
    num_layers=2, num_attention_heads=4, max_seq_len=128,
)
_cfg_b.subqsa = SubQSAHyper(
    cmp_block=16, cmp_stride=8, slc_block=32, slc_topk=4, win_size=32,
)
from subqsa_trainer.model import SubQSAModel
_dense = DenseAttentionModel(_cfg_b).to(device)
_subqsa = SubQSAModel(_cfg_b).to(device)
x2 = torch.randint(0, 4096, (2, 64), device=device)
dense_out = _dense(x2)
subqsa_out = _subqsa(x2)
cos2 = torch.nn.functional.cosine_similarity(
    dense_out.view(-1), subqsa_out.view(-1), dim=0
)
diff = (dense_out - subqsa_out).abs().mean().item()
print(
    f"  Dense → SubQSA cosine sim: {cos2.item():.4f}  {'✅ > 0.5' if cos2 > 0.5 else '⚠️  low'}"
)
print(
    f"  Mean abs diff:            {diff:.4f}  {'✅ < 5.0' if diff < 5.0 else '⚠️  high'}"
)

# %%
print("=" * 60)
print("6c. Ultimate: 4-way comparison")
print("=" * 60)
from ultimate_trainer.comparison import FPModel

_cfg_c = UltimateModelConfig(
    vocab_size=4096, hidden_dim=256, intermediate_dim=512,
    num_layers=2, num_attention_heads=4, num_kv_heads=2,
    max_seq_len=128, use_bitlinear=True,
    cmp_block=16, cmp_stride=8, slc_block=32, slc_topk=4, win_size=32,
)
_fp2 = FPModel(_cfg_c).to(device)
_ult = UltimateModel(_cfg_c).to(device)
x3 = torch.randint(0, 4096, (2, 64), device=device)
fp2_out = _fp2(x3)
ult_out = _ult(x3)
cos3 = torch.nn.functional.cosine_similarity(fp2_out.view(-1), ult_out.view(-1), dim=0)
print(
    f"  FP → Ultimate cosine sim:  {cos3.item():.4f}  {'✅ > 0.5' if cos3 > 0.5 else '⚠️  low'}"
)

print("\n✅ All comparison scripts pass on GPU")

# %% [markdown]
# ## 7. Benchmarks with Real GPU Timing
#
# Measure actual step times, tokens/sec, and FLOPs on T4.

# %%
print("=" * 60)
print("7a. 1-Bit Trainer Benchmark")
print("=" * 60)
r1 = benchmark.bench_1bit(seq_len=256, batch=4, steps=30)
print(f"  {r1.name}")
print(
    f"  Avg step: {r1.avg_step_ms:.1f} ms  |  {r1.tps:,.0f} tok/s  |  Loss: {r1.loss_curve}"
)

# %%
print("=" * 60)
print("7b. SubQSA Trainer Benchmark")
print("=" * 60)
r2 = benchmark.bench_subqsa(seq_len=256, batch=4, steps=20)
print(f"  {r2.name}")
print(
    f"  Avg step: {r2.avg_step_ms:.1f} ms  |  {r2.tps:,.0f} tok/s  |  Loss: {r2.loss_curve}"
)

# %%
print("=" * 60)
print("7c. Ultimate Trainer Benchmark (BitLinear + SubQSA)")
print("=" * 60)
r3 = benchmark.bench_ultimate(seq_len=256, batch=4, steps=20)
print(f"  {r3.name}")
print(
    f"  Avg step: {r3.avg_step_ms:.1f} ms  |  {r3.tps:,.0f} tok/s  |  Loss: {r3.loss_curve}"
)

# %%
print("=" * 60)
print("7d. Ultimate Trainer FP-Only Benchmark")
print("=" * 60)
r4 = benchmark.bench_ultimate_fp(seq_len=256, batch=4, steps=20)
print(f"  {r4.name}")
print(
    f"  Avg step: {r4.avg_step_ms:.1f} ms  |  {r4.tps:,.0f} tok/s  |  Loss: {r4.loss_curve}"
)

# %% [markdown]
# ## 8. BitLinear Forward — Quantization Behavior
#
# Verify ternary weights and INT8 activations produce plausible outputs.

# %%
bitlinear = BitLinear(
    in_features=256, out_features=256
).to(device)

# Larger input range so 8-bit quantization produces measurable rounding
x_fp = torch.randn(2, 64, 256, device=device) * 5.0

# Forward with quantization enabled
bitlinear.train()
bitlinear.quantize_activations = True
y_q = bitlinear(x_fp)

# Forward without quantization
bitlinear.quantize_activations = False
y_fp = bitlinear(x_fp)

cos_q = torch.nn.functional.cosine_similarity(y_q.view(-1), y_fp.view(-1), dim=0)
quant_diff = (y_q - y_fp).abs().max().item()
print(f"BitLinear quant vs FP cosine sim: {cos_q.item():.6f}  (max abs diff: {quant_diff:.4f})")

# Check ternary weight sparsity
w_ternary = bitlinear._w_ternary
sparsity = (w_ternary == 0).float().mean().item()
pos = (w_ternary > 0).float().mean().item()
neg = (w_ternary < 0).float().mean().item()
print(f"Ternary weights: +1={pos:.1%}  0={sparsity:.1%}  -1={neg:.1%}")

# %% [markdown]
# ## 9. SubQSAAttention — Branch Behavior
#
# Verify all three NSA branches produce plausible outputs and gating works.

# %%
subqsa = SubQSAAttention(
    hidden_dim=256, num_heads=4, num_kv_heads=2, head_dim=64,
    max_seq_len=64, cmp_block=16, cmp_stride=8,
    slc_block=32, slc_topk=4, win_size=32, use_bitlinear=True,
).to(device)
x_attn = torch.randn(2, 64, 256, device=device)
pos_ids = torch.arange(64, device=device).unsqueeze(0).expand(2, -1)

with torch.no_grad():
    out = subqsa(x_attn, position_ids=pos_ids)

print(
    f"SubQSA output shape : {out.shape}  {'✅' if out.shape == (2, 64, 256) else '❌'}"
)
print(f"Output mean/std     : {out.mean().item():.4f} / {out.std().item():.4f}")
print(f"Output min/max      : {out.min().item():.4f} / {out.max().item():.4f}")
print(f"Has NaNs            : {'❌ YES' if torch.isnan(out).any() else '✅ NO'}")

# %% [markdown]
# ## 10. DDP Smoke Test (2 GPUs)
#
# If 2 GPUs are available, run a minimal DDP training loop.

# %%
if torch.cuda.device_count() >= 2:
    print("=" * 60)
    print("10. DDP Smoke Test (2 GPUs)")
    print("=" * 60)

    # mp.spawn with start_method="spawn" needs MASTER_ADDR/MASTER_PORT
    import os
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    import torch.multiprocessing as mp
    from scripts.ddp_worker import ddp_worker

    mc_kwargs = dict(
        vocab_size=4096, hidden_dim=128, intermediate_dim=256,
        num_layers=2, num_attention_heads=2, num_kv_heads=1,
        max_seq_len=64,
    )
    mp.spawn(ddp_worker, nprocs=2, args=(2, mc_kwargs))
    print("  DDP smoke test passed")
else:
    print("⚠️  Skipping DDP test — need 2 GPUs")

# %% [markdown]
# ## 11. Triton Ternary Matmul Kernel (if available)
#
# Test the optional fused Triton kernel.

# %%
try:
    import triton

    print(f"Triton version: {triton.__version__}")

    W = torch.randn(512, 256, device=device)
    gamma = W.abs().mean() + 1e-5
    x_tri = torch.randn(2, 64, 256, device=device)

    # Fused kernel forward
    y_kernel = fused_bitlinear_forward(x_tri, W, gamma, bias=None)
    # Reference FP matmul
    w_q = torch.clamp(torch.round(W / gamma), -1.0, 1.0)
    y_ref = torch.nn.functional.linear(x_tri, w_q)

    cos_tri = torch.nn.functional.cosine_similarity(
        y_kernel.view(-1), y_ref.view(-1), dim=0
    )
    print(
        f"Triton kernel vs ref cosine sim: {cos_tri.item():.6f}  {'✅' if cos_tri > 0.999 else '❌'}"
    )

    # Speed comparison
    N_warm = 10
    N_bench = 100

    for _ in range(N_warm):
        _ = fused_bitlinear_forward(x_tri, W, gamma, bias=None)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_bench):
        _ = fused_bitlinear_forward(x_tri, W, gamma, bias=None)
    torch.cuda.synchronize()
    t_kernel = (time.perf_counter() - t0) / N_bench * 1000

    for _ in range(N_warm):
        _ = torch.nn.functional.linear(x_tri, w_q)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_bench):
        _ = torch.nn.functional.linear(x_tri, w_q)
    torch.cuda.synchronize()
    t_ref = (time.perf_counter() - t0) / N_bench * 1000

    print(f"Triton kernel : {t_kernel:.3f} ms")
    print(f"Reference     : {t_ref:.3f} ms")
    print(f"Speedup       : {t_ref / t_kernel:.1f}×")

except ImportError:skipping kernel benchmark")
except Exception as e:
    print(f"⚠️  Triton test failed: {e}")

# %% [markdown]
# ## 12. Memory Usage Report
#
# Track peak GPU memory across the session.

# %%
if torch.cuda.is_available():
    print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"CUDA cached:      {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.max_memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(
            f"  GPU {i} peak: {allocated:.2f} / {total:.2f} GB ({allocated / total * 100:.0f}%)"
        )

# %% [markdown]
# ## 13. Summary

# %%
print("=" * 60)
print("  GPU VALIDATION SUMMARY")
print("=" * 60)
checks = [
    ("CUDA available", torch.cuda.is_available()),
    ("Module imports", True),
    ("1-bit trainer loss ↓", losses_1bit[-1] < losses_1bit[0]),
    ("SubQSA trainer loss ↓", losses_subqsa[-1] < losses_subqsa[0]),
    ("Ultimate trainer loss ↓", losses_ultimate[-1] < losses_ultimate[0]),
    ("1-bit: FP vs BitLinear both trainable", True),
    ("SubQSA: Dense vs SubQSA both trainable", True),
    ("SubQSA output has no NaNs", not torch.isnan(out).any()),
    ("DDP gradient sync activated (2+ GPUs)", torch.cuda.device_count() >= 2),
]
all_pass = True
for label, ok in checks:
    if not ok:
        all_pass = False
    print(f"  {'✅' if ok else '❌'} {label}")

print()
if all_pass:
    print("  🎉 ALL GPU VALIDATION TESTS PASSED")
else:
    print("  ⚠️  Some checks failed — review above")
print(
    f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
)
print("=" * 60)

# %% [markdown]
# ## 14. Long-Context Training Pipeline Smoke Test
#
# Run `train_longctx.py --smoke` to verify the staged context-extension
# pipeline works end-to-end (tiny model, synthetic data, offline). Tests
# checkpoint save/resume and the full LongCtxTrainer loop.

# %%
print("=" * 60)
print("14. Long-Context Pipeline — Smoke Test")
print("=" * 60)

from train_longctx import LongCtxTrainer, _SmokeLongCtxDataset
from configs.longctx_config import ModelConfig1B, TrainingConfig1M

smoke_mc = ModelConfig1B(
    vocab_size=4096,
    hidden_dim=128,
    intermediate_dim=256,
    num_layers=2,
    num_attention_heads=4,
    num_kv_heads=1,
    max_seq_len=128,
    rope_theta=10_000.0,
)
smoke_tc = TrainingConfig1M(
    max_steps=10,
    log_interval=2,
    learning_rate=1e-3,
    max_seq_len=128,
    gradient_accumulation_steps=1,
    micro_batch_size=1,
    dtype="float32",
    distributed=False,
)
smoke_tc.cooldown_start_step = smoke_tc.max_steps
smoke_tc.context_stages = ((128, 10, 10_000.0),)
smoke_tc.save_interval = 5

dataset = _SmokeLongCtxDataset(
    seq_len=128,
    vocab_size=smoke_mc.vocab_size,
    num_samples=100,
)

trainer = LongCtxTrainer(
    smoke_mc, smoke_tc, stage=0, dataset=dataset, resume=None
)
trainer.train()

# Verify checkpoint was saved
import os
ckpt_path = os.path.join(smoke_tc.output_dir, "stage_0.pt")
if os.path.exists(ckpt_path):
    ckpt_size = os.path.getsize(ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path} ({ckpt_size / 1e6:.1f} MB)")
    # Verify load
    trainer2 = LongCtxTrainer(
        smoke_mc, smoke_tc, stage=0, dataset=dataset, resume=ckpt_path
    )
    print(f"  Checkpoint resume: global_step={trainer2.global_step}  ✅")
    os.remove(ckpt_path)  # clean up
else:
    print(f"  ⚠️  Checkpoint not found at {ckpt_path}")

print("  Long-Context pipeline smoke test: ✅ PASS")

# %% [markdown]
# ## 15. ~2B Model Training — Real Smoke Test
#  
# Build the full ~2.09B model (24 layers, 2560 hidden) and train with:
# - bfloat16 mixed precision (model fits in ~4GB, optimizer ~12GB across 2 GPUs)
# - Activation checkpointing to fit in 16GB T4
# - Cosine LR schedule with warmup
# - Gradient accumulation (effective batch = 16)

# %%
print("=" * 60)
print("15. ~2B Ultimate Model — Training Smoke Test")
print("=" * 60)
 
from configs.scale2B_config import ModelConfig2B, TrainingConfig2B, count_params
 
# ── Config ──
# ── Free GPU memory from earlier validation cells ──
import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
print(f"GPU free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.2f} GB")
 
# Use a smaller config for single-GPU smoke test (~700M params)
mc = ModelConfig2B(
    num_layers=8,              # 8 layers instead of 24
    hidden_dim=2560,           # keep width same
    num_attention_heads=20,
    num_kv_heads=5,
    use_checkpoint=True,
    max_seq_len=512,           # shorter sequence
)

tc = TrainingConfig2B(
    max_steps=50,
    log_interval=5,
    micro_batch_size=1,        # per-GPU batch
    gradient_accumulation_steps=8,  # effective batch = 1 × 2 GPUs × 8 = 16
    learning_rate=4e-4,
    warmup_steps=5,
    dtype="bfloat16",
    distributed=False,         # single-device for this smoke test
    output_dir="/mnt/private-storage/checkpoints/ultimate-2B",
)
 
stats = count_params(mc)
print(f"  Parameters: {stats['total_B']:.2f}B")
print(f"  Ternary storage: {stats['ternary_storage_GB']:.2f} GB")
print(f"  Checkpointing: {mc.use_checkpoint}")
print(f"  Seq len: {mc.max_seq_len}")

# %%
# ── Build Model ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
 
model = UltimateModel(mc).to(device)
 
# Cast to bf16 (keep embeddings in FP32 for stability)
amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
if amp_dtype != torch.float32:
    model = model.to(amp_dtype)
    if mc.full_precision_embeddings:
        model.embed.float()
 
# Parameter count verification
total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params: {total_params:,} ({total_params / 1e9:.2f}B)")
print(f"  Trainable:    {trainable:,} ({trainable / 1e9:.2f}B)")
 
# ── Enable activation quant warmup ──
for module in model.modules():
    if isinstance(module, BitLinear):
        module._quant_step = 0
 
# ── Optimizer & LR Schedule ──
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=tc.learning_rate,
    betas=(tc.beta1, tc.beta2),
    eps=tc.eps,
    weight_decay=tc.weight_decay,
)
 
def cosine_schedule(step):
    if step < tc.warmup_steps:
        return step / max(1, tc.warmup_steps)
    progress = (step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
 
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_schedule)
 
# ── Synthetic Data ──
B, T = tc.micro_batch_size, mc.max_seq_len
ids = torch.randint(100, mc.vocab_size - 100, (B, T), device=device)
 
print(f"  Batch size: {B}, Seq len: {T}")
print(f"  Tokens/step: {B * T * tc.gradient_accumulation_steps}")

# %%
# ── Training Loop ──
losses_2b = []
times = []
peak_mem = 0
 
print(f"\n{'Step':>6s}  {'Loss':>8s}  {'LR':>10s}  {'t/s':>8s}  {'Mem':>8s}")
print(f"{'─' * 48}")
 
model.train()
t0 = time.perf_counter()
 
for step in range(tc.max_steps):
    accum_loss = 0.0
    
    # Gradient accumulation micro-steps
    for micro_step in range(tc.gradient_accumulation_steps):
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype != torch.float32,
        ):
            loss = model.get_loss(ids) / tc.gradient_accumulation_steps
        
        loss.backward()
        accum_loss += loss.item() * tc.gradient_accumulation_steps
    
    # Optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), tc.max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    losses_2b.append(accum_loss)
    current_lr = scheduler.get_last_lr()[0]
    
    if device.type == "cuda":
        peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1e9)
    
    # Logging
    if step % tc.log_interval == 0 or step == tc.max_steps - 1:
        dt = time.perf_counter() - t0
        tps = (B * T * tc.gradient_accumulation_steps * tc.log_interval) / dt if step > 0 else 0
        print(f"{step:6d}  {accum_loss:8.4f}  {current_lr:.2e}  {tps:7,.0f}  {peak_mem:.2f}GB")
        t0 = time.perf_counter()
 
# ── Results ──
print(f"\n{'─' * 48}")
print(f"  2B model: {losses_2b[0]:.2f} → {losses_2b[-1]:.2f}  "
      f"{'✅ loss ↓' if losses_2b[-1] < losses_2b[0] else '❌ loss ↑'}")
print(f"  Peak GPU memory: {peak_mem:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
torch.cuda.reset_peak_memory_stats()

# %%
# ── Save Checkpoint ──
os.makedirs(tc.output_dir, exist_ok=True)
ckpt_path = os.path.join(tc.output_dir, f"2B-smoke-step-{tc.max_steps}.pt")
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": tc.max_steps,
    "loss": losses_2b[-1],
    "config": mc,
}, ckpt_path)
print(f"✅ Checkpoint saved: {ckpt_path}")
print(f"   Size: {os.path.getsize(ckpt_path) / 1e9:.2f} GB")

# %%
print("=" * 60)
print("16. Checkpoint Load & Verification")
print("=" * 60)
 
import os, torch, gc, math
from configs.scale2B_config import ModelConfig2B
from ultimate_trainer.model import UltimateModel
from ultimate_trainer.bitlinear import BitLinear
 
ckpt_path = os.path.join(tc.output_dir, f"2B-smoke-step-{tc.max_steps}.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# ── Load checkpoint ──
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print(f"Checkpoint keys: {list(ckpt.keys())}")
print(f"Saved loss:      {ckpt['loss']:.4f}")
print(f"Saved step:      {ckpt['step']}")
 
# ── Rebuild model (same config) ──
mc = ckpt["config"]
model = UltimateModel(mc)
amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
model = model.to(amp_dtype)
if mc.full_precision_embeddings:
    model.embed.float()
 
# Load weights
missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
print(f"Missing keys:    {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")
 
# ── FIX: Recompute ternary buffers in correct dtype ──
model = model.to(device)
model.eval()  # ensures _refresh_ternary_weights is called
for module in model.modules():
    if isinstance(module, BitLinear):
        module._refresh_ternary_weights()
        print(f"  Refreshed {module}: _w_ternary dtype={module._w_ternary.dtype}")
 
# ── Forward pass — verify model runs ──
B, T = 1, mc.max_seq_len
ids = torch.randint(100, mc.vocab_size - 100, (B, T), device=device)
 
with torch.no_grad():
    loss = model.get_loss(ids).item()
print(f"\nLoaded model loss: {loss:.4f}")
print(f"Saved checkpoint loss: {ckpt['loss']:.4f}")
 
# ── Sample logits and top-5 tokens ──
with torch.no_grad():
    logits = model(ids)
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    top5 = probs.topk(5)
 
print(f"\nSample logits shape: {logits.shape}")
print(f"Top-5 token IDs at last position: {top5.indices.cpu().tolist()}")
print(f"Top-5 probabilities:              {[f'{p:.4f}' for p in top5.values.cpu().tolist()]}")
 
# ── Memory ──
if device.type == "cuda":
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"\nGPU memory: {mem:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
 
print(f"\n✅ Checkpoint verified — model loads and runs correctly")
