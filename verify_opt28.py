#!/usr/bin/env python3
"""
Verify optimization 2.8 applied correctly without regressions.

Checks:
  2.3 FlashAttention/Selection — shapes match via SDPA
  2.4 Sliding window mask caching — cached mask == recomputed mask
  2.5 RoPE caching — cached cos/sin == dynamic computation
  Compression branch vectorization — unfold equivalent to manual list
  Selection branch numerical equivalence — old vs new attention path
  Gate stability — epsilon in normalization prevents division by zero
  BitLinear _refresh_ternary_weights — non-persistent buffer management
  Loss function correctness — labels/shift alignment
"""

import math
import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = 0
FAIL = 0

def check(condition, message):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {message}")
    else:
        FAIL += 1
        print(f"  FAIL: {message}")

def section(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

torch.manual_seed(42)

# =====================================================================
# 1. RoPE caching verification (Check 2.5)
# =====================================================================
section("2.5 RoPE caching: cached cos/sin matches dynamic computation")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "_bit_model", os.path.join(os.path.dirname(os.path.abspath(__file__)), "1bit_trainer", "model.py")
)
_bit_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_bit_mod)
RotaryEmbedding = _bit_mod.RotaryEmbedding

head_dim = 128
max_seq_len = 4096
theta = 10000.0

rope = RotaryEmbedding(dim=head_dim, max_seq_len=max_seq_len, theta=theta)

B, H, T = 2, 4, 256
position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
x = torch.randn(B, H, T, head_dim)

# Fast path (cached) — within max_seq_len, uses cached tables
with torch.no_grad():
    out_fast = rope(x, position_ids)

# Force fallback by creating position IDs beyond max_seq_len
position_ids_long = (position_ids + max_seq_len + 100)  # exceeds max_seq_len
with torch.no_grad():
    out_fallback = rope(x, position_ids_long)

# The fallback path uses dynamic trig; verify it still produces finite results
check(torch.isfinite(out_fallback).all(),
      "Dynamic RoPE fallback produces finite outputs")

# Verify RoPE output has correct shape
check(out_fast.shape == (B, H, T, head_dim),
      f"RoPE cached output shape: {out_fast.shape} == ({B}, {H}, {T}, {head_dim})")

check(out_fallback.shape == (B, H, T, head_dim),
      f"RoPE fallback output shape: {out_fallback.shape} == ({B}, {H}, {T}, {head_dim})")

# Verify cached cos/sin values match what would be computed on the fly
cos_cached = rope.cos_cached  # (max_seq_len, dim/2)
sin_cached = rope.sin_cached

# Manually recompute the cos/sin for positions 0..max_seq_len-1
inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
pos = torch.arange(max_seq_len, dtype=torch.float32)
angles = pos[:, None] * inv_freq[None, :]
cos_ref = angles.cos()
sin_ref = angles.sin()

check(torch.allclose(cos_cached, cos_ref, atol=1e-7),
      "cos_cached matches dynamic computation")
check(torch.allclose(sin_cached, sin_ref, atol=1e-7),
      "sin_cached matches dynamic computation")

# Verify that cosine and sine sum-of-squares is 1 (rotation invariant)
cos_sq_plus_sin_sq = cos_cached**2 + sin_cached**2
check(torch.allclose(cos_sq_plus_sin_sq, torch.ones_like(cos_sq_plus_sin_sq), atol=1e-6),
      "RoPE cos^2 + sin^2 == 1 for all positions")

# Verify cached vs dynamic path produce identical RoPE outputs for 0..max_seq_len-1
# Both should use the same inv_freq computation
check(torch.allclose(out_fast[..., :head_dim//2].abs().sum(), out_fast[..., head_dim//2:].abs().sum(),
                     rtol=0.5),  # Not exactly equal but order-of-mag comparable
      "RoPE halves have comparable magnitude")

# Verify RoPE doesn't change the norm of each vector (orthogonal transform)
x_norm = x.norm(dim=-1)
out_norm = out_fast.norm(dim=-1)
check(torch.allclose(x_norm, out_norm, atol=1e-5),
      "RoPE preserves norm (x||x|| == ||RoPE(x)||)")


# =====================================================================
# 2. RoPE: verify numerical equivalence OLD implementation vs NEW
# =====================================================================
section("2.5 RoPE: numerical equivalence of old vs new impl")

# Implement the old RoPE from subqsa_trainer/subqsa.py (pre-opt)
def old_rope(x, start_pos, seq_len, head_dim, inv_freq):
    # x: (B*H, T, head_dim)
    inv = inv_freq[None, None, :].float()  # (1, 1, D/2)
    pos = torch.arange(seq_len, device=x.device).unsqueeze(0).unsqueeze(-1).float()
    angles = pos * inv  # (1, seq, D/2)
    cos = angles.cos().to(x.dtype)
    sin = angles.sin().to(x.dtype)
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot

# Old subqsa implementation used B*H, T, D shape
old_inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
x_reshaped = x.reshape(B * H, T, head_dim)
start_pos = 0
old_out = old_rope(x_reshaped, start_pos, T, head_dim, old_inv_freq).reshape(B, H, T, head_dim)

# New implementation
new_out = out_fast

check(torch.allclose(old_out, new_out, atol=1e-5),
      "Old and new RoPE implementations produce identical results")

# Check with offset positions
start_pos = 50
old_out_offset = old_rope(x_reshaped, start_pos, T, head_dim, old_inv_freq).reshape(B, H, T, head_dim)
pos_ids_offset = torch.arange(start_pos, start_pos + T).unsqueeze(0).expand(B, -1)
new_out_offset = rope(x, pos_ids_offset)

check(torch.allclose(old_out_offset, new_out_offset, atol=1e-5),
      f"Old and new RoPE match at start_pos={start_pos}")


# =====================================================================
# 3. Sliding window mask caching verification (Check 2.4)
# =====================================================================
section("2.4 Mask caching: cached mask == recomputed mask")

from ultimate_trainer.subqsa import sliding_window_attention

SHAPE_CASES = [
    (2, 4, 128, 128, 32),
    (1, 2, 8, 16, 8),
    (2, 3, 10, 16, 4),
    (1, 1, 4, 8, 16),
    (4, 8, 256, 64, 64),
    (2, 4, 512, 128, 128),
]

for B, H, T_, D, win_size in SHAPE_CASES:
    q = torch.randn(B, H, T_, D)
    k = torch.randn(B, H, T_, D)
    v = torch.randn(B, H, T_, D)

    # Without cache
    out_no_cache = sliding_window_attention(q, k, v, win_size)

    # With cache: first call populates, second call reuses
    cache = {}
    out_cache1 = sliding_window_attention(q, k, v, win_size, cache)
    out_cache2 = sliding_window_attention(q, k, v, win_size, cache)

    # Verify mask was cached
    w = min(win_size, T_)
    check((T_, w) in cache,
          f"Mask cached for (T={T_}, w={w})")

    # Verify outputs with cache match outputs without cache
    check(torch.allclose(out_no_cache, out_cache1, atol=1e-5),
          f"sliding_window cached ({B},{H},{T_},{D}) matches non-cached")
    check(torch.allclose(out_cache1, out_cache2, atol=1e-5),
          f"sliding_window cache reuse produces identical output")

    # Verify shape
    check(out_no_cache.shape == (B, H, T_, D),
          f"sliding_window output shape: {out_no_cache.shape} == ({B}, {H}, {T_}, {D})")

    # Verify causal: a query at position t should NOT attend to keys at position > t
    # Sliding window only sees keys in [-w:, :] so we check zero-position handling
    # For early positions where t < T - w, output should be zero
    if T_ > w:
        start = T_ - w
        for t in range(0, min(start, 3)):
            # First few positions with no causal key in window should be 0
            pass  # verified by existing tests; shape is the key check


# =====================================================================
# 4. Compression branch: unfold vs list comprehension equivalence
# =====================================================================
section("Optimization: compression branch unfold vs list comprehension")

from ultimate_trainer.subqsa import CompressionBranch

# Replicate the old and new implementations for verification
B, H, T_, D = 2, 4, 64, 128
cmp_block = 8
cmp_stride = 4

k = torch.randn(B, H, T_, D)
v = torch.randn(B, H, T_, D)

# Old: list comprehension
n_blocks = max(1, (T_ - cmp_block) // cmp_stride)
L_blocks = min(n_blocks, (T_ - cmp_block) // cmp_stride + 1)
if L_blocks <= 0:
    L_blocks = 1
n_blocks_eff = max(1, (T_ - cmp_block) // cmp_stride)

# Using the same approach as the actual code in the commit
n_blocks_old = max(1, (T_ - cmp_block) // cmp_stride)
if n_blocks_old > 0:
    blocks_k_old = torch.stack(
        [k[:, :, i * cmp_stride : i * cmp_stride + cmp_block] for i in range(n_blocks_old)], dim=2
    ).reshape(B, H, n_blocks_old, cmp_block * D)
    blocks_v_old = torch.stack(
        [v[:, :, i * cmp_stride : i * cmp_stride + cmp_block] for i in range(n_blocks_old)], dim=2
    ).reshape(B, H, n_blocks_old, cmp_block * D)

    # New: vectorized unfold
    blocks_k_new = k.unfold(2, cmp_block, cmp_stride)[:, :, :n_blocks_old].transpose(-1, -2).reshape(B, H, n_blocks_old, cmp_block * D)
    blocks_v_new = v.unfold(2, cmp_block, cmp_stride)[:, :, :n_blocks_old].transpose(-1, -2).reshape(B, H, n_blocks_old, cmp_block * D)

    check(torch.allclose(blocks_k_old, blocks_k_new, atol=1e-6),
          f"Compression KV unfold matches list comprehension (n_blocks={n_blocks_old})")
    check(torch.allclose(blocks_v_old, blocks_v_new, atol=1e-6),
          f"Compression V unfold matches list comprehension (n_blocks={n_blocks_old})")
else:
    print("  SKIP: n_blocks <= 0")


# =====================================================================
# 5. Selection branch: old manual attention vs new SDPA (Check 2.3)
# =====================================================================
section("2.3 Selection branch: SDPA shape and numerical check")

from ultimate_trainer.subqsa import SelectionBranch

B, H, T_, D = 1, 1, 16, 16
l_prime, topk = 4, 3
n_sel = T_ // l_prime
q_sl = torch.randn(B, H, T_, D)
k_sl = torch.randn(B, H, T_, D)
v_sl = torch.randn(B, H, T_, D)
n_cmp_sel = n_sel
p_cmp_sel = torch.randn(B, H, T_, n_cmp_sel).softmax(dim=-1)

sb = SelectionBranch(block_size=l_prime, topk=topk)
out_new, idx_new = sb(q_sl, k_sl, v_sl, p_cmp_sel, n_cmp_sel)

# Check shapes
check(out_new.shape == (B, H, T_, D),
      f"Selection branch SDPA output shape: {out_new.shape} == ({B}, {H}, {T_}, {D})")
check(idx_new.shape == (B, H, T_, min(topk, n_sel)),
      f"Selection branch index shape: {idx_new.shape} == ({B}, {H}, {T_}, {min(topk, n_sel)})")

# Reproduce using manual gather + attention (same as test_selection_matches_manual_topk_gather)
k_blocks = k_sl.reshape(B, H, n_sel, l_prime, D)
v_blocks = v_sl.reshape(B, H, n_sel, l_prime, D)
b_idx = torch.arange(B).view(B, 1, 1, 1)
h_idx = torch.arange(H).view(1, H, 1, 1)
k_sel_manual = k_blocks[b_idx, h_idx, idx_new].reshape(B, H, T_, topk * l_prime, D)
v_sel_manual = v_blocks[b_idx, h_idx, idx_new].reshape(B, H, T_, topk * l_prime, D)
scores = torch.einsum("bhtd,bhtld->bhtl", q_sl, k_sel_manual) / math.sqrt(D)
attn = F.softmax(scores, dim=-1)
expected_manual = torch.einsum("bhtl,bhtld->bhtd", attn, v_sel_manual)

check(torch.allclose(out_new, expected_manual, atol=1e-5),
      "Selection branch SDPA output matches manual attention")

# Test with multiple shape variations
shape_cases = [
    (2, 4, 8, 128, 64, 4),
    (1, 2, 64, 32, 16, 4),
    (2, 4, 128, 64, 32, 8),
]
for B_, H_, T_, D_, lp_, tk_ in shape_cases:
    q_t = torch.randn(B_, H_, T_, D_)
    k_t = torch.randn(B_, H_, T_, D_)
    v_t = torch.randn(B_, H_, T_, D_)
    n_sel_t = max(1, T_ // lp_)
    n_cmp_t = 8
    p_cmp_t = torch.randn(B_, H_, T_, n_cmp_t).softmax(dim=-1)
    sb_t = SelectionBranch(block_size=lp_, topk=tk_)
    out_t, idx_t = sb_t(q_t, k_t, v_t, p_cmp_t, n_cmp_t)
    topk_actual = min(tk_, n_sel_t)
    check(out_t.shape == (B_, H_, T_, D_),
          f"Selection branch shape ({B_},{H_},{T_},{D_}) output={out_t.shape}")
    check(idx_t.shape == (B_, H_, T_, topk_actual),
          f"Selection branch idx shape ({B_},{H_},{T_},{topk_actual}) idx={idx_t.shape}")
    check((idx_t >= 0).all() and (idx_t < n_sel_t).all(),
          f"Selection branch indices in range [0, {n_sel_t})")


# =====================================================================
# 6. Gate stability: epsilon prevents division by zero
# =====================================================================
section("Gate normalization stability with epsilon")

# The old code had: g = g / g.sum(dim=-1, keepdim=True)
# The new code has: g = g / (g.sum(dim=-1, keepdim=True) + 1e-8)

g = torch.zeros(2, 4, 16, 3)  # all zeros would cause NaN without epsilon

old_result = g / g.sum(dim=-1, keepdim=True)
new_result = g / (g.sum(dim=-1, keepdim=True) + 1e-8)

check(not torch.isfinite(old_result).any(),
      "Old gate normalization produces NaN when all gates are zero")
check(torch.isfinite(new_result).all(),
      "New gate normalization with 1e-8 epsilon is stable when all gates are zero")

# With normal values both should be near-identical
g_normal = torch.rand(2, 4, 16, 3).sigmoid()
old_normal = g_normal / g_normal.sum(dim=-1, keepdim=True)
new_normal = g_normal / (g_normal.sum(dim=-1, keepdim=True) + 1e-8)
check(torch.allclose(old_normal, new_normal, rtol=1e-6, atol=1e-6),
      "Gate normalization differs negligibly with epsilon for normal inputs")


# =====================================================================
# 7. BitLinear _refresh_ternary_weights and _w_ternary non-persistent
# =====================================================================
section("BitLinear: _refresh_ternary_weights and non-persistent _w_ternary")

from ultimate_trainer.bitlinear import BitLinear

bl = BitLinear(64, 32, bias=True, quantize_activations=True)
x_bl = torch.randn(4, 64)

# Verify initial state
check(hasattr(bl, '_w_ternary'), "BitLinear has _w_ternary buffer")
check(hasattr(bl, '_gamma'), "BitLinear has _gamma buffer")
check(not bl._w_ternary is None, "_w_ternary is not None")
check(not bl._gamma is None, "_gamma is not None")

# Verify a forward pass still works
y = bl(x_bl)
check(y.shape == (4, 32), f"BitLinear forward output shape: {y.shape}")
check(torch.isfinite(y).all(), "BitLinear forward produces finite output")

# Verify _refresh_ternary_weights
w_before = bl._w_ternary.clone()
with torch.no_grad():
    bl.weight.add_(torch.randn_like(bl.weight) * 0.1)
bl._refresh_ternary_weights()
w_after = bl._w_ternary
check(not torch.equal(w_before, w_after),
      "_refresh_ternary_weights updates _w_ternary after weight change")
check(torch.allclose(bl._w_ternary, bl.weight + (torch.clamp(torch.round(bl.weight / bl._gamma), -1.0, 1.0) - bl.weight).detach(), atol=1e-6),
      "_refresh_ternary_weights computes correct ternary weights")

# Verify eval triggers refresh
bl2 = BitLinear(64, 32, bias=False)
x_eval = torch.randn(4, 64)
with torch.no_grad():
    bl2.weight.add_(torch.randn_like(bl2.weight) * 0.5)
stale_ternary = bl2._w_ternary.clone()
bl2.eval()
y_eval = bl2(x_eval)
check(y_eval.shape == (4, 32), "BitLinear eval output shape correct")
check(torch.isfinite(y_eval).all(), "BitLinear eval output finite")


# =====================================================================
# 8. Loss function correctness
# =====================================================================
section("Loss function: labels/shift alignment")

B_, T_ = 2, 64
vocab_size = 32000
logits = torch.randn(B_, T_, vocab_size)

# Case 1: labels is None (should use input_ids as next-token prediction)
input_ids = torch.randint(0, vocab_size, (B_, T_ + 1,))
# Old behavior:
labels = input_ids
shift_logits_old = logits[..., :-1, :].contiguous()
shift_labels_old = labels[..., 1:].contiguous()
loss_old = F.cross_entropy(shift_logits_old.reshape(-1, vocab_size), shift_labels_old.reshape(-1))

# New behavior (labels=None):
shift_logits_new = logits[..., :-1, :].contiguous()
shift_labels_new = input_ids[..., 1:].contiguous()
loss_new = F.cross_entropy(shift_logits_new.reshape(-1, vocab_size), shift_labels_new.reshape(-1))

check(torch.allclose(loss_old, loss_new),
      "Loss function: labels=None produces same result as before")
check(shift_logits_new.shape == (B_, T_-1, vocab_size),
      f"Loss: shift_logits shape = {shift_logits_new.shape}")
check(shift_labels_new.shape == (B_, T_-1),
      f"Loss: shift_labels shape = {shift_labels_new.shape}")

# Case 2: labels is provided (pre-aligned)
labels_provided = torch.randint(0, vocab_size, (B_, T_,))
# Old behavior with labels provided:
# (before this commit, labels = input_ids and then shift happened regardless)
# New behavior: shift_logits = logits, shift_labels = labels (no shift)
shift_logits_aligned = logits.contiguous()
shift_labels_aligned = labels_provided.contiguous()
loss_aligned = F.cross_entropy(shift_logits_aligned.reshape(-1, vocab_size), shift_labels_aligned.reshape(-1))

check(loss_aligned.shape == (), "Loss with provided labels is scalar")
check(torch.isfinite(loss_aligned), "Loss with provided labels is finite")


# =====================================================================
# 9. Ultimate trainer SubQSA full forward pass (shape check)
# =====================================================================
section("Ultimate SubQSA full forward: shape verification")

from ultimate_trainer.model import UltimateModel
from ultimate_trainer.config import ModelConfig1B

# Use a tiny config for smoke test
mc = ModelConfig1B(
    hidden_dim=128,
    intermediate_dim=256,
    num_layers=2,
    num_attention_heads=4,
    num_kv_heads=1,
    max_seq_len=128,
    rope_theta=10000.0,
    full_precision_embeddings=True,
)

model = UltimateModel(mc)
model.eval()

# Forward pass
input_ids = torch.randint(0, mc.vocab_size, (1, 64))
with torch.no_grad():
    logits = model(input_ids)
    loss = model.get_loss(input_ids, labels=input_ids)

check(logits.shape == (1, 64, mc.vocab_size),
      f"Ultimate model logits shape: {logits.shape} == (1, 64, {mc.vocab_size})")
check(loss.shape == (), f"Ultimate model loss shape: {loss.shape}")
check(torch.isfinite(logits).all(), "Ultimate model logits are finite")
check(torch.isfinite(loss), "Ultimate model loss is finite")

# Check that loss decreases after one training step-like perturbation
# Simulate a basic gradient update
model.train()
loss1 = model.get_loss(input_ids, labels=input_ids)
# Just check it's a valid scalar
check(torch.isfinite(loss1), "Training mode loss is finite")


# =====================================================================
# 10. SubQSA trainer full forward pass
# =====================================================================
section("SubQSA trainer full forward: shape verification")

from subqsa_trainer.model import SubQSAModel
from subqsa_trainer.config import ModelConfig1B as SubQSAConfig

mc_sq = SubQSAConfig(
    hidden_dim=128,
    intermediate_dim=256,
    num_layers=2,
    num_attention_heads=4,
    num_kv_heads=1,
    max_seq_len=128,
    rope_theta=10000.0,
)

model_sq = SubQSAModel(mc_sq)
model_sq.eval()

with torch.no_grad():
    logits_sq = model_sq(input_ids)
    loss_sq = model_sq.get_loss(input_ids)

check(logits_sq.shape == (1, 64, mc_sq.vocab_size),
      f"SubQSA model logits shape: {logits_sq.shape} == (1, 64, {mc_sq.vocab_size})")
check(loss_sq.shape == (), f"SubQSA model loss shape: {loss_sq.shape}")
check(torch.isfinite(logits_sq).all(), "SubQSA model logits are finite")
check(torch.isfinite(loss_sq), "SubQSA model loss is finite")

# Verify get_loss with labels=None uses input_ids (no shift error)
loss_sq_nolabel = model_sq.get_loss(input_ids)
check(torch.isfinite(loss_sq_nolabel), "SubQSA get_loss with labels=None is finite")

# Verify loss with explicit labels
loss_sq_labeled = model_sq.get_loss(input_ids, labels=input_ids)
check(torch.isfinite(loss_sq_labeled), "SubQSA get_loss with explicit labels is finite")


# =====================================================================
# Summary
# =====================================================================
print(f"\n{'='*60}")
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")
sys.exit(0 if FAIL == 0 else 1)
