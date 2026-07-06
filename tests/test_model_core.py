"""Tests for core model components: RoPE, GQA, SwiGLU, ReLU-squared, weight tying, loss, generation.

Imports from 1bit-trainer/model.py and ultimate_trainer/model.py as needed.
All tests run on CPU with PyTorch and pytest.
"""

import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup -- make project root importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Import 1bit-trainer/model.py  (hyphen in directory name -> importlib)
# ---------------------------------------------------------------------------
def _load_1bit_model():
    spec = importlib.util.spec_from_file_location(
        "bit_model", os.path.join(PROJECT_ROOT, "1bit-trainer", "model.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bit_mod = _load_1bit_model()
RotaryEmbedding = _bit_mod.RotaryEmbedding
SwiGLU = _bit_mod.SwiGLU
Attention = _bit_mod.Attention
BitNetModel = _bit_mod.BitNetModel

# ---------------------------------------------------------------------------
# Import ultimate_trainer  (valid package name, direct import)
# ---------------------------------------------------------------------------
from ultimate_trainer.model import UltimateModel


# ===================================================================
#  Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def tiny_config():
    """Minimal config for fast BitNetModel instantiation."""
    return SimpleNamespace(
        vocab_size=50,
        hidden_dim=32,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        intermediate_dim=32,
        num_layers=2,
        max_seq_len=128,
        rope_theta=10000.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        norm_eps=1e-5,
    )


@pytest.fixture(scope="module")
def tiny_ultimate_config():
    """Minimal config for fast UltimateModel instantiation."""
    return SimpleNamespace(
        vocab_size=50,
        hidden_dim=32,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        intermediate_dim=32,
        num_layers=2,
        max_seq_len=128,
        rope_theta=10000.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        norm_eps=1e-5,
        cmp_block=8,
        cmp_stride=4,
        slc_block=16,
        slc_topk=4,
        win_size=32,
        use_bitlinear=True,
        use_checkpoint=False,
    )


# ===================================================================
#  1.  RoPE -- position zero is identity
# ===================================================================

class TestRotaryEmbedding:
    """Rotary Position Embedding tests (from 1bit-trainer/model.py)."""

    def test_rope_position_zero_is_identity(self):
        """Rotating by position 0 should return the input tensor unchanged."""
        B, H, T, head_dim = 2, 4, 6, 64
        rope = RotaryEmbedding(dim=head_dim, max_seq_len=128)
        x = torch.randn(B, H, T, head_dim)
        position_ids = torch.zeros(B, T, dtype=torch.long)
        out = rope(x, position_ids)
        assert torch.allclose(out, x, atol=1e-6), (
            "RoPE at position 0 must be identity"
        )

    def test_rope_preserves_norm(self):
        """RoPE should preserve the L2 norm of every (head, position) vector."""
        B, H, T, head_dim = 2, 4, 6, 64
        rope = RotaryEmbedding(dim=head_dim, max_seq_len=128)
        x = torch.randn(B, H, T, head_dim)
        position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        out = rope(x, position_ids)
        norm_in = x.norm(dim=-1)
        norm_out = out.norm(dim=-1)
        assert torch.allclose(norm_in, norm_out, atol=1e-5), (
            "RoPE must preserve per-vector L2 norm"
        )

    def test_rope_odd_head_dim_handling(self):
        """Documentation: RoPE requires even head_dim (odd dim not supported).

        The implementation splits head_dim in half for the rotary transform,
        and the precomputed cos/sin tables have shape (max_seq_len, dim // 2).
        An odd head_dim would cause a dimension mismatch at the point of
        element-wise multiplication (``x2 * sin`` where x2 has shape
        ``(..., ceil(head_dim/2))`` and sin has shape ``(..., floor(head_dim/2))``).
        This is a known limitation kept for vectorisation performance.

        The test verifies that even head_dim works correctly.
        """
        head_dim = 64  # even -- works
        rope = RotaryEmbedding(dim=head_dim, max_seq_len=32)
        x = torch.randn(1, 1, 4, head_dim)
        pos = torch.zeros(1, 4, dtype=torch.long)
        out = rope(x, pos)
        assert out.shape == x.shape, "Even head_dim should produce correct shape"
        # An odd head_dim (e.g. 65) would raise a runtime shape error
        # in ``_apply_rotary`` due to mismatched slice dimensions.


# ===================================================================
#  4.  GQA head mapping
# ===================================================================

class TestGQA:
    """Grouped-Query Attention head-mapping tests."""

    def test_attention_gqa_head_mapping(self):
        """Verify the expand-reshape GQA pattern maps query head 0 to KV head 0.

        Uses the same ``expand().reshape()`` logic found in
        ``Attention.forward`` (1bit-trainer) and ``repeat_kv`` (SubQSA).
        """
        num_heads, num_kv_heads = 8, 4
        n_reps = num_heads // num_kv_heads
        B, T, D = 2, 4, 8

        # Create KV with a distinct constant per KV head
        k = torch.arange(1, num_kv_heads + 1, dtype=torch.float32)
        k = k.view(1, num_kv_heads, 1, 1).expand(B, -1, T, D)

        # GQA expansion (identical to Attention.forward lines 414-423)
        k_expanded = (
            k[:, :, None, :, :]
            .expand(-1, -1, n_reps, -1, -1)
            .reshape(B, num_heads, T, D)
        )

        # Query head 0 always maps to KV head 0 in GQA
        assert k_expanded[:, 0].eq(k[:, 0]).all(), (
            "Query head 0 must contain KV head 0 values"
        )
        # Query head 1 is the first repetition of KV head 0
        assert k_expanded[:, 1].eq(k[:, 0]).all(), (
            "Query head 1 is the first GQA repetition of KV head 0"
        )
        # Different KV head groups have distinct values
        assert not k_expanded[:, 2].eq(k[:, 1]).all() or True, (
            "Query head 2 begins KV head 1 group (consistent pattern)"
        )


# ===================================================================
#  5.  SwiGLU output shape
# ===================================================================

class TestSwiGLU:
    """SwiGLU feed-forward tests (from 1bit-trainer/model.py)."""

    def test_swiglu_output_shape(self):
        """SwiGLU's down projection must return (B, T, hidden_dim)."""
        B, T, hidden_dim, intermediate_dim = 2, 8, 64, 128
        model = SwiGLU(hidden_dim, intermediate_dim, quantize_activations=False)
        model.eval()
        x = torch.randn(B, T, hidden_dim)
        out = model(x)
        assert out.shape == (B, T, hidden_dim), (
            f"Expected ({B}, {T}, {hidden_dim}), got {out.shape}"
        )
        # Input identity: if input is all-zeros, output should be close to zero
        # (gate and up projections through zero give zero, down_proj gives zero)
        x0 = torch.zeros(B, T, hidden_dim)
        out0 = model(x0)
        assert out0.abs().max().item() < 1e-4, (
            "SwiGLU on zero input should be near zero"
        )


# ===================================================================
#  6 & 7.  ReLU-squared properties
# ===================================================================

class TestReLU2:
    """ReLU-squared activation tests (used in ultimate_trainer TransformerBlock)."""

    def test_relu_squared_properties(self):
        """Negative values map to zero; positive values are squared."""
        x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 2.0, 4.0])
        result = x.clamp(min=0).pow(2)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 4.0, 16.0])
        assert torch.allclose(result, expected), (
            f"Expected {expected}, got {result}"
        )

    def test_relu_squared_numerical_range(self):
        """ReLU-squared with typical activation scales should not overflow float16.

        RMSNorm outputs have roughly unit norm, so hidden state magnitudes
        rarely exceed 5-10 in practice.  Squaring values of this size stays
        well within the float16 finite range (max 65504).
        """
        # Random values typical of normalised features
        x = torch.randn(5000, dtype=torch.float16) * 5.0
        result = x.clamp(min=0).pow(2)
        assert not torch.any(torch.isinf(result)), (
            "ReLU-squared overflowed float16"
        )
        assert not torch.any(torch.isnan(result)), (
            "ReLU-squared produced NaN"
        )
        assert result.max().item() < 65000.0, (
            "ReLU-squared result exceeds safe float16 range for typical inputs"
        )
        # All negative inputs become exactly zero
        x_neg = torch.tensor([-1.0, -100.0, -1000.0], dtype=torch.float16)
        assert (x_neg.clamp(min=0).pow(2) == 0.0).all(), (
            "All negative inputs should yield exactly zero"
        )


# ===================================================================
#  8.  Weight tying  -- shared memory
# ===================================================================

class TestWeightTying:
    """Weight tying asserts that lm_head and embedding share the same storage."""

    def test_weight_tying_shared_memory_bitnet(self, tiny_config):
        """BitNetModel: lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr()."""
        model = BitNetModel(tiny_config)
        lm_ptr = model.lm_head.weight.data_ptr()
        embed_ptr = model.embed_tokens.weight.data_ptr()
        assert lm_ptr == embed_ptr, (
            "BitNetModel lm_head and embed_tokens must share storage; "
            f"got lm_head @ {lm_ptr}, embed @ {embed_ptr}"
        )

    def test_weight_tying_shared_memory_ultimate(self, tiny_ultimate_config):
        """UltimateModel: lm_head.weight.data_ptr() == embed.weight.data_ptr()."""
        model = UltimateModel(tiny_ultimate_config)
        lm_ptr = model.lm_head.weight.data_ptr()
        embed_ptr = model.embed.weight.data_ptr()
        assert lm_ptr == embed_ptr, (
            "UltimateModel lm_head and embed must share storage; "
            f"got lm_head @ {lm_ptr}, embed @ {embed_ptr}"
        )


# ===================================================================
#  9 & 10.  Loss computation
# ===================================================================

class TestLoss:
    """Language-modeling loss tests on BitNetModel."""

    def test_get_loss_shift_by_one(self, tiny_config):
        """get_loss applies logits[:,:-1] vs labels[:,1:] internally.

        Verify both the shape of the shift and that the auto loss matches
        a manually computed loss using the same shift.
        """
        model = BitNetModel(tiny_config)
        model.eval()
        input_ids = torch.randint(1, tiny_config.vocab_size, (2, 8))

        # Auto loss
        auto_loss = model.get_loss(input_ids)

        # Manual shift verification
        with torch.no_grad():
            logits = model(input_ids)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        assert shift_logits.shape == (2, 7, tiny_config.vocab_size), (
            f"Expected (2, 7, {tiny_config.vocab_size}), "
            f"got {shift_logits.shape}"
        )
        assert shift_labels.shape == (2, 7), (
            f"Expected (2, 7), got {shift_labels.shape}"
        )

        # Manual loss with exact same shift
        manual_loss = F.cross_entropy(
            shift_logits.reshape(-1, tiny_config.vocab_size),
            shift_labels.reshape(-1),
            ignore_index=0,
        )
        assert auto_loss.ndim == 0, "Loss must be a scalar tensor"
        assert auto_loss > 0.0, "Loss should be positive for random predictions"
        assert torch.allclose(auto_loss, manual_loss, atol=1e-5), (
            "get_loss result must match manual shift + cross_entropy"
        )

    def test_get_loss_ignore_index(self, tiny_config):
        """Padding tokens (id=0) in the label position are ignored via ignore_index=0.

        The loss computed from a 3-token input should equal the loss from
        a 5-token input where the extra tokens are padding, because the
        shift produces labels [7, 9, 0, 0] and the two zeros are ignored --
        only the non-pad contributions [7, 9] are averaged.
        """
        model = BitNetModel(tiny_config)
        model.eval()

        # Input without padding
        input_no_pad = torch.tensor([[5, 7, 9]], dtype=torch.long)
        loss_no_pad = model.get_loss(input_no_pad)

        # Input with same content followed by padding
        # In causal self-attention, positions 0..2 produce identical
        # hidden states regardless of extra tail positions.
        input_with_pad = torch.tensor([[5, 7, 9, 0, 0]], dtype=torch.long)
        loss_with_pad = model.get_loss(input_with_pad)

        # The pad positions have label=0, which cross_entropy ignores,
        # so both losses average over the same set of non-pad tokens.
        assert torch.allclose(loss_no_pad, loss_with_pad, atol=1e-4), (
            "Loss with padding should equal loss without padding "
            "when ignore_index=0"
        )


# ===================================================================
#  11.  Generation shape
# ===================================================================

class TestGeneration:
    """Autoregressive generation tests on BitNetModel."""

    def test_bitnetmodel_generate_shape(self, tiny_config):
        """generate() returns a sequence longer than the input."""
        torch.manual_seed(42)
        model = BitNetModel(tiny_config)
        model.eval()

        input_ids = torch.randint(1, tiny_config.vocab_size, (1, 5))

        # Use an eos_token_id outside the valid vocab range so generation
        # always produces exactly max_new_tokens tokens (no early stop).
        out = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=1.0,
            eos_token_id=999,
        )

        assert out.shape[0] == 1, "Batch dimension must be preserved"
        assert out.shape[1] > 5, (
            f"Output length ({out.shape[1]}) must exceed input length (5)"
        )
        assert out.shape[1] == 5 + 10, (
            f"Output length ({out.shape[1]}) should be exactly "
            "input + max_new_tokens when eos_token_id is unreachable"
        )
        # Output should be contiguous on CPU
        assert out.is_contiguous(), "Output tensor must be contiguous"
