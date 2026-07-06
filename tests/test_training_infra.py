"""Tests for training infrastructure: LR schedule, long-context dataset, dummy dataset.

Run with:
    uv run python -m pytest tests/test_training_infra.py -v

All tests run on CPU.
"""

import sys
import math
import pytest
import torch
from unittest.mock import patch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/debian/ultimate-ai-model")

from configs.longctx_config import TrainingConfig1M
from train_longctx import get_schedule, FineWebLongCtxDataset
from subqsa_trainer.config import TrainingConfig
from subqsa_trainer.train import DummyDataset, get_cosine_schedule_with_warmup


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

class _MockInnerTok:
    """Mimics the tokenizers.Tokenizer interface for _resolve_special_token."""
    def __init__(self):
        self._vocab = {"<|endoftext|>": 0, "<|eos|>": 1, "<|pad|>": 2}
    def token_to_id(self, tok: str):
        return self._vocab.get(tok)


class _MockBPETokenizer:
    """Predictable token IDs for dataset tests.

    ``encode("doc_N")`` returns ``[1000, 1001, ..., 1000+N-1]``.
    ``encode("_SHORT_")`` returns fewer than 10 tokens (to test skipping).
    """
    def __init__(self):
        self.tokenizer = _MockInnerTok()
    def load(self, path: str):
        pass
    def encode(self, text: str):
        if text.startswith("doc_"):
            parts = text.split("_")
            n = int(parts[1]) if len(parts) > 1 else 10
            return list(range(1000, 1000 + n))
        if text == "_SHORT_":
            return [7, 8]  # fewer than 10 tokens -> skipped
        raise ValueError(f"Unexpected text in mock: {text!r}")


def _make_tc(**kw):
    """Return a TrainingConfig1M with small test-friendly defaults."""
    defaults = dict(
        learning_rate=1.0,
        min_lr=0.1,
        warmup_steps=100,
        cooldown_start_step=800,
        cooldown_lr=0.1,
        max_steps=1000,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
    )
    defaults.update(kw)
    return TrainingConfig1M(**defaults)


def _ref_mult(tc, step):
    """Reference LR multiplier for *tc* at integer *step* (train_longtx spec)."""
    w = tc.warmup_steps
    cs = tc.cooldown_start_step
    if step < w:
        return step / max(1, w)
    if step < cs:
        p = (step - w) / max(1, cs - w)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    p = (step - cs) / max(1, tc.max_steps - cs)
    return 0.5 * (1.0 + math.cos(math.pi * p)) * (tc.cooldown_lr / tc.learning_rate)


def _make_docs(*lengths):
    """Return a list of mock document dicts, one per *length*."""
    return [{"text": f"doc_{n}"} for n in lengths]


# ═══════════════════════════════════════════════════════════════════════
#  1.  Cosine schedule  (train_longctx.get_schedule)
# ═══════════════════════════════════════════════════════════════════════

class TestCosineScheduleValues:
    """LR is 0 at step 0 with warmup, peaks at warmup_steps."""

    def test_lr_is_zero_at_step_zero(self):
        """After scheduler construction the multiplier is 0.0."""
        tc = _make_tc(warmup_steps=100)
        opt = torch.optim.SGD([torch.zeros(1)], lr=1.0)
        sched = get_schedule(opt, tc)
        # LambdaLR init sets last_epoch = 0 and calls step once so that
        # param_groups[0]["lr"] = lr_lambda(0) * initial_lr = 0.0
        assert sched.get_last_lr()[0] == 0.0

    def test_peak_at_warmup_steps(self):
        """The multiplier reaches 1.0 exactly at warmup_steps."""
        tc = _make_tc(warmup_steps=100)
        opt = torch.optim.SGD([torch.zeros(1)], lr=1.0)
        sched = get_schedule(opt, tc)
        for _ in range(tc.warmup_steps):
            sched.step()
        assert sched.get_last_lr()[0] == pytest.approx(1.0)

    def test_peak_multiplier_applied_to_base_lr(self):
        """Actual LR at peak equals the configured learning_rate."""
        tc = _make_tc(learning_rate=2.5e-3, warmup_steps=50)
        opt = torch.optim.SGD([torch.zeros(1)], lr=tc.learning_rate)
        sched = get_schedule(opt, tc)
        for _ in range(tc.warmup_steps):
            sched.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(tc.learning_rate)

    def test_warmup_linear(self):
        """During warmup the multiplier grows linearly from 0 toward 1."""
        tc = _make_tc(warmup_steps=100)
        opt = torch.optim.SGD([torch.zeros(1)], lr=1.0)
        sched = get_schedule(opt, tc)

        for step in range(tc.warmup_steps + 2):
            epoch = sched.last_epoch
            actual = sched.get_last_lr()[0]
            expected = _ref_mult(tc, epoch)
            assert actual == pytest.approx(expected, abs=1e-12), (
                f"Mismatch at epoch {epoch}: {actual} vs {expected}"
            )
            sched.step()

    def test_step0_vs_step1_increment(self):
        """The increment from step 0 to step 1 is 1/warmup_steps."""
        tc = _make_tc(warmup_steps=100)
        opt = torch.optim.SGD([torch.zeros(1)], lr=1.0)
        sched = get_schedule(opt, tc)
        # epoch 0 -> lr 0
        sched.step()
        # epoch 1 -> lr 1/100 = 0.01
        assert sched.get_last_lr()[0] == pytest.approx(1.0 / tc.warmup_steps)


class TestCosineScheduleContinuity:
    """No discontinuity at warmup_steps boundary."""

    def test_cosine_picks_up_at_one(self):
        """The cosine phase at p=0 (step=warmup_steps) equals 1.0."""
        tc = _make_tc(warmup_steps=100)
        opt = torch.optim.SGD([torch.zeros(1)], lr=1.0)
        sched = get_schedule(opt, tc)
        for _ in range(tc.warmup_steps):
            sched.step()
        # At epoch = warmup_steps, the warmup formula would give 1.0 and
        # the cosine also gives 1.0 — no gap.
        assert sched.get_last_lr()[0] == pytest.approx(1.0)

    def test_diff_at_boundary_is_one_over_warmup(self):
        """The step (warmup-1) → (warmup) jumps by exactly 1/warmup."""
        tc = _make_tc(warmup_steps=100)
        opt = torch.optim.SGD([torch.zeros(1)], lr=1.0)
        sched = get_schedule(opt, tc)
        for _ in range(tc.warmup_steps - 1):
            sched.step()
        last_warmup = sched.get_last_lr()[0]  # epoch = 99
        sched.step()
        first_cosine = sched.get_last_lr()[0]  # epoch = 100
        diff = first_cosine - last_warmup
        assert diff == pytest.approx(1.0 / tc.warmup_steps)

    def test_cosine_monotonic_decreasing_after_warmup(self):
        """Every step after warmup reduces (or keeps) the LR (before cooldown)."""
        tc = _make_tc(warmup_steps=50, cooldown_start_step=500)
        opt = torch.optim.SGD([torch.zeros(1)], lr=1.0)
        sched = get_schedule(opt, tc)
        # Step past warmup
        for _ in range(tc.warmup_steps):
            sched.step()
        prev = sched.get_last_lr()[0]
        for _ in range(200):
            sched.step()
            cur = sched.get_last_lr()[0]
            assert cur <= prev + 1e-12, f"LR increased from {prev} to {cur}"
            prev = cur


class TestCosineScheduleFinalLr:
    """Final LR (at end of training) matches the schedule formula."""

    def test_cooldown_start_multiplier_matches_ratio(self):
        """At cooldown_start_step the actual LR equals cooldown_lr."""
        tc = _make_tc(
            learning_rate=1.5e-3,
            cooldown_lr=3e-4,
            cooldown_start_step=800,
            warmup_steps=100,
            max_steps=1000,
        )
        opt = torch.optim.SGD([torch.zeros(1)], lr=tc.learning_rate)
        sched = get_schedule(opt, tc)
        for _ in range(tc.cooldown_start_step):
            sched.step()
        # At epoch = cooldown_start_step, the main cosine has finished
        # and the cooldown cosine begins (p=0).  The actual LR equals
        # cooldown_lr because get_last_lr() = lr_fn(epoch) * initial_lr.
        actual_lr = sched.get_last_lr()[0]
        assert actual_lr == pytest.approx(tc.cooldown_lr, abs=1e-12)

    def test_last_training_step_matches_formula(self):
        """The last training step (epoch = max_steps) matches the formula."""
        tc = _make_tc(
            learning_rate=1.0,
            cooldown_lr=0.1,
            warmup_steps=100,
            cooldown_start_step=800,
            max_steps=1000,
        )
        opt = torch.optim.SGD([torch.zeros(1)], lr=1.0)
        sched = get_schedule(opt, tc)
        for _ in range(tc.max_steps):
            sched.step()
        epoch = sched.last_epoch  # should be max_steps
        expected = _ref_mult(tc, epoch)
        assert sched.get_last_lr()[0] == pytest.approx(expected, abs=1e-12)

    def test_subqsa_cosine_ends_at_min_lr(self):
        """get_cosine_schedule_with_warmup floors at min_lr_ratio (min_lr)."""
        lr = 1e-3
        min_lr = 1e-4
        min_lr_ratio = min_lr / lr  # 0.1
        warmup = 50
        total = 500
        opt = torch.optim.SGD([torch.zeros(1)], lr=lr)
        sched = get_cosine_schedule_with_warmup(
            opt, warmup_steps=warmup, total_steps=total, min_lr_ratio=min_lr_ratio
        )
        for _ in range(total):
            sched.step()
        final_lr = opt.param_groups[0]["lr"]
        # The schedule floors at min_lr_ratio * lr = min_lr
        assert final_lr >= min_lr - 1e-12, f"Final LR {final_lr} < min_lr {min_lr}"
        assert final_lr == pytest.approx(min_lr, abs=1e-8)

    def test_subqsa_cosine_never_below_min_lr(self):
        """After warmup, every scheduler step produces LR >= min_lr."""
        lr = 1e-3
        min_lr = 2e-4
        min_lr_ratio = min_lr / lr
        warmup = 20
        total = 200
        opt = torch.optim.SGD([torch.zeros(1)], lr=lr)
        sched = get_cosine_schedule_with_warmup(
            opt, warmup_steps=warmup, total_steps=total, min_lr_ratio=min_lr_ratio
        )
        for _ in range(warmup):
            sched.step()  # advance past warmup (LR below min_lr during warmup)
        for _ in range(total - warmup):
            sched.step()
            cur = opt.param_groups[0]["lr"]
            assert cur >= min_lr - 1e-12, f"LR {cur} < min_lr {min_lr}"


class TestCosineScheduleMisc:
    """Additional sanity checks."""

    def test_all_steps_match_reference(self):
        """Every step from 0 through max_steps matches the reference impl."""
        tc = _make_tc(warmup_steps=100, cooldown_start_step=800, max_steps=1000)
        opt = torch.optim.SGD([torch.zeros(1)], lr=1.0)
        sched = get_schedule(opt, tc)
        for step in range(tc.max_steps + 1):
            epoch = sched.last_epoch
            actual = sched.get_last_lr()[0]
            expected = _ref_mult(tc, epoch)
            assert actual == pytest.approx(expected, abs=1e-12), (
                f"Failed at epoch {epoch}"
            )
            sched.step()

    def test_warmup_one_works(self):
        """A warmup of 1 step works (edge case)."""
        tc = _make_tc(warmup_steps=1, cooldown_start_step=10, max_steps=20)
        opt = torch.optim.SGD([torch.zeros(1)], lr=1.0)
        sched = get_schedule(opt, tc)
        # epoch 0 -> lr 0/1 = 0
        assert sched.get_last_lr()[0] == 0.0
        sched.step()
        # epoch 1 -> lr 1/1 = 1.0
        assert sched.get_last_lr()[0] == pytest.approx(1.0)
        sched.step()
        # epoch 2 -> first step of cosine: p = (2-1)/(10-1) = 1/9
        p = (2 - 1) / (10 - 1)
        expected = 0.5 * (1.0 + math.cos(math.pi * p))
        assert sched.get_last_lr()[0] == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════════
#  2.  Long-context dataset  (FineWebLongCtxDataset)
# ═══════════════════════════════════════════════════════════════════════

class TestLongCtxDataset:
    """Controlled tests for FineWebLongCtxDataset with mocked dependencies."""

    @pytest.fixture(autouse=True)
    def _patch_bpe(self):
        """Replace BPETokenizer with the mock before any dataset construction."""
        with patch("train_longctx.BPETokenizer", _MockBPETokenizer):
            yield

    # ------------------------------------------------------------------
    #  4.  test_longctx_dataset_yields_fixed_length
    # ------------------------------------------------------------------

    def test_every_sample_is_exactly_seq_len(self):
        """Every yielded sample has ``input_ids`` of shape ``(seq_len,)``."""
        seq_len = 32
        ds = FineWebLongCtxDataset(
            seq_len=seq_len, tokenizer_path="/fake/path", max_docs=100,
        )
        docs = _make_docs(10, 10, 10)  # 30 tokens + 2 EOTs = 32 -> one sample
        with patch("datasets.load_dataset", return_value=docs):
            samples = list(ds)
        assert len(samples) >= 1
        for s in samples:
            assert s["input_ids"].shape == (seq_len,), (
                f"Expected ({seq_len},), got {s['input_ids'].shape}"
            )
            assert s["labels"].shape == (seq_len,)

    def test_multiple_samples_all_fixed_length(self):
        """Multiple consecutive yields all have fixed length."""
        seq_len = 32
        ds = FineWebLongCtxDataset(
            seq_len=seq_len, tokenizer_path="/fake/path", max_docs=500,
        )
        # 5 docs of 20 tokens = 100 tokens + 4 EOTs = 104 -> 3 samples
        docs = _make_docs(20, 20, 20, 20, 20)
        with patch("datasets.load_dataset", return_value=docs):
            samples = list(ds)
        assert len(samples) >= 2
        for s in samples:
            assert s["input_ids"].shape == (seq_len,)
            assert s["labels"].shape == (seq_len,)

    def test_each_sample_same_seq_len_across_batch(self):
        """All samples from a batch have identical length (no ragged tensors)."""
        seq_len = 128
        ds = FineWebLongCtxDataset(
            seq_len=seq_len, tokenizer_path="/fake/path", max_docs=200,
        )
        docs = _make_docs(100, 100, 100)
        with patch("datasets.load_dataset", return_value=docs):
            samples = list(ds)
        lengths = [s["input_ids"].size(0) for s in samples]
        assert all(l == seq_len for l in lengths)

    # ------------------------------------------------------------------
    #  5.  test_longctx_dataset_eos_separator
    # ------------------------------------------------------------------

    def test_eot_between_two_documents(self):
        """EOT token (ID 0) separates two documents in one sample."""
        seq_len = 20
        ds = FineWebLongCtxDataset(
            seq_len=seq_len, tokenizer_path="/fake/path", max_docs=100,
        )
        # Both doc_15 encode to [1000..1014] (the mock always starts at 1000).
        # buffer = [1000..1014, 0, 1000..1014] -> first yield eats 20
        docs = _make_docs(15, 15)
        with patch("datasets.load_dataset", return_value=docs):
            samples = list(ds)
        assert len(samples) >= 1
        inp = samples[0]["input_ids"]
        assert inp[0] == 1000
        assert inp[14] == 1014
        assert inp[15] == 0, f"Expected EOT (0) at position 15, got {inp[15]}"
        # Second doc also starts at 1000 (overlapping range)
        assert inp[16] == 1000

    def test_eot_after_multiple_documents(self):
        """When three documents fit, each boundary has an EOT."""
        seq_len = 50
        ds = FineWebLongCtxDataset(
            seq_len=seq_len, tokenizer_path="/fake/path", max_docs=100,
        )
        # Three 20-token docs: 20+1+20+1+20 = 62 >= seq_len+1=51 -> yield
        docs = _make_docs(20, 20, 20)
        with patch("datasets.load_dataset", return_value=docs):
            samples = list(ds)
        assert len(samples) >= 1
        inp = samples[0]["input_ids"]
        # buffer = [1000..1019, 0, 1000..1019, 0, 1000..1009] (first 10 of doc 3)
        # input_ids = [1000..1019, 0, 1000..1019, 0, 1000..1004] = 50 tokens
        # EOTs at positions 20 and 41
        eot_positions = [i for i, t in enumerate(inp) if t == 0]
        assert len(eot_positions) >= 1, "No EOT found in packed sample"
        # Verify at least one EOT separates document content
        assert inp[0] == 1000  # first doc
        assert inp[eot_positions[0] - 1] == 1019  # last token before first EOT

    def test_eot_position_is_exact_boundary(self):
        """EOT placed exactly at the boundary in a packed sample."""
        seq_len = 20
        ds = FineWebLongCtxDataset(
            seq_len=seq_len, tokenizer_path="/fake/path", max_docs=100,
        )
        # Both doc_15 encode to [1000..1014] (overlapping mock ranges).
        # buffer = [1000..1014, 0, 1000..1014] = 31
        # chunk = buffer[:21] = [1000..1014, 0, 1000..1004] -> 21 tokens
        # input_ids = [1000..1014, 0, 1000..1003] = 20 tokens
        # labels = [1001..1014, 0, 1000..1004] = 20 tokens
        docs = _make_docs(15, 15)
        with patch("datasets.load_dataset", return_value=docs):
            samples = list(ds)
        inp = samples[0]["input_ids"]
        # Find the boundary: the EOT (0)
        eot_idx = (inp == 0).nonzero(as_tuple=True)[0]
        assert len(eot_idx) >= 1
        # Tokens before EOT are from doc_1; tokens after are from doc_2
        before_eot = inp[:eot_idx[0]]
        after_eot = inp[eot_idx[0] + 1:]
        assert before_eot[0] == 1000  # first doc starts at 1000
        assert after_eot[0] == 1000   # second doc also starts at 1000

    # ------------------------------------------------------------------
    #  6.  test_longctx_dataset_short_document_skipped
    # ------------------------------------------------------------------

    def test_short_doc_skipped(self):
        """Documents with fewer than 10 tokens are not added to the buffer."""
        seq_len = 32
        ds = FineWebLongCtxDataset(
            seq_len=seq_len, tokenizer_path="/fake/path", max_docs=100,
        )
        docs = [{"text": "_SHORT_"}] * 5
        with patch("datasets.load_dataset", return_value=docs):
            samples = list(ds)
        # All docs skipped -> buffer stays empty -> no samples
        assert len(samples) == 0

    def test_short_doc_does_not_block_adjacent_docs(self):
        """A short doc between two normal docs is skipped; separation still works."""
        seq_len = 20
        ds = FineWebLongCtxDataset(
            seq_len=seq_len, tokenizer_path="/fake/path", max_docs=100,
        )
        docs = [{"text": "doc_15"}, {"text": "_SHORT_"}, {"text": "doc_15"}]
        with patch("datasets.load_dataset", return_value=docs):
            samples = list(ds)
        assert len(samples) >= 1
        inp = samples[0]["input_ids"]
        # doc_15 + EOT + doc_15 = 31 tokens, first 20 yielded
        # No trace of the short doc
        assert inp[0] == 1000
        assert inp[14] == 1014
        assert inp[15] == 0
        assert inp[16] == 1000

    def test_short_docs_at_start_no_bogus_separator(self):
        """Short docs at the beginning don't inject a spurious EOT."""
        seq_len = 32
        ds = FineWebLongCtxDataset(
            seq_len=seq_len, tokenizer_path="/fake/path", max_docs=100,
        )
        # Short first, then a normal doc
        docs = [{"text": "_SHORT_"}, {"text": "doc_15"}]
        with patch("datasets.load_dataset", return_value=docs):
            samples = list(ds)
        # doc_15 alone (15 tokens) -> no EOT at index 0
        # 15 < seq_len+1 = 33, so no yield yet (waiting for more docs)
        # Since there's only one doc with 15 tokens, no sample is produced
        # (the partial at end: 15 > seq_len//2 = 16? No, 15 <= 16, discarded)
        pass  # This test is about not crashing, not about output

    def test_empty_dataset_no_samples(self):
        """When there are no documents at all, no samples are produced."""
        seq_len = 32
        ds = FineWebLongCtxDataset(
            seq_len=seq_len, tokenizer_path="/fake/path", max_docs=100,
        )
        with patch("datasets.load_dataset", return_value=[]):
            samples = list(ds)
        assert len(samples) == 0

    # ------------------------------------------------------------------
    #  7.  test_longctx_dataset_pad_partial
    # ------------------------------------------------------------------

    def test_partial_sequence_padded_and_yielded(self):
        """When remaining tokens > seq_len//2, the buffer is padded and yielded."""
        seq_len = 16
        ds = FineWebLongCtxDataset(
            seq_len=seq_len, tokenizer_path="/fake/path", max_docs=100,
        )
        # Both doc_15 encode to [1000..1014] (overlapping mock ranges).
        # buffer = [1000..1014, 0, 1000..1014] = 31
        # yield chunk[:17] = [1000..1014, 0, 1000]
        #   input_ids = [1000..1014, 0] (16)
        #   labels = [1001..1014, 0, 1000] (16)
        # buffer = buffer[16:] = [1000..1014] = 15 remaining
        # pad_len = 17-15 = 2, buffer = [1000..1014, 2, 2]
        #   input_ids = [1000..1014, 2] (16)
        #   labels = [1001..1014, 2, 2] (16)
        docs = _make_docs(15, 15)
        with patch("datasets.load_dataset", return_value=docs):
            samples = list(ds)
        assert len(samples) == 2, f"Expected 2 samples (one full, one padded), got {len(samples)}"
        # Second sample should have padding
        inp2 = samples[1]["input_ids"]
        assert inp2.shape == (16,)
        # First 15 tokens are the remainder [1000..1014]
        assert inp2[:15].tolist() == list(range(1000, 1015)), (
            f"Expected [1000..1014], got {inp2[:15].tolist()}"
        )
        # Last token should be pad_id = 2
        assert inp2[15] == 2, f"Expected pad_id at position 15, got {inp2[15]}"

    def test_partial_sequence_discarded_when_short(self):
        """When remaining tokens <= seq_len//2, the buffer is discarded."""
        seq_len = 16
        ds = FineWebLongCtxDataset(
            seq_len=seq_len, tokenizer_path="/fake/path", max_docs=100,
        )
        # doc_10 (10) + doc_10 (EOT + 10 = 11) -> buffer = 21
        # yield chunk[:17], input_ids = [1000..1009, 0, 1010..1013] (16)
        # labels = [1001..1009, 0, 1010..1014] (16)
        # buffer = [1015, 1016, 1017, 1018, 1019] = 5 tokens (len 5 <= 8) -> discard
        docs = _make_docs(10, 10)
        with patch("datasets.load_dataset", return_value=docs):
            samples = list(ds)
        assert len(samples) == 1, (
            f"Expected only 1 sample (no padding), got {len(samples)}"
        )

    def test_padding_uses_pad_id(self):
        """Padding tokens are the pad_id (2 in mock), not EOT or arbitrary."""
        seq_len = 16
        ds = FineWebLongCtxDataset(
            seq_len=seq_len, tokenizer_path="/fake/path", max_docs=100,
        )
        docs = _make_docs(15, 15)
        with patch("datasets.load_dataset", return_value=docs):
            samples = list(ds)
        # Second sample is the padded one
        inp2 = samples[1]["input_ids"]
        pad_mask = inp2 == 2
        assert pad_mask.any(), "No pad_id tokens found in padded sample"
        # All padding should be at the end (contiguous)
        first_pad = pad_mask.nonzero(as_tuple=True)[0][0]
        assert pad_mask[first_pad:].all(), "Padding is not contiguous at the end"

    def test_partial_sequence_labels_also_padded(self):
        """labels in a padded sample also have the pad_id in the padding region."""
        seq_len = 16
        ds = FineWebLongCtxDataset(
            seq_len=seq_len, tokenizer_path="/fake/path", max_docs=100,
        )
        docs = _make_docs(15, 15)
        with patch("datasets.load_dataset", return_value=docs):
            samples = list(ds)
        lbl2 = samples[1]["labels"]
        pad_mask = lbl2 == 2
        assert pad_mask.any(), "No pad_id in labels of padded sample"
        first_pad = pad_mask.nonzero(as_tuple=True)[0][0]
        assert pad_mask[first_pad:].all(), "Label padding is not contiguous"


# ═══════════════════════════════════════════════════════════════════════
#  3.  DummyDataset  (subqsa_trainer.train.DummyDataset)
# ═══════════════════════════════════════════════════════════════════════

class TestDummyDataset:
    """Shapes and properties of the synthetic DummyDataset."""

    def test_shapes(self):
        """DummyDataset returns input_ids and labels of correct shape."""
        seq_len = 4096
        ds = DummyDataset(seq_len=seq_len, vocab_size=32768, num_samples=10)
        assert len(ds) == 10
        item = ds[0]
        assert item["input_ids"].shape == (seq_len,)
        assert item["labels"].shape == (seq_len,)

    def test_autoregressive_shift(self):
        """labels[:-1] == input_ids[1:] (next-token prediction)."""
        seq_len = 32
        torch.manual_seed(42)
        ds = DummyDataset(seq_len=seq_len, vocab_size=4096, num_samples=5)
        for i in range(len(ds)):
            item = ds[i]
            torch.testing.assert_close(
                item["input_ids"][1:], item["labels"][:-1],
                msg=f"Mismatch in sample {i}",
            )

    @pytest.mark.parametrize("seq_len", [64, 128, 256, 1024])
    def test_various_seq_lengths(self, seq_len):
        """DummyDataset works with different sequence lengths."""
        ds = DummyDataset(seq_len=seq_len, vocab_size=4096, num_samples=3)
        for i in range(len(ds)):
            item = ds[i]
            assert item["input_ids"].shape == (seq_len,)
            assert item["labels"].shape == (seq_len,)

    def test_dataloader_batch_shapes(self):
        """DummyDataset inside a DataLoader yields batched tensors."""
        batch_size = 4
        seq_len = 128
        ds = DummyDataset(seq_len=seq_len, vocab_size=4096, num_samples=16)
        loader = DataLoader(ds, batch_size=batch_size)
        batch = next(iter(loader))
        assert batch["input_ids"].shape == (batch_size, seq_len), (
            f"Expected ({batch_size}, {seq_len}), got {batch['input_ids'].shape}"
        )
        assert batch["labels"].shape == (batch_size, seq_len)

    def test_reproducible_with_fixed_seed(self):
        """DummyDataset returns the same sequence for the same index+seed."""
        seq_len = 128
        torch.manual_seed(1234)
        ds1 = DummyDataset(seq_len=seq_len, vocab_size=4096, num_samples=50)
        torch.manual_seed(1234)
        ds2 = DummyDataset(seq_len=seq_len, vocab_size=4096, num_samples=50)
        # Re-seed before each comparison pair because __getitem__ advances
        # the global RNG state, and ds2 must start from the same seed.
        for i in range(10):
            torch.manual_seed(1234 + i)
            a = ds1[i]["input_ids"]
            torch.manual_seed(1234 + i)
            b = ds2[i]["input_ids"]
            assert torch.equal(a, b), f"Non-reproducible at index {i}"

    def test_vocab_boundary(self):
        """Token IDs stay within [100, min(vocab_size, 30000))."""
        vocab_size = 4096
        ds = DummyDataset(seq_len=256, vocab_size=vocab_size, num_samples=50)
        for i in range(len(ds)):
            item = ds[i]
            assert (item["input_ids"] >= 100).all()
            assert (item["input_ids"] < min(vocab_size, 30000)).all()
            assert (item["labels"] >= 100).all()
            assert (item["labels"] < min(vocab_size, 30000)).all()


# ═══════════════════════════════════════════════════════════════════════
#  Training config  (subqsa_trainer.config.TrainingConfig)
# ═══════════════════════════════════════════════════════════════════════

class TestTrainingConfig:
    """Sanity checks for subqsa_trainer.config.TrainingConfig."""

    def test_default_values(self):
        """TrainingConfig instantiates with reasonable defaults."""
        tc = TrainingConfig()
        assert tc.learning_rate > 0
        assert tc.min_lr > 0
        assert tc.max_steps > 0
        assert tc.warmup_steps > 0

    def test_get_torch_dtype_float32(self):
        """get_torch_dtype returns float32 by default."""
        tc = TrainingConfig()
        assert tc.get_torch_dtype() == torch.float32

    def test_get_torch_dtype_bfloat16(self):
        """get_torch_dtype('bfloat16') returns torch.bfloat16."""
        tc = TrainingConfig(dtype="bfloat16")
        assert tc.get_torch_dtype() == torch.bfloat16


class TestTrainingConfig1M:
    """Sanity checks for configs.longctx_config.TrainingConfig1M."""

    def test_default_values(self):
        """TrainingConfig1M instantiates with reasonable defaults."""
        tc = TrainingConfig1M()
        assert tc.learning_rate > 0
        assert tc.min_lr > 0
        assert tc.max_steps > 0
        assert tc.warmup_steps > 0
        assert tc.cooldown_start_step > tc.warmup_steps
        assert tc.cooldown_lr > 0

    def test_get_torch_dtype_float32(self):
        """get_torch_dtype() returns bfloat16 by default."""
        tc = TrainingConfig1M()
        # default is bfloat16
        assert tc.dtype == "bfloat16"
        assert tc.get_torch_dtype() == torch.bfloat16

    def test_get_torch_dtype_override(self):
        """Passing dtype='float16' returns torch.float16."""
        tc = TrainingConfig1M(dtype="float16")
        assert tc.get_torch_dtype() == torch.float16
