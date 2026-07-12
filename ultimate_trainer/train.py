"""Ultimate Trainer training loop."""

import os
import sys
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
from ultimate_trainer.config import UltimateModelConfig, UltimateTrainingConfig
from ultimate_trainer.model import UltimateModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def get_cosine_schedule_with_warmup(
    optimizer, warmup_steps, total_steps, min_lr_ratio=0.1
):
    """Cosine LR schedule with linear warmup.

    Warmup: 0 → 1 over warmup_steps.
    Decay:  1 → min_lr_ratio over remaining steps.
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )

    return LambdaLR(optimizer, lr_lambda)


class DummyDataset(Dataset):
    """Synthetic token dataset for smoke-testing training."""

    def __init__(self, seq_len, vocab_size=32768, num_samples=500):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens = torch.randint(100, min(self.vocab_size, 30000), (self.seq_len + 1,))
        return {
            "input_ids": tokens[: self.seq_len],
            "labels": tokens[1 : self.seq_len + 1],
        }


class UltimateTrainer:
    def __init__(self, mc, tc, dataset=None):
        self.mc = mc
        self.tc = tc
        self.global_step = 0

        # ── DDP ──────────────────────────────────────────────────────
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if self.local_rank >= 0:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl")
            self.device = f"cuda:{self.local_rank}"
        else:
            self.device = "cpu"

        # ── Model ────────────────────────────────────────────────────
        self.model = UltimateModel(mc).to(self.device)
        if self.local_rank >= 0:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank]
            )

        # ── Optimizer ────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=tc.learning_rate,
            betas=(tc.beta1, tc.beta2),
            eps=tc.eps,
            weight_decay=tc.weight_decay,
        )

        # ── LR Scheduler ─────────────────────────────────────────────
        if tc.learning_rate > 0:
            min_lr_ratio = tc.min_lr / tc.learning_rate
        else:
            min_lr_ratio = 0.1
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=tc.warmup_steps,
            total_steps=tc.max_steps,
            min_lr_ratio=min_lr_ratio,
        )

        # ── Dataset & DataLoader ─────────────────────────────────────
        if dataset is None:
            dataset = DummyDataset(mc.max_seq_len, vocab_size=mc.vocab_size)
        self.dataset = dataset
        self._build_dataloader()

        # ── Staged context extension ─────────────────────────────────
        self.context_stages = list(tc.context_stages) if tc.context_stages else []
        self._current_stage = 0

    def step(self):
        try:
            batch = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            batch = next(self.it)
        ids = batch["input_ids"].to(self.device)
        lbl = batch["labels"].to(self.device)
        # Go through forward() so DDP hooks sync gradients
        logits = self.model(ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            lbl.view(-1),
            ignore_index=0,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1
        self._maybe_extend_context()
        return loss.item()

    def _build_dataloader(self):
        if self.local_rank >= 0:
            sampler = DistributedSampler(self.dataset)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.tc.micro_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
            drop_last=True,
        )
        self.it = iter(self.loader)

    def _maybe_extend_context(self):
        """Extend max_seq_len when the current training stage threshold is hit."""
        if self._current_stage >= len(self.context_stages):
            return
        target_seq_len, _ = self.context_stages[self._current_stage]
        cumulative = sum(
            s[1] for s in self.context_stages[: self._current_stage + 1]
        )
        if self.global_step >= cumulative:
            if target_seq_len > self.mc.max_seq_len:
                logger.info(
                    f"Extending context from {self.mc.max_seq_len} to "
                    f"{target_seq_len} at step {self.global_step}"
                )
                self.mc.max_seq_len = target_seq_len
                self.dataset = DummyDataset(
                    target_seq_len, vocab_size=self.mc.vocab_size
                )
                self._build_dataloader()
            self._current_stage += 1

    def train(self):
        logger.info("ULTIMATE TRAINER: BitNet b1.58 2B4T + SubQSA")
        logger.info(
            f"  layers={self.mc.num_layers}, hidden={self.mc.hidden_dim}, "
            f"bitlinear={self.mc.use_bitlinear}"
        )
        self.model.train()
        # Enable activation quantization warmup ramp (default _quant_step=5000
        # skips warmup; reset to 0 so the linear ramp activates).
        from ultimate_trainer.bitlinear import BitLinear
        for module in self.model.modules():
            if isinstance(module, BitLinear):
                module._quant_step = 0
        for step in range(self.tc.max_steps):
            loss = self.step()
            if step % self.tc.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Step {step}/{self.tc.max_steps} | loss={loss:.4f} | lr={lr:.2e}"
                )
        # ── Final checkpoint save ──────────────────────────────────────
        import os as _os
        ckpt_dir = _os.path.join(self.tc.output_dir, f"step_{self.global_step}")
        _os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), _os.path.join(ckpt_dir, "model.pt"))
        torch.save(self.optimizer.state_dict(), _os.path.join(ckpt_dir, "optim.pt"))
        logger.info(f"Checkpoint saved to {ckpt_dir}")
        logger.info("Training complete.")
        if self.local_rank >= 0:
            dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--real-data", action="store_true", help="Use FineWeb dataset")
    parser.add_argument(
        "--smoke", action="store_true", default=True, help="Smoke test with tiny config"
    )
    args = parser.parse_args()

    if args.smoke and not args.real_data:
        mc = UltimateModelConfig(
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
        tc = UltimateTrainingConfig(max_steps=20, log_interval=5, learning_rate=1e-3)
        trainer = UltimateTrainer(mc, tc)
    elif args.real_data:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data_pipeline import DataConfig, BPETokenizer, FineWebDataset

        tok = BPETokenizer()
        if os.path.exists("data/tokenizer.json"):
            tok.load("data/tokenizer.json")
        else:
            logger.info(
                "No tokenizer found. Run: python data_pipeline.py --train-tokenizer"
            )
            sys.exit(1)

        mc = UltimateModelConfig()
        tc = UltimateTrainingConfig(max_steps=100, log_interval=10, learning_rate=1e-3)
        dcfg = DataConfig(max_seq_len=mc.max_seq_len, max_samples=5000)
        ds = FineWebDataset(dcfg, tok)
        logger.info(f"FineWeb dataset: {len(ds)} samples (seq_len={mc.max_seq_len})")
        trainer = UltimateTrainer(mc, tc, dataset=ds)
    else:
        mc = UltimateModelConfig()
        tc = UltimateTrainingConfig()
        trainer = UltimateTrainer(mc, tc)
    trainer.train()
