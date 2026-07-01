"""
Training loop for SubQSA (NSA-style) transformer.
Supports staged context extension and DDP.
"""

import os, sys, math, time, json, logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
from subqsa_trainer.config import ModelConfig, TrainingConfig
from subqsa_trainer.model import SubQSAModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DummyDataset(Dataset):
    """Synthetic token dataset for smoke-testing without real data."""

    def __init__(self, seq_len, vocab_size=32768, num_samples=1000):
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


def get_cosine_schedule_with_warmup(
    optimizer, warmup_steps, total_steps, min_lr_ratio=0.1
):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class SubQSATrainer:
    def __init__(self, model_config, train_config):
        self.tc = train_config
        self.mc = model_config
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.global_step = 0

        if self.tc.distributed and self.world_size > 1:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)

        self.model = SubQSAModel(model_config).to(self.device)

        if self.tc.distributed and self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            betas=(train_config.beta1, train_config.beta2),
            eps=train_config.eps,
            weight_decay=train_config.weight_decay,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            train_config.warmup_steps,
            train_config.max_steps,
            train_config.min_lr / train_config.learning_rate,
        )
        self.train_loader = self._build_dataloader()

    def _build_dataloader(self):
        dataset = DummyDataset(
            seq_len=self.mc.max_seq_len, vocab_size=self.mc.vocab_size
        )
        if self.tc.distributed and self.world_size > 1:
            sampler = DistributedSampler(
                dataset, num_replicas=self.world_size, rank=self.local_rank
            )
        else:
            sampler = None
        return DataLoader(
            dataset,
            batch_size=self.tc.micro_batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=0,
            drop_last=True,
        )

    def _get_model(self):
        if self.world_size > 1 and hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def train_step(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        loss = self._get_model().get_loss(input_ids, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1
        return loss.item()

    def train(self):
        logger.info("=" * 60)
        logger.info("Starting SubQSA training (NSA-style sparse attention)")
        logger.info(f"  Model: {self.mc.num_layers}L, hidden={self.mc.hidden_dim}")
        logger.info(
            f"  SubQSA: cmp_block={self.mc.subqsa.cmp_block}, slc_topk={self.mc.subqsa.slc_topk}"
        )
        logger.info(f"  Stages: {self.tc.context_stages}")
        self.model.train()
        data_iter = iter(self.train_loader)
        for step in range(self.tc.max_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
            loss = self.train_step(batch)
            if step % self.tc.log_interval == 0 and self.local_rank == 0:
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    f"Step {step}/{self.tc.max_steps} | Loss: {loss:.4f} | LR: {lr:.2e}"
                )
        if self.local_rank == 0:
            logger.info("Training complete!")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="checkpoints/subqsa_trainer")
    parser.add_argument(
        "--smoke", action="store_true", default=True, help="Smoke test with tiny config"
    )
    args = parser.parse_args()

    if args.smoke:
        mc = ModelConfig(
            vocab_size=4096,
            hidden_dim=256,
            intermediate_dim=512,
            num_layers=2,
            num_attention_heads=4,
            max_seq_len=128,
        )
        tc = TrainingConfig(
            max_steps=10, log_interval=2, learning_rate=1e-3, max_seq_len=128
        )
    else:
        mc = ModelConfig()
        tc = TrainingConfig()
    tc.output_dir = args.output_dir
    os.makedirs(tc.output_dir, exist_ok=True)

    trainer = SubQSATrainer(mc, tc)
    trainer.train()


if __name__ == "__main__":
    main()
