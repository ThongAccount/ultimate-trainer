"""Staged long-context training: 1B model, 1M context, FineWeb data.

Usage:
    # Stage 1 — pretrain @ 4K
    uv run torchrun --nproc_per_node=8 train_longctx.py --stage 0

    # Continue with context extension (requires +256K nodes)
    uv run torchrun --nproc_per_node=8 train_longctx.py --stage 3 --resume checkpoints/1B-stress-test/stage_2/

    # Single-GPU smoke test (tiny 300M ablation)
    uv run python3 train_longctx.py --smoke --stage 0 --max-steps 10
"""

import os, sys, math, time, json, logging, argparse
from dataclasses import dataclass
from typing import Optional, Iterator

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("train_longctx")

# ── Imports (project root) ────────────────────────────────────────────
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from configs.longctx_config import ModelConfig1B, TrainingConfig1M, count_params
from data_pipeline import BPETokenizer


# ── Long-context dataset: concatenate FineWeb docs ────────────────────


class FineWebLongCtxDataset(IterableDataset):
    """Streaming FineWeb dataset that concatenates documents to fill context.

    Combines multiple FineWeb documents with <|endoftext|> separators
    until each training sample reaches exactly `seq_len` tokens.
    """

    def __init__(
        self,
        seq_len: int = 4096,
        tokenizer_path: str = "data/tokenizer.json",
        max_docs: int = 100_000,
        dataset_name: str = "HuggingFaceFW/fineweb",
    ):
        self.seq_len = seq_len
        self.max_docs = max_docs
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(tokenizer_path)
        self.eot_id = self.tokenizer.tokenizer.token_to_id("<|endoftext|>") or 0
        self.dataset_name = dataset_name

    def __iter__(self) -> Iterator[dict]:
        from datasets import load_dataset

        ds = load_dataset(self.dataset_name, split="train", streaming=True)

        buffer: list[int] = []
        doc_count = 0

        for sample in ds:
            if doc_count >= self.max_docs:
                break
            doc_count += 1
            tokens = self.tokenizer.encode(sample["text"])
            if len(tokens) < 10:
                continue  # skip trivial docs

            if buffer:
                buffer.append(self.eot_id)
            buffer.extend(tokens)

            # Yield full sequences
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                yield {
                    "input_ids": torch.tensor(chunk[: self.seq_len], dtype=torch.long),
                    "labels": torch.tensor(
                        chunk[1 : self.seq_len + 1], dtype=torch.long
                    ),
                }

        # Last partial sequence (pad or discard)
        if len(buffer) > self.seq_len // 2:
            pad_len = self.seq_len + 1 - len(buffer)
            buffer.extend([self.eot_id] * pad_len)
            yield {
                "input_ids": torch.tensor(buffer[: self.seq_len], dtype=torch.long),
                "labels": torch.tensor(buffer[1 : self.seq_len + 1], dtype=torch.long),
            }


# ── Learning rate schedule ───────────────────────────────────────────


def get_schedule(optimizer, tc: TrainingConfig1M):
    """Two-stage cosine LR (2B4T spec)."""

    def lr_fn(step):
        if step < tc.warmup_steps:
            return step / max(1, tc.warmup_steps)
        if step < tc.cooldown_start_step:
            p = (step - tc.warmup_steps) / max(
                1, tc.cooldown_start_step - tc.warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * p))
        p = (step - tc.cooldown_start_step) / max(
            1, tc.max_steps - tc.cooldown_start_step
        )
        return 0.5 * (1.0 + math.cos(math.pi * p)) * (tc.cooldown_lr / tc.learning_rate)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


# ── Trainer ──────────────────────────────────────────────────────────


class LongCtxTrainer:
    def __init__(self, mc: ModelConfig1B, tc: TrainingConfig1M, stage: int = 0):
        self.mc = mc
        self.tc = tc
        self.stage = stage
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.global_step = 0

        # Staged context setup
        seq_len, steps, rope_base = tc.context_stages[stage]
        mc.max_seq_len = seq_len
        mc.rope_theta = rope_base
        tc.max_steps = steps + (tc.cooldown_start_step if stage == 0 else 0)

        if self.world_size > 1:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)

        # Build model
        from ultimate_trainer.model import UltimateModel

        self.model = UltimateModel(mc).to(self.device)
        logger.info(f"Model built: {count_params(mc)['total_M']:.0f}M params")
        logger.info(
            f"Stage {stage}: seq_len={seq_len}, rope_base={rope_base}, steps={tc.max_steps}"
        )

        if self.world_size > 1:
            self.model = DDP(
                self.model, device_ids=[self.local_rank], find_unused_parameters=False
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=tc.learning_rate,
            betas=(tc.beta1, tc.beta2),
            eps=tc.eps,
            weight_decay=tc.weight_decay,
        )
        self.scheduler = get_schedule(self.optimizer, tc)

        # Data
        self.dataset = FineWebLongCtxDataset(
            seq_len=seq_len,
            max_docs=100_000 if stage == 0 else 10_000,
        )

    def _get_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def train(self):
        if self.local_rank == 0:
            logger.info(f"Starting training stage {self.stage}")
            logger.info(f"  Context: {self.mc.max_seq_len:,} tokens")
            logger.info(f"  Steps: {self.tc.max_steps:,}")
            logger.info(f"  Device: {self.device}")

        self._get_model().train()
        data_iter = iter(self.dataset)
        accum_loss = 0.0
        t0 = time.perf_counter()

        for step in range(self.tc.max_steps):
            # Gradient accumulation
            for _ in range(self.tc.gradient_accumulation_steps):
                batch = next(data_iter)
                ids = batch["input_ids"].to(self.device)
                lbl = batch["labels"].to(self.device)

                loss = (
                    self._get_model().get_loss(ids, labels=lbl)
                    / self.tc.gradient_accumulation_steps
                )
                loss.backward()
                accum_loss += loss.item() * self.tc.gradient_accumulation_steps

            # Optimizer step
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.tc.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1

            # Logging
            if step % self.tc.log_interval == 0 and self.local_rank == 0:
                dt = time.perf_counter() - t0
                tps = (
                    self.tc.micro_batch_size
                    * self.mc.max_seq_len
                    * self.tc.gradient_accumulation_steps
                    * self.tc.log_interval
                    * self.world_size
                ) / dt
                logger.info(
                    f"Step {step}/{self.tc.max_steps} | "
                    f"loss={accum_loss / self.tc.log_interval:.3f} | "
                    f"lr={self.scheduler.get_last_lr()[0]:.2e} | "
                    f"tps={tps:,.0f} | "
                    f"ctx={self.mc.max_seq_len:,}"
                )
                accum_loss = 0.0
                t0 = time.perf_counter()

        if self.local_rank == 0:
            logger.info(
                f"Stage {self.stage} complete! Ready for stage {self.stage + 1}."
            )


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", type=int, default=0, help="Context extension stage index"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Checkpoint dir to resume from"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    mc = ModelConfig1B()
    tc = TrainingConfig1M()

    if args.smoke:
        # Tiny 300M ablation for quick smoke test
        mc = ModelConfig1B(
            hidden_dim=768,
            intermediate_dim=2048,
            num_layers=6,
            num_attention_heads=8,
            num_kv_heads=2,
            max_seq_len=4096,
        )
        tc.max_steps = args.max_steps or 20
        tc.log_interval = 5
        tc.distributed = False
        tc.dtype = "float32"
        tc.gradient_accumulation_steps = 1
        tc.micro_batch_size = 1

    if args.max_steps:
        tc.max_steps = args.max_steps

    trainer = LongCtxTrainer(mc, tc, stage=args.stage)
    trainer.train()

    # Print next step
    next_stage = args.stage + 1
    if next_stage < len(tc.context_stages):
        seq, steps, base = tc.context_stages[next_stage]
        mc.rope_theta = base
        logger.info(f"\nNext: stage {next_stage} @ {seq:,} ctx, RoPE base={base:,.0f}")


if __name__ == "__main__":
    main()
