"""
Training loop for native 1-bit (ternary) LLM — BitNet b1.58.

Key training features:
  - Native 1-bit training: weights are ternary {-1, 0, +1} from scratch
  - Straight-Through Estimator for differentiable quantization
  - 8-bit activation quantization with gradual warmup
  - Cosine LR schedule with warmup
  - Gradient clipping
  - Distributed Data Parallel (DDP) support
  - Checkpointing with ternary weights stored efficiently

Usage:
    python train.py --config defaults
    torchrun --nproc_per_node=8 train.py --distributed
"""

import os
import sys
import math
import time
import json
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from config import ModelConfig, TrainingConfig
from model import BitNetModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────────


class TextDataset(Dataset):
    """Simple memory-mapped token dataset for pre-training."""

    def __init__(self, path: str, seq_len: int):
        self.seq_len = seq_len
        if path.endswith(".npy"):
            self.data = np.memmap(path, dtype=np.int32, mode="r")
        elif path.endswith(".bin"):
            self.data = np.memmap(path, dtype=np.int16, mode="r")
        else:
            self.data = np.memmap(path, dtype=np.int32, mode="r")
        self.num_tokens = len(self.data)

    def __len__(self):
        # Number of complete sequences
        return self.num_tokens // self.seq_len

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for labels (shift by 1)
        tokens = torch.from_numpy(self.data[start:end].astype(np.int64))
        return {
            "input_ids": tokens[:self.seq_len],
            "labels": tokens[1:self.seq_len + 1],
        }


class StreamingJsonlDataset(Dataset):
    """On-the-fly streaming dataset from pre-tokenized JSONL files."""

    def __init__(self, path: str, seq_len: int, world_size: int = 1):
        self.seq_len = seq_len
        self.samples: list[torch.Tensor] = []
        total_tokens = 0

        if os.path.isfile(path):
            files = [path]
        else:
            files = [
                os.path.join(path, f) for f in os.listdir(path)
                if f.endswith(".jsonl")
            ]

        for filepath in files:
            with open(filepath, "r") as f:
                for line in f:
                    data = json.loads(line)
                    tokens = torch.tensor(data["tokens"], dtype=torch.long)
                    total_tokens += len(tokens)
                    # Chunk into sequences
                    for i in range(0, len(tokens) - seq_len, seq_len // 2):
                        chunk = tokens[i:i + seq_len + 1]
                        if len(chunk) >= seq_len + 1:
                            self.samples.append(chunk)

        logger.info(
            f"Loaded {len(self.samples)} samples "
            f"({total_tokens:,} total tokens) from {path}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        tokens = self.samples[idx]
        return {
            "input_ids": tokens[:self.seq_len],
            "labels": tokens[1:self.seq_len + 1],
        }


# ──────────────────────────────────────────────────────────────────────
#  Learning Rate Scheduler
# ──────────────────────────────────────────────────────────────────────


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Cosine LR schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        # Warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────────────
#  Training Loop
# ──────────────────────────────────────────────────────────────────────


class Trainer:
    """
    Trainer for BitNet b1.58 with native 1-bit quantization.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainingConfig,
    ):
        self.tc = train_config
        self.mc = model_config
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        self.global_step = 0

        # Initialize distributed
        if self.tc.distributed and self.world_size > 1:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
            logger.info(f"Initialized DDP: rank {dist.get_rank()}/{dist.get_world_size()}")

        # Model
        self.model = BitNetModel(model_config)
        self.model.to(self.device)

        if self.tc.distributed and self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=False,
            )

        self._log_model_size()

        # Optimizer — AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            betas=(train_config.beta1, train_config.beta2),
            eps=train_config.eps,
            weight_decay=train_config.weight_decay,
        )

        # LR Scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=train_config.warmup_steps,
            total_steps=train_config.max_steps,
            min_lr_ratio=train_config.min_lr / train_config.learning_rate,
        )

        # Gradient scaler (only for fp16)
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(train_config.dtype == "float16")
        )

        # Data
        self.train_loader = self._build_dataloader()

    def _log_model_size(self):
        """Log model parameter counts, broken down by type."""
        total = sum(p.numel() for p in self.model.parameters())
        if self.world_size > 1 and hasattr(self.model, "module"):
            mp = self.model.module
        else:
            mp = self.model

        # Count BitLinear vs non-quantized params
        bitlinear_params = 0
        other_params = 0
        for name, mod in mp.named_modules():
            if isinstance(mod, nn.Embedding):
                other_params += sum(p.numel() for p in mod.parameters())

        for name, p in mp.named_parameters():
            if "embed_tokens" in name:
                continue
            if "weight" in name:
                bitlinear_params += p.numel()
            else:
                other_params += p.numel()

        logger.info(
            f"Model: {total:,} total parameters\n"
            f"  BitLinear (ternary) params: {bitlinear_params:,}\n"
            f"  Full-precision params: {other_params:,}\n"
            f"  Effective memory: {total * 1.58 / 8 / 1e9:.2f} GB (ternary)"
        )

    def _build_dataloader(self) -> DataLoader:
        """Build training dataloader."""
        dataset = StreamingJsonlDataset(
            path=self.tc.dataset_path,
            seq_len=self.tc.max_seq_len,
        )

        sampler = None
        if self.tc.distributed and self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True,
            )

        return DataLoader(
            dataset,
            batch_size=self.tc.micro_batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=True,
            drop_last=True,
        )

    def _get_model(self) -> nn.Module:
        """Return the underlying model (unwrap DDP wrapper)."""
        if self.world_size > 1 and hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def train_step(self, batch: dict) -> float:
        """Single training step with native 1-bit quantization."""
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        # Activation quantization warmup
        # Gradually enable activation quantization over the first N steps
        _maybe_set_act_quant(self._get_model(), self.global_step, self.tc)

        # Forward
        with torch.cuda.amp.autocast(
            enabled=(self.tc.dtype != "float32"),
            dtype=self.tc.get_torch_dtype(),
        ):
            loss = self._get_model().get_loss(input_ids, labels=labels)

        # Backward
        self.scaler.scale(loss).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.tc.max_grad_norm
        )

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return loss.item()

    def eval(self) -> dict:
        """Run evaluation on a small validation set."""
        self._get_model().eval()
        total_loss = 0.0
        num_batches = 0

        # Use a subset of training data for eval
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                if i >= 20:  # eval on 20 batches
                    break
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                loss = self._get_model().get_loss(input_ids, labels=labels)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        perplexity = math.exp(avg_loss)

        self._get_model().train()
        return {"loss": avg_loss, "perplexity": perplexity}

    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        model = self._get_model()
        output_dir = os.path.join(self.tc.output_dir, f"step_{step}")
        os.makedirs(output_dir, exist_ok=True)

        # Save model weights (master weights stored in FP32)
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, "model.pt"))

        # Save configs
        with open(os.path.join(output_dir, "model_config.json"), "w") as f:
            f.write(json.dumps({
                k: v for k, v in self.mc.__dict__.items()
                if not k.startswith("_")
            }, indent=2))

        with open(os.path.join(output_dir, "training_config.json"), "w") as f:
            f.write(json.dumps({
                k: v for k, v in self.tc.__dict__.items()
                if not k.startswith("_")
            }, indent=2))

        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))

        logger.info(f"Checkpoint saved to {output_dir}")

    def train(self):
        """Main training loop."""
        logger.info("=" * 60)
        logger.info("Starting BitNet b1.58 training with native 1-bit quantization")
        logger.info(f"  Model: {self.mc.num_layers} layers, {self.mc.hidden_dim} hidden")
        logger.info(f"  Weights: ternary {{-1, 0, +1}} (absmean quantization)")
        logger.info(f"  Activations: {self.mc.activation_bits}-bit per token")
        logger.info(f"  Max steps: {self.tc.max_steps}")
        logger.info(f"  Batch size (global): {self.tc.micro_batch_size * self.tc.gradient_accumulation_steps * self.world_size}")
        logger.info(f"  Learning rate: {self.tc.learning_rate} → {self.tc.min_lr} (cosine)")
        logger.info("=" * 60)

        self.model.train()
        total_tokens_seen = 0
        accumulation_loss = 0.0
        step_start_time = time.time()

        # Training loop
        data_iter = iter(self.train_loader)
        while self.global_step < self.tc.max_steps:
            # Gradient accumulation
            for _ in range(self.tc.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                loss = self.train_step(batch)
                accumulation_loss += loss

            self.global_step += 1
            total_tokens_seen += (
                self.tc.micro_batch_size
                * self.tc.max_seq_len
                * self.tc.gradient_accumulation_steps
                * self.world_size
            )

            # Logging
            if self.global_step % self.tc.log_interval == 0 and self.local_rank == 0:
                step_time = time.time() - step_start_time
                tokens_per_sec = (
                    self.tc.micro_batch_size
                    * self.tc.max_seq_len
                    * self.tc.gradient_accumulation_steps
                    * self.tc.log_interval
                    * self.world_size
                ) / step_time
                lr = self.scheduler.get_last_lr()[0]

                logger.info(
                    f"Step {self.global_step}/{self.tc.max_steps} | "
                    f"Loss: {accumulation_loss / self.tc.log_interval:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Tokens: {total_tokens_seen:,} | "
                    f"Tokens/s: {tokens_per_sec:,.0f}"
                )
                accumulation_loss = 0.0
                step_start_time = time.time()

            # Evaluation
            if self.global_step % self.tc.eval_interval == 0 and self.local_rank == 0:
                eval_metrics = self.eval()
                logger.info(
                    f"Eval — Loss: {eval_metrics['loss']:.4f}, "
                    f"Perplexity: {eval_metrics['perplexity']:.2f}"
                )

            # Save checkpoint
            if self.global_step % self.tc.save_interval == 0 and self.local_rank == 0:
                self.save_checkpoint(self.global_step)

        # Final save
        if self.local_rank == 0:
            self.save_checkpoint(self.global_step)
            logger.info("Training complete!")


def _maybe_set_act_quant(
    model: nn.Module,
    global_step: int,
    tc: TrainingConfig,
):
    """Gradually enable activation quantization during warmup."""
    if global_step < tc.act_quant_warmup_steps:
        ratio = global_step / tc.act_quant_warmup_steps
        for module in model.modules():
            if hasattr(module, "quantize_activations"):
                # Gradually turn on quantization
                module.quantize_activations = (ratio > 0.5)
    else:
        for module in model.modules():
            if hasattr(module, "quantize_activations") and not module.quantize_activations:
                module.quantize_activations = True


# ──────────────────────────────────────────────────────────────────────
#  Main Entry Point
# ──────────────────────────────────────────────────────────────────────


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train BitNet b1.58 1-bit LLM")
    parser.add_argument("--config", type=str, default=None, help="Config file (JSON)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--dataset", type=str, default="data/redpajama")
    parser.add_argument("--output_dir", type=str, default="checkpoints/1bit-trainer")
    args = parser.parse_args()

    # Model config
    model_config = ModelConfig()

    # Training config
    train_config = TrainingConfig()
    if args.dataset:
        train_config.dataset_path = args.dataset
    if args.output_dir:
        train_config.output_dir = args.output_dir

    # Override from JSON config
    if args.config:
        with open(args.config) as f:
            cfg_data = json.load(f)
        for k, v in cfg_data.get("model", {}).items():
            setattr(model_config, k, v)
        for k, v in cfg_data.get("training", {}).items():
            setattr(train_config, k, v)

    # Create trainer
    trainer = Trainer(model_config, train_config)

    # Resume from checkpoint
    if args.resume:
        model = trainer._get_model()
        state_dict = torch.load(
            os.path.join(args.resume, "model.pt"),
            map_location=trainer.device,
        )
        model.load_state_dict(state_dict)
        opt_path = os.path.join(args.resume, "optimizer.pt")
        if os.path.exists(opt_path):
            trainer.optimizer.load_state_dict(
                torch.load(opt_path, map_location=trainer.device)
            )
        logger.info(f"Resumed from {args.resume}")

    # Train
    trainer.train()


if __name__ == "__main__":
    try:
        import numpy as np
    except ImportError:
        print("numpy is required. Install with: pip install numpy")
        sys.exit(1)
    main()
