"""Ultimate Trainer training loop."""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
from ultimate_trainer.training_mode import (
    TrainingMode,
    ModeConfig,
    compute_loss,
    get_loss_fn,
)

# ── WandB ───────────────────────────────────────────────────────────────────
try:
    import wandb as _wandb

    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

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
    def __init__(self, mc, tc, dataset=None, validation_dataset=None):
        self.mc = mc
        self.tc = tc
        self.global_step = 0
        self.best_val_loss = float("inf")

        # ── Device & DDP ──────────────────────────────────────────────
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if self.local_rank >= 0:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl")
            self.device = f"cuda:{self.local_rank}"
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.local_rank = -1  # ensure non-DDP path
        else:
            self.device = "cpu"

        # ── Model ────────────────────────────────────────────────────
        self.model = UltimateModel(mc)
        if tc.dtype in ('bfloat16', 'float16'):
            self.model = self.model.to(dtype=tc.get_torch_dtype())
        self.model = self.model.to(self.device)
        if self.local_rank >= 0:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank],
                find_unused_parameters=True,
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

        # ── Validation Dataset ────────────────────────────────────────
        self.validation_dataset = validation_dataset
        self._val_loader = None
        if validation_dataset is not None:
            self._build_val_loader()

        # ── Staged context extension ─────────────────────────────────
        self.context_stages = list(tc.context_stages) if tc.context_stages else []
        self._current_stage = 0

        # ── WandB ───────────────────────────────────────────────────────
        self._wandb_run = None
        if tc.use_wandb and _HAS_WANDB and self.local_rank <= 0:
            wandb_kwargs = dict(
                project=tc.wandb_project,
                name=tc.run_name,
                config={
                    "model": str(mc),
                    "training": str(tc),
                    "hidden_dim": mc.hidden_dim,
                    "num_layers": mc.num_layers,
                    "heads": mc.num_attention_heads,
                    "kv_heads": mc.num_kv_heads,
                    "max_seq_len": mc.max_seq_len,
                    "bitlinear": mc.use_bitlinear,
                    "subqsa": mc.use_subqsa,
                    "act_bits": mc.activation_bits,
                    "max_steps": tc.max_steps,
                    "batch_size": tc.micro_batch_size,
                    "grad_accum": tc.gradient_accumulation_steps,
                    "lr": tc.learning_rate,
                    "dtype": tc.dtype,
                },
            )
            if tc.wandb_entity:
                wandb_kwargs["entity"] = tc.wandb_entity
            self._wandb_run = _wandb.init(**wandb_kwargs)

        # ── Training mode ──────────────────────────────────────────────
        self.mode = tc.get_mode()
        self.mode_config = tc.get_mode_config()
        logger.info("Training mode: %s", self.mode)

        # ── Gradient accumulation ──────────────────────────────────────
        self._acc_counter = 0

    def train_step(self):
        """Forward + backward for one micro-batch.

        Accumulates gradients; optimizer step fires every
        ``gradient_accumulation_steps`` micro-batches.  Returns the
        *scaled* loss and auxiliary metrics dict.
        """
        try:
            batch = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            batch = next(self.it)

        # ── Mode-aware forward + loss ─────────────────────────────────
        if self.mode.uses_cross_entropy:
            ids = batch["input_ids"].to(self.device)
            lbl = batch["labels"].to(self.device)
            logits = self.model(ids)
            loss, aux = compute_loss(self.mode, logits, {"labels": lbl}, self.mode_config)
        elif self.mode == TrainingMode.DPO:
            chosen_ids = batch["chosen_input_ids"].to(self.device)
            chosen_lbl = batch["chosen_labels"].to(self.device)
            rejected_ids = batch["rejected_input_ids"].to(self.device)
            rejected_lbl = batch["rejected_labels"].to(self.device)
            chosen_logits = self.model(chosen_ids)
            rejected_logits = self.model(rejected_ids)
            loss, aux = compute_loss(
                self.mode,
                {"chosen": chosen_logits, "rejected": rejected_logits},
                {"chosen_labels": chosen_lbl, "rejected_labels": rejected_lbl},
                self.mode_config,
            )
        elif self.mode == TrainingMode.RL:
            loss, aux = compute_loss(
                self.mode,
                None,
                {
                    "log_probs": batch["log_probs"].to(self.device),
                    "old_log_probs": batch["old_log_probs"].to(self.device),
                    "advantages": batch["advantages"].to(self.device),
                    "rewards": batch.get("rewards", torch.tensor(0.0)).to(self.device),
                },
                self.mode_config,
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Scale loss for gradient accumulation
        loss = loss / self.tc.gradient_accumulation_steps
        # DDP: only sync gradients on the last micro-batch of each accumulation cycle
        is_last_acc = (self._acc_counter + 1) % self.tc.gradient_accumulation_steps == 0
        if self.local_rank >= 0 and not is_last_acc:
            with self.model.no_sync():
                loss.backward()
        else:
            loss.backward()
        self._acc_counter += 1

        if self._acc_counter % self.tc.gradient_accumulation_steps == 0:
            self._optimizer_step()

        self._maybe_extend_context()
        # Return unscaled loss + aux metrics for logging
        return loss.item() * self.tc.gradient_accumulation_steps, aux

    def _optimizer_step(self):
        """Clip gradients, update weights, advance LR scheduler."""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1

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

    def _build_val_loader(self):
        self._val_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.tc.micro_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

    @torch.no_grad()
    def evaluate(self):
        """Compute validation metrics over the held-out set.

        Returns: (primary_metric, perplexity_or_aux_dict)
          - PRETRAIN/SFT: (val_loss, perplexity)
          - DPO: (val_dpo_loss, {"accuracy": ...})
          - RL:  (mean_reward, {})
        """
        if self.validation_dataset is None or len(self.validation_dataset) == 0:
            return None, None

        self.model.eval()
        total_metric = 0.0
        num_batches = 0
        aux_metrics = {}

        if self.mode.uses_cross_entropy:
            for batch in self._val_loader:
                ids = batch["input_ids"].to(self.device)
                lbl = batch["labels"].to(self.device)
                logits = self.model(ids)
                loss, aux = compute_loss(self.mode, logits, {"labels": lbl}, self.mode_config)
                total_metric += loss.item()
                num_batches += 1

            avg_loss = total_metric / max(1, num_batches)
            perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
            self.model.train()
            return avg_loss, perplexity

        elif self.mode == TrainingMode.DPO:
            for batch in self._val_loader:
                ci = batch["chosen_input_ids"].to(self.device)
                cl = batch["chosen_labels"].to(self.device)
                ri = batch["rejected_input_ids"].to(self.device)
                rl = batch["rejected_labels"].to(self.device)
                chosen_logits = self.model(ci)
                rejected_logits = self.model(ri)
                loss, aux = compute_loss(
                    self.mode,
                    {"chosen": chosen_logits, "rejected": rejected_logits},
                    {"chosen_labels": cl, "rejected_labels": rl},
                    self.mode_config,
                )
                total_metric += loss.item()
                for k, v in aux.items():
                    aux_metrics.setdefault(k, 0.0)
                    aux_metrics[k] += v
                num_batches += 1

            avg_loss = total_metric / max(1, num_batches)
            avg_aux = {k: v / max(1, num_batches) for k, v in aux_metrics.items()}
            self.model.train()
            return avg_loss, avg_aux

        raise ValueError(f"Unknown mode: {self.mode}")

    @torch.no_grad()
    def evaluate_one_batch(self) -> float:
        """Evaluate a single batch and return the scalar loss/reward.

        Used by Ctrl-Z to collect *M* samples per evaluation.
        """
        if self.validation_dataset is None or len(self.validation_dataset) == 0:
            return 0.0

        self.model.eval()
        # Pick a random batch from the val loader
        batch = next(iter(self._val_loader))

        if self.mode.uses_cross_entropy:
            ids = batch["input_ids"].to(self.device)
            lbl = batch["labels"].to(self.device)
            logits = self.model(ids)
            loss, _ = compute_loss(self.mode, logits, {"labels": lbl}, self.mode_config)
            self.model.train()
            return loss.item()

        elif self.mode == TrainingMode.DPO:
            ci = batch["chosen_input_ids"].to(self.device)
            cl = batch["chosen_labels"].to(self.device)
            ri = batch["rejected_input_ids"].to(self.device)
            rl = batch["rejected_labels"].to(self.device)
            chosen_logits = self.model(ci)
            rejected_logits = self.model(ri)
            loss, _ = compute_loss(
                self.mode,
                {"chosen": chosen_logits, "rejected": rejected_logits},
                {"chosen_labels": cl, "rejected_labels": rl},
                self.mode_config,
            )
            self.model.train()
            return loss.item()

        elif self.mode == TrainingMode.RL:
            # Return a dummy reward for now
            logger.warning("RL evaluate_one_batch() not yet implemented — returning 0.0")
            self.model.train()
            return 0.0

        raise ValueError(f"Unknown mode: {self.mode}")

    def _save_checkpoint(self, is_best=False):
        """Save model (as BF16) and optimizer state dicts."""
        suffix = "best" if is_best else f"step_{self.global_step}"
        ckpt_dir = os.path.join(self.tc.output_dir, suffix)
        os.makedirs(ckpt_dir, exist_ok=True)
        bf16_sd = {
            k: v.to(torch.bfloat16) if v.is_floating_point() else v
            for k, v in self.model.state_dict().items()
        }
        torch.save(bf16_sd, os.path.join(ckpt_dir, "model.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(ckpt_dir, "optim.pt"))
        logger.info(f"Checkpoint saved to {ckpt_dir}")

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
        micro_batches = self.tc.max_steps * self.tc.gradient_accumulation_steps
        for mb in range(micro_batches):
            loss, aux = self.train_step()
            current_opt_step = self.global_step

            # Report every log_interval optimizer steps
            if current_opt_step > 0 and current_opt_step % self.tc.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                aux_str = " | ".join(f"{k}={v:.4f}" for k, v in aux.items()) if aux else ""
                logger.info(
                    f"Opt step {current_opt_step}/{self.tc.max_steps} | "
                    f"loss={loss:.4f} | lr={lr:.2e}" + (f" | {aux_str}" if aux_str else "")
                )
                if self._wandb_run is not None:
                    log_dict = {
                        "train/loss": loss,
                        "train/lr": lr,
                        "train/step": current_opt_step,
                    }
                    for k, v in aux.items():
                        log_dict[f"train/{k}"] = v
                    self._wandb_run.log(log_dict)

            #─ Evaluation ────────────────────────────────────────────
            if current_opt_step > 0 and current_opt_step % self.tc.eval_interval == 0:
                val_loss, val_aux = self.evaluate()
                if val_loss is not None:
                    logger.info(
                        f"Eval opt step {current_opt_step}: val_loss={val_loss:.4f}"
                    )
                    if self._wandb_run is not None:
                        self._wandb_run.log({
                            "eval/loss": val_loss,
                            "eval/step": current_opt_step,
                        })
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(is_best=True)

        logger.info("Training complete.")

        # Final checkpoint save
        self._save_checkpoint(is_best=False)

        if self._wandb_run is not None:
            self._wandb_run.finish()

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
            use_checkpoint=False,
            cmp_block=16,
            cmp_stride=8,
            slc_block=32,
            slc_topk=4,
            win_size=32,
        )
        tc = UltimateTrainingConfig(max_steps=20, log_interval=5, eval_interval=10, learning_rate=1e-3)
        train_ds = DummyDataset(mc.max_seq_len, vocab_size=mc.vocab_size, num_samples=500)
        val_ds = DummyDataset(mc.max_seq_len, vocab_size=mc.vocab_size, num_samples=50)
        trainer = UltimateTrainer(mc, tc, dataset=train_ds, validation_dataset=val_ds)
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

        mc = UltimateModelConfig(use_checkpoint=True)
        tc = UltimateTrainingConfig(max_steps=100, log_interval=10, learning_rate=1e-3, dtype="bfloat16")
        dcfg = DataConfig(max_seq_len=mc.max_seq_len, max_samples=5000)
        ds = FineWebDataset(dcfg, tok)
        logger.info(f"FineWeb dataset: {len(ds)} samples (seq_len={mc.max_seq_len})")
        trainer = UltimateTrainer(mc, tc, dataset=ds)
    else:
        mc = UltimateModelConfig()
        tc = UltimateTrainingConfig()
        trainer = UltimateTrainer(mc, tc)
    trainer.train()
