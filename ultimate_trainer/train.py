"""Ultimate Trainer training loop."""

import os
import sys
import math
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
from ultimate_trainer.config import UltimateModelConfig, UltimateTrainingConfig
from ultimate_trainer.model import UltimateModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


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
        self.device = "cpu"
        self.global_step = 0
        self.model = UltimateModel(mc).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=tc.learning_rate,
            betas=(tc.beta1, tc.beta2),
            eps=tc.eps,
            weight_decay=tc.weight_decay,
        )
        if dataset is None:
            dataset = DummyDataset(mc.max_seq_len, vocab_size=mc.vocab_size)
        self.loader = DataLoader(
            dataset,
            batch_size=tc.micro_batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        self.it = iter(self.loader)

    def step(self):
        try:
            batch = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            batch = next(self.it)
        ids = batch["input_ids"].to(self.device)
        loss = self.model.get_loss(ids, labels=ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.global_step += 1
        return loss.item()

    def train(self):
        logger.info("ULTIMATE TRAINER: BitNet b1.58 2B4T + SubQSA")
        logger.info(
            f"  layers={self.mc.num_layers}, hidden={self.mc.hidden_dim}, "
            f"bitlinear={self.mc.use_bitlinear}"
        )
        self.model.train()
        for step in range(self.tc.max_steps):
            loss = self.step()
            if step % self.tc.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Step {step}/{self.tc.max_steps} | loss={loss:.4f} | lr={lr:.2e}"
                )
        logger.info("Training complete.")


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
