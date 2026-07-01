"""FineWeb data pipeline: streaming download + BPE tokenizer + batched dataset.

Usage:
    python data_pipeline.py --train-tokenizer   # Train BPE tokenizer on sample data
    python data_pipeline.py --tokenize          # Pre-tokenize and cache
    python data_pipeline.py --smoke             # Quick smoke test
"""

import os, json, math, logging
from typing import Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ── BPE Tokenizer (trained on FineWeb samples) ──────────────────────────


class BPETokenizer:
    """Simple BPE tokenizer wrapping HuggingFace tokenizers."""

    def __init__(self, vocab_size: int = 32_768):
        self.vocab_size = vocab_size
        self.tokenizer = None

    def train(self, texts: list[str], save_path: str = "data/tokenizer.json"):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"],
            min_frequency=2,
        )
        tokenizer.train_from_iterator(texts, trainer)
        tokenizer.save(save_path)
        self.tokenizer = tokenizer
        logger.info(f"Trained BPE tokenizer (vocab={self.vocab_size}) → {save_path}")

    def load(self, path: str = "data/tokenizer.json"):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(path)
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]


# ── Dataset ────────────────────────────────────────────────────────────


@dataclass
class DataConfig:
    dataset_name: str = "HuggingFaceFW/fineweb"
    split: str = "train"
    max_samples: int = 10_000
    max_seq_len: int = 4096
    tokenizer_path: str = "data/tokenizer.json"
    cache_dir: str = "data/tokenized"
    shuffle_buffer: int = 10_000


class FineWebDataset(Dataset):
    """Streaming FineWeb dataset with on-the-fly tokenization + caching."""

    def __init__(self, config: DataConfig, tokenizer: BPETokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.samples: list[torch.Tensor] = []
        os.makedirs(config.cache_dir, exist_ok=True)
        cache_path = os.path.join(config.cache_dir, f"samples_{config.max_seq_len}.pt")

        if os.path.exists(cache_path):
            self.samples = torch.load(cache_path)
            logger.info(f"Loaded {len(self.samples)} cached samples from {cache_path}")
            return

        self._build(config, tokenizer)
        torch.save(self.samples, cache_path)
        logger.info(f"Cached {len(self.samples)} samples to {cache_path}")

    def _build(self, config: DataConfig, tokenizer: BPETokenizer):
        from datasets import load_dataset

        ds = load_dataset(
            config.dataset_name,
            split=config.split,
            streaming=True,
        )
        count = 0
        for i, sample in enumerate(ds):
            if i >= config.max_samples:
                break
            text = sample["text"]
            ids = tokenizer.encode(text)
            # Chunk into max_seq_len segments
            for start in range(0, len(ids), config.max_seq_len):
                chunk = ids[start : start + config.max_seq_len + 1]
                if len(chunk) < config.max_seq_len // 2:
                    continue
                if len(chunk) < config.max_seq_len + 1:
                    chunk = chunk + [0] * (config.max_seq_len + 1 - len(chunk))
                t = torch.tensor(chunk[: config.max_seq_len + 1], dtype=torch.long)
                self.samples.append(t)
                count += 1
                if count >= config.max_samples * 2:
                    break
            if count >= config.max_samples * 2:
                break
        logger.info(
            f"Built {len(self.samples)} chunks from {min(config.max_samples, i + 1)} documents"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t = self.samples[idx]
        return {
            "input_ids": t[: self.config.max_seq_len],
            "labels": t[1 : self.config.max_seq_len + 1],
        }


# ── CLI ────────────────────────────────────────────────────────────────


def train_tokenizer(vocab_size=32768):
    """Train BPE tokenizer on FineWeb samples."""
    from datasets import load_dataset
    import itertools

    def text_iterator():
        ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
        for i, sample in enumerate(itertools.islice(ds, 5000)):
            if i % 1000 == 0:
                logger.info(f"Tokenizer training: read {i} docs")
            yield sample["text"]

    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(text_iterator())
    print(f"Tokenizer trained: vocab_size={tokenizer.tokenizer.get_vocab_size()}")


def tokenize_and_cache(vocab_size=32768, max_seq_len=4096, max_samples=5000):
    """Pre-tokenize FineWeb and cache."""
    from datasets import load_dataset

    tokenizer = BPETokenizer(vocab_size=vocab_size)
    if os.path.exists("data/tokenizer.json"):
        tokenizer.load("data/tokenizer.json")
    else:
        logger.info("No tokenizer found, training...")
        train_tokenizer(vocab_size)

    cfg = DataConfig(max_samples=max_samples, max_seq_len=max_seq_len)
    ds = FineWebDataset(cfg, tokenizer)
    print(f"Cached {len(ds)} samples (seq_len={max_seq_len})")

    # Quick test
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    print(
        f"Batch: input_ids={batch['input_ids'].shape}, labels={batch['labels'].shape}"
    )
    print(f"Sample tokens: {batch['input_ids'][0, :10].tolist()}")


def smoke_test():
    """Quick smoke test with minimal data."""
    from datasets import load_dataset
    import itertools

    tokenizer = BPETokenizer(vocab_size=4096)
    texts = ["Hello world, this is a test of the BPE tokenizer." for _ in range(10)]
    tokenizer.train(texts)

    ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
    sample_text = next(iter(ds))["text"]
    ids = tokenizer.encode(sample_text)
    print(f"Sample text ({len(sample_text)} chars) → {len(ids)} tokens")
    print(f"First 20 tokens: {ids[:20]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tokenizer", action="store_true")
    parser.add_argument("--tokenize", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.train_tokenizer:
        train_tokenizer()
    elif args.tokenize:
        tokenize_and_cache()
    elif args.smoke:
        smoke_test()
    else:
        parser.print_help()
