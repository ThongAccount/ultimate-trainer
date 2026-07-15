"""1B model, 128K vocab, staged 32K→256K context, ~250 steps, FineWeb real data."""
import os, sys, logging

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from ultimate_trainer.config import UltimateModelConfig, UltimateTrainingConfig
from ultimate_trainer.train import UltimateTrainer
from data_pipeline import DataConfig, BPETokenizer, FineWebDataset, tokenize_and_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── Tokenizer ──
tok = BPETokenizer()
if os.path.exists("data/tokenizer.json"):
    tok.load("data/tokenizer.json")
    logger.info(f"Loaded tokenizer from data/tokenizer.json")
else:
    logger.info("No tokenizer found. Run: python data_pipeline.py --train-tokenizer")
    sys.exit(1)

# ── Config ──
mc = UltimateModelConfig(
    vocab_size=128256,
    hidden_dim=2048,
    intermediate_dim=5632,
    num_layers=20,
    num_attention_heads=16,
    num_kv_heads=4,
    head_dim=128,
    max_seq_len=32768,
    use_bitlinear=True,
    use_subqsa=True,
    use_checkpoint=True,
    cmp_block=64,
    cmp_stride=32,
    slc_block=128,
    slc_topk=32,
    win_size=1024,
)

tc = UltimateTrainingConfig(
    max_steps=250,
    micro_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    min_lr=3e-5,
    warmup_steps=25,
    max_grad_norm=1.0,
    weight_decay=0.1,
    log_interval=10,
    eval_interval=50,
    save_interval=50,
    dtype="bfloat16",
    context_stages=(
        (32768, 80),   # steps 0-79:   32K
        (65536, 60),   # steps 80-139: 64K
        (131072, 60),  # steps 140-199: 128K
        (262144, 50),  # steps 200-249: 256K
    ),
    output_dir="checkpoints/1b_256k",
    run_name="1b-256k-run1",
)

# ── Build initial dataset at 32K ──
logger.info(f"Building FineWeb dataset at seq_len={mc.max_seq_len}...")
dcfg = DataConfig(max_seq_len=mc.max_seq_len, max_samples=5000)
ds = FineWebDataset(dcfg, tok)
logger.info(f"Dataset: {len(ds)} samples (seq_len={mc.max_seq_len})")

# ── Train ──
trainer = UltimateTrainer(mc, tc, dataset=ds)
trainer.train()
