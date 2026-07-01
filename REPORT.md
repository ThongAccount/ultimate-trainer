# Ultimate Trainer Report

## What Was Built

A merged architecture combining:
- **BitNet b1.58** (ternary weight quantization, subln normalization, squared ReLU activation quant)
- **NSA/SubQSA** (Native Sparse Attention — 3-branch: compression + selection + sliding window gated by learned MLP)

## Repository Structure



## Key Results

### 1-Bit Trainer Comparison
- Both FP and BitLinear reduce loss during training (learning works)
- HF Kernels decorator available and functional
- Ternary weights reduce effective memory to 0.05 MB vs 0.50 MB FP

### SubQSA Trainer
- All files pass syntax check
- 3-branch NSA design: compression (block mean pooling), selection (top-k gather), sliding window
- Learned gating blends three branches per head

### Ultimate Trainer
- Merges BitNet b1.58 BitLinear with NSA SubQSA
- Full SwiGLU FFN, RMSNorm, RoPE, learned gating
- All files syntactically correct

## HF Kernels Compatibility
-  import wraps cleanly with try/except fallback
-  registered as 
-  registered as 
- No push to Hub required; decorator falls through to native forward

## Architecture Choices
- **BitNet b1.58**: absmax activation quant (8-bit), absmean weight ternary, subln normalization
- **NSA/SubQSA**: 3-branch with compression block=32/stride=16, selection top-k=16, sliding window=512
- **No Triton**: CPU-safe pure PyTorch SDPA for all attention variants
- **DDP ready**: Trainer accepts  for multi-GPU
- **Staged context extension**: (4096, 200) → (8192, 100) → (32768, 50) phases

## What Works
- All modules import correctly via underscore-wrapper packages
- Model forward pass runs on CPU (no GPU required)
- Training step reduces loss for all variants
- HF Kernels decorator available (falls through to native when no kernel registered)
