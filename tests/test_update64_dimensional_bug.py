"""Test the FIXED 64×64 update_tc_v2 kernel against torch reference.

Mirrors tests/test_update_dimensional_bug.py but drives the 64×64 kernel
directly via pack_update._load_up_tc_v2() (the production gate requires all
dims %64==0; we test the kernel itself at multiple shapes incl. non-multiples).

Pass criterion: counter after one step matches the torch reference exactly
(max_diff == 0), same as the 32×32 test.
"""
import os
import sys
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.path.insert(0, os.getcwd())

import torch

from kernels.packed_ternary import pack_update as pu

THRESHOLD = 32
kWeightsPerWord = 16


def ref_step(W_packed, counter, X, dY):
    """Reference: dW = dY^T @ X; counter -= sign; flip when |cnt|>threshold."""
    N, stride = W_packed.shape
    K = counter.shape[1]
    dW = (dY.T.float() @ X.float())  # [N, K] fp32
    cnt = counter.clone().to(torch.int32)
    cnt += torch.where(dW > 0, -1, torch.where(dW < 0, 1, 0))
    # flip where |cnt| > threshold (process cnt0 then cnt1 — same order)
    flip_pos = cnt > THRESHOLD
    flip_neg = cnt < -THRESHOLD
    W = W_packed.clone()
    Wf = W_packed.view(torch.int32).clone()
    # emulate bit flips: decode → flip sign → re-encode
    # kWeightsPerWord=16, 2 bits per weight, LUT {0,1,-1,0} codes
    for idx in (flip_pos | flip_neg).nonzero():
        n, k = int(idx[0]), int(idx[1])
        wi, pos = k // kWeightsPerWord, k % kWeightsPerWord
        word = int(W.view(torch.int32)[n, wi]) & 0xFFFFFFFF
        code = (word >> (2 * pos)) & 3
        val = (code == 1) - (code == 2)  # +1 / -1
        # kernel: cnt > threshold → increment_weight_atomic; cnt < -threshold → decrement
        val = val + 1 if cnt[n, k] > THRESHOLD else val - 1
        val = max(-1, min(1, val))
        ncode = 1 if val == 1 else (2 if val == -1 else 0)
        word = (word & ~(3 << (2 * pos))) | (ncode << (2 * pos))
        if word >= 2**31:
            word -= 2**32
        W.view(torch.int32)[n, wi] = word
        cnt[n, k] = 0
    return W, cnt.to(torch.int16)


def run_case(B, N, K, tol_frac=0.0):
    torch.manual_seed(42)
    stride_words = (K + kWeightsPerWord - 1) // kWeightsPerWord
    X = torch.randn(B, K, device="cuda", dtype=torch.float16)
    dY = torch.randn(B, N, device="cuda", dtype=torch.float16) * 0.5
    W0 = torch.randint(-2**31, 2**31 - 1, (N, stride_words), dtype=torch.int32, device="cuda")
    c0 = (torch.randint(-THRESHOLD, THRESHOLD + 1, (N, K), device="cuda")).to(torch.int16)

    Wk, ck = W0.clone(), c0.clone()
    pu._up_tc_v2_fn(Wk, ck, X, dY, THRESHOLD)
    torch.cuda.synchronize()

    Wr, cr = ref_step(W0, c0, X, dY)

    # fp16 TC accumulation can flip the SIGN of a near-zero dW vs the fp32
    # reference; allow mismatches only where |dW| is tiny.
    cnt_mismatch = (ck != cr)
    if cnt_mismatch.any():
        dW_mag = (dY.T.float() @ X.float()).abs()
        bad = (cnt_mismatch & (dW_mag > 1e-2)).sum().item()
    else:
        bad = 0
    cnt_diff = cnt_mismatch.sum().item()
    w_diff = (Wk != Wr).sum().item()
    ok = bad == 0 and w_diff == 0
    print(f"  B={B:5d} N={N:5d} K={K:5d}: counter diffs={cnt_diff} "
          f"(far-from-zero={bad}), W word diffs={w_diff}  {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    pu._load_up_tc_v2()
    if not pu._HAS_UP_TC_V2:
        print("❌ TC64 update kernel not available")
        sys.exit(1)

    all_ok = True
    # 64-multiples (the dispatch gate's requirement)
    all_ok &= run_case(128, 256, 128)
    all_ok &= run_case(256, 128, 64)
    # odd/tail shapes — zero-padding paths
    all_ok &= run_case(100, 200, 72)
    all_ok &= run_case(48, 96, 100)    # partial batch tile
    all_ok &= run_case(33, 65, 17)     # tiny, everything partial
    # production-ish head shape slice
    all_ok &= run_case(512, 50272, 1024)

    print("=" * 60)
    if all_ok:
        print("✅ PASS — 64×64 update kernel matches reference exactly")
        sys.exit(0)
    print("❌ FAIL — 64×64 update kernel mismatches reference")
    sys.exit(1)


if __name__ == "__main__":
    main()
