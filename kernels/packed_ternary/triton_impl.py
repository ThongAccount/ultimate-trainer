"""Triton JIT kernels: packed ternary linear (fwd / bwd_dx / counter update).

Matches the CUDA stack's packed 2-bit ternary layout:
  - W packed int32 [N, stride_words], 16 codes/word little-endian (0→0,1→+1,2→−1)
  - counter int16 [N, K]
  - update: counter += sign(dW); flip when |counter| > threshold; reset.

Uses triton.jit only — no custom CUDA C++, no torch.utils.cpp_extension.

Semantics follow ``torch_impl.py`` (the oracle), NOT the drift in some .cu files:
the counter convention is ASCENT — ``counter += sign(dW)``, and a counter over
+threshold increments the ternary value (−1→0→+1), below −threshold decrements
(+1→0→−1), then resets.  (gemm_update.cu / gemm_update_tc*.cu use the opposite,
gradient-descent sign; per the task contract torch_impl wins.)

``gamma`` is accepted for signature parity with torch_impl but is a no-op here,
matching the CUDA forward kernels (gamma is folded into packing, not the GEMM).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def ternary_fwd_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_xb, stride_xk,
    stride_wn, stride_wk,   # W stride_words
    stride_yb, stride_yn,
    B, K, N, KWORDS,
    BLOCK_B: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Y[B,N] = X[B,K] @ W[N,K]^T ; W decoded from packed ternary."""
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    b_off = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_off = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        kk = k0 + k_off
        x = tl.load(X_ptr + b_off[:, None] * stride_xb + kk[None, :] * stride_xk,
                    mask=(b_off[:, None] < B) & (kk[None, :] < K), other=0.0)
        # W tile: [BLOCK_N, BLOCK_K_WORDS] decoded on the fly
        wk = kk // 16  # word index per column
        wmask = (n_off[:, None] < N) & (wk[None, :] < KWORDS)
        w = tl.load(W_ptr + n_off[:, None] * stride_wn + wk[None, :] * stride_wk,
                    mask=wmask, other=0)
        pos = kk % 16
        code = (w >> (2 * pos)) & 3
        wt = tl.where(code == 1, 1.0, tl.where(code == 2, -1.0, 0.0)).to(tl.float16)
        acc += tl.dot(x, tl.trans(wt))

    y = acc.to(tl.float16)
    y_off = b_off[:, None] * stride_yb + n_off[None, :] * stride_yn
    tl.store(Y_ptr + y_off, y, mask=(b_off[:, None] < B) & (n_off[None, :] < N))


@triton.jit
def ternary_bwd_dx_kernel(
    dY_ptr, W_ptr, dX_ptr,
    stride_dyb, stride_dyn,
    stride_wn, stride_wk,
    stride_dxb, stride_dxk,
    B, K, N, KWORDS,
    BLOCK_B: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """dX[B,K] = dY[B,N] @ W[N,K]."""
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    b_off = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    k_off = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_off = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)

    for n0 in range(0, N, BLOCK_N):
        nn = n0 + n_off
        dy = tl.load(dY_ptr + b_off[:, None] * stride_dyb + nn[None, :] * stride_dyn,
                     mask=(b_off[:, None] < B) & (nn[None, :] < N), other=0.0)
        wk = k_off // 16
        wmask = (nn[:, None] < N) & (wk[None, :] < KWORDS)
        w = tl.load(W_ptr + nn[:, None] * stride_wn + wk[None, :] * stride_wk,
                    mask=wmask, other=0)
        pos = k_off % 16
        code = (w >> (2 * pos)) & 3
        wt = tl.where(code == 1, 1.0, tl.where(code == 2, -1.0, 0.0)).to(tl.float16)
        acc += tl.dot(dy, wt)  # [BB, BN] @ [BN, BK]

    dx = acc.to(tl.float16)
    dx_off = b_off[:, None] * stride_dxb + k_off[None, :] * stride_dxk
    tl.store(dX_ptr + dx_off, dx, mask=(b_off[:, None] < B) & (k_off[None, :] < K))


@triton.jit
def ternary_update_kernel(
    X_ptr, dY_ptr, W_ptr, Cnt_ptr,
    stride_xb, stride_xk,
    stride_dyb, stride_dyn,
    stride_wn, stride_wk,
    stride_cn, stride_ck,
    B, K, N, KWORDS, THRESH,
    BLOCK_B: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """counter[N,K] += sign(Σ_b dY[b,n] X[b,k]); flip on |cnt| > THRESH.

    Tiled by (N, KWORDS): each program owns one whole 16-code word, so the
    packed-word flip is a plain store — no atomics, no cross-program races.

    Within a program, the 16 lanes (code positions) all target the SAME word
    address, so the per-position bit-diffs are OR-reduced across the 16 lanes
    into a single new word, then stored once per row.  (A per-lane scatter of
    differing full-word values to one address would be an undefined write race.)
    """
    pid_n = tl.program_id(0)
    pid_w = tl.program_id(1)

    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_off = pid_w * 16 + tl.arange(0, 16)          # [16] code positions in this word

    # dW[n,k] = Σ_b dY[b,n]·X[b,k] — FP32 accumulator, never materialized.
    acc = tl.zeros((BLOCK_N, 16), dtype=tl.float32)
    for b0 in range(0, B, BLOCK_B):
        bb = b0 + tl.arange(0, BLOCK_B)
        x = tl.load(X_ptr + bb[:, None] * stride_xb + k_off[None, :] * stride_xk,
                    mask=(bb[:, None] < B) & (k_off[None, :] < K), other=0.0)
        dy = tl.load(dY_ptr + bb[:, None] * stride_dyb + n_off[None, :] * stride_dyn,
                     mask=(bb[:, None] < B) & (n_off[None, :] < N), other=0.0)
        acc += tl.dot(tl.trans(dy), x)             # [BN,B] @ [B,16] → [BN,16]

    sign = tl.where(acc > 0, 1, tl.where(acc < 0, -1, 0)).to(tl.int16)

    cnt = tl.load(Cnt_ptr + n_off[:, None] * stride_cn + k_off[None, :] * stride_ck,
                  mask=(n_off[:, None] < N) & (k_off[None, :] < K), other=0)
    cnt = (cnt - sign).to(tl.int16)  # DESCENT: positive dW pushes weight down

    pos = cnt > THRESH
    neg = cnt < -THRESH

    # Owned word: one address per row (same for all 16 lanes).
    word = tl.load(W_ptr + n_off * stride_wn + pid_w * stride_wk,
                   mask=(n_off < N), other=0)      # [BN]
    posk = k_off % 16                              # [16]
    code = (word[:, None] >> (2 * posk[None, :])) & 3
    val = tl.where(code == 1, 1, tl.where(code == 2, -1, 0))

    new_val = tl.where(pos, tl.minimum(val + 1, 1),
              tl.where(neg, tl.maximum(val - 1, -1), val))
    new_code = tl.where(new_val == 0, 0, tl.where(new_val == 1, 1, 2))

    # Bits flip only where the code changed.  Positions are disjoint 2-bit
    # slots, so sum (== XOR/OR) over the 16 lanes yields one coherent word.
    changed = (new_code != code) & (n_off[:, None] < N) & (k_off[None, :] < K)
    diff = tl.where(changed, (code ^ new_code) << (2 * posk[None, :]), 0)
    word_diff = tl.sum(diff, axis=1)               # [BN]
    new_word = word ^ word_diff

    tl.store(W_ptr + n_off * stride_wn + pid_w * stride_wk, new_word,
             mask=(n_off < N))

    cnt = tl.where(pos | neg, tl.zeros_like(cnt), cnt)
    tl.store(Cnt_ptr + n_off[:, None] * stride_cn + k_off[None, :] * stride_ck,
             cnt, mask=(n_off[:, None] < N) & (k_off[None, :] < K))


def ternary_forward(packed, X, K, gamma=1.0):
    B, N = X.shape[0], packed.shape[0]
    kw = packed.shape[1]
    Y = torch.empty(B, N, dtype=torch.float16, device=X.device)
    BK, BN = 64, 64
    grid = (triton.cdiv(B, BK), triton.cdiv(N, BN))
    ternary_fwd_kernel[grid](
        X, packed, Y,
        X.stride(0), X.stride(1), packed.stride(0), packed.stride(1),
        Y.stride(0), Y.stride(1),
        B, K, N, kw,
        BLOCK_B=BK, BLOCK_N=BN, BLOCK_K=64,
    )
    return Y


def ternary_backward_dx(packed, dY, K, gamma=1.0):
    B, N = dY.shape[0], packed.shape[0]
    kw = packed.shape[1]
    dX = torch.empty(B, K, dtype=torch.float16, device=dY.device)
    BK, BKk, BN = 64, 64, 64
    grid = (triton.cdiv(B, BK), triton.cdiv(K, BKk))
    ternary_bwd_dx_kernel[grid](
        dY, packed, dX,
        dY.stride(0), dY.stride(1), packed.stride(0), packed.stride(1),
        dX.stride(0), dX.stride(1),
        B, K, N, kw,
        BLOCK_B=BK, BLOCK_K=BKk, BLOCK_N=BN,
    )
    return dX


def ternary_update(packed, counter, X, dY, threshold, K):
    B, N = dY.shape[0], packed.shape[0]
    kw = packed.shape[1]
    BK, BN = 32, 32
    grid = (triton.cdiv(N, BN), kw)
    ternary_update_kernel[grid](
        X, dY, packed, counter,
        X.stride(0), X.stride(1), dY.stride(0), dY.stride(1),
        packed.stride(0), packed.stride(1), counter.stride(0), counter.stride(1),
        B, K, N, kw, threshold,
        BLOCK_B=BK, BLOCK_N=BN,
    )
    return packed


def ternary_backward_update(packed, counter, dY, X, threshold, K):
    """Fused dX backward + counter update (mirrors pack_update.backward_update).

    dX = dY @ W is computed from the pre-update packed snapshot (as the CUDA
    fused kernel reads W_read before mutating W_mut), then W_packed and counter
    are updated in place.  Returns dX for the upstream layer.
    """
    dX = ternary_backward_dx(packed, dY, K)
    ternary_update(packed, counter, X, dY, threshold, K)
    return dX
