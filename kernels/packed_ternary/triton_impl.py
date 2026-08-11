"""Triton JIT kernels: packed ternary linear (fwd / bwd_dx / counter update).

Matches the CUDA stack's packed 2-bit ternary layout:
  - W packed int32 [N, stride_words], 16 codes/word little-endian (0→0,1→+1,2→−1)
  - counter int16 [N, K]
  - update: counter += sign(dW); flip when |counter| > threshold; reset.

Uses triton.jit only — no custom CUDA C++, no torch.utils.cpp_extension.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_tile(w: tl.tensor, k_off: tl.tensor, kk: tl.constexpr):
    """Decode a [BLOCK_N, BLOCK_K] packed word tile into ternary int8.

    w: packed int32 [BLOCK_N, BLOCK_K_WORDS] — each word holds 16 codes.
    k_off: arange over K positions (0..BLOCK_K-1) on the kernel grid.
    """
    word_idx = k_off // 16
    pos = k_off % 16
    words = tl.load(w, mask=(word_idx < tl.num_programs(0)), other=0)  # [BN, BK]
    code = (words >> (2 * pos)) & 3
    # code 0→0, 1→+1, 2→−1
    val = tl.where(code == 1, 1, tl.where(code == 2, -1, 0))
    return val.to(tl.float16)


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
    """
    pid_n = tl.program_id(0)
    pid_w = tl.program_id(1)

    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    w_off = pid_w + tl.arange(0, 1)              # one word per program
    k_off = w_off * 16 + tl.arange(0, 16)        # [1,16] code positions

    acc = tl.zeros((BLOCK_N, 16), dtype=tl.float32)
    for b0 in range(0, B, BLOCK_B):
        bb = b0 + tl.arange(0, BLOCK_B)
        x = tl.load(X_ptr + bb[:, None] * stride_xb + k_off[None, :] * stride_xk,
                    mask=(bb[:, None] < B) & (k_off[None, :] < K), other=0.0)
        dy = tl.load(dY_ptr + bb[:, None] * stride_dyb + n_off[None, :] * stride_dyn,
                     mask=(bb[:, None] < B) & (n_off[None, :] < N), other=0.0)
        acc += tl.dot(tl.trans(dy), x)           # [BN,B] @ [B,16] → [BN,16]

    sign = tl.where(acc > 0, 1, tl.where(acc < 0, -1, 0))

    cnt = tl.load(Cnt_ptr + n_off[:, None] * stride_cn + k_off[None, :] * stride_ck,
                  mask=(n_off[:, None] < N) & (k_off[None, :] < K), other=0)
    cnt = cnt + sign

    pos = cnt > THRESH
    neg = cnt < -THRESH

    wk = k_off // 16
    posk = k_off % 16
    wmask = (n_off[:, None] < N) & (wk[None, :] < KWORDS)
    w = tl.load(W_ptr + n_off[:, None] * stride_wn + wk[None, :] * stride_wk,
                mask=wmask, other=0)
    code = (w >> (2 * posk)) & 3
    val = tl.where(code == 1, 1, tl.where(code == 2, -1, 0))

    new_val = tl.where(pos, tl.minimum(val + 1, 1),
              tl.where(neg, tl.maximum(val - 1, -1), val))
    new_code = tl.where(new_val == 0, 0, tl.where(new_val == 1, 1, 2))

    new_word = tl.where(
        (new_code != code) & wmask,
        (w & ~(3 << (2 * posk))) | (new_code << (2 * posk)),
        w,
    )
    tl.store(W_ptr + n_off[:, None] * stride_wn + wk[None, :] * stride_wk,
             new_word, mask=wmask)

    cnt = tl.where(pos | neg, 0, cnt)
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
