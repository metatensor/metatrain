"""Memory-efficient attention with first and second backward support in Triton.

Supported surface:
- CUDA tensors
- query/key/value shaped [B, H, L_or_S, D]
- optional per-token additive attention weights shaped [B, S]
- non-causal attention
- no dropout

The implementation stores output and per-row logsumexp from the forward pass.
Backward and backward-of-backward recompute softmax tiles instead of
materializing the full [B, H, L, S] attention matrix.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _triton_attention_fwd_kernel(
    query,
    key,
    value,
    bias,
    output,
    logsumexp,
    batch: tl.constexpr,
    heads: tl.constexpr,
    query_len: tl.constexpr,
    key_len: tl.constexpr,
    head_dim: tl.constexpr,
    sm_scale: tl.constexpr,
    has_bias: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_id = pid_bh // heads

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, head_dim)

    q_ptrs = query + (pid_bh * query_len + offs_m[:, None]) * head_dim + offs_d[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < query_len, other=0.0)

    row_max = tl.full((block_m,), -float("inf"), tl.float32)
    row_sum = tl.zeros((block_m,), tl.float32)
    acc = tl.zeros((block_m, head_dim), tl.float32)

    for start_n in range(0, key_len, block_n):
        cols = start_n + offs_n
        k_ptrs = key + (pid_bh * key_len + cols[:, None]) * head_dim + offs_d[None, :]
        v_ptrs = value + (pid_bh * key_len + cols[:, None]) * head_dim + offs_d[None, :]
        k = tl.load(k_ptrs, mask=cols[:, None] < key_len, other=0.0)
        v = tl.load(v_ptrs, mask=cols[:, None] < key_len, other=0.0)

        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
        if has_bias:
            token_bias = tl.load(bias + batch_id * key_len + cols, mask=cols < key_len, other=0.0)
            scores += token_bias[None, :]

        scores = tl.where((offs_m[:, None] < query_len) & (cols[None, :] < key_len), scores, -float("inf"))
        new_row_max = tl.maximum(row_max, tl.max(scores, axis=1))
        alpha = tl.exp(row_max - new_row_max)
        probs = tl.exp(scores - new_row_max[:, None])
        new_row_sum = row_sum * alpha + tl.sum(probs, axis=1)
        acc = acc * alpha[:, None] + tl.dot(probs, v, input_precision="ieee")
        row_max = new_row_max
        row_sum = new_row_sum

    out = acc / row_sum[:, None]
    lse = row_max + tl.log(row_sum)

    out_ptrs = output + (pid_bh * query_len + offs_m[:, None]) * head_dim + offs_d[None, :]
    lse_ptrs = logsumexp + pid_bh * query_len + offs_m
    tl.store(out_ptrs, out, mask=offs_m[:, None] < query_len)
    tl.store(lse_ptrs, lse, mask=offs_m < query_len)


@triton.jit
def _triton_attention_bwd_dkdv_kernel(
    query,
    key,
    value,
    bias,
    grad_output,
    logsumexp,
    delta,
    grad_key,
    grad_value,
    grad_bias,
    batch: tl.constexpr,
    heads: tl.constexpr,
    query_len: tl.constexpr,
    key_len: tl.constexpr,
    head_dim: tl.constexpr,
    sm_scale: tl.constexpr,
    has_bias: tl.constexpr,
    bias_requires_grad: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_id = pid_bh // heads

    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_m = tl.arange(0, block_m)
    offs_d = tl.arange(0, head_dim)

    k_ptrs = key + (pid_bh * key_len + offs_n[:, None]) * head_dim + offs_d[None, :]
    v_ptrs = value + (pid_bh * key_len + offs_n[:, None]) * head_dim + offs_d[None, :]
    k = tl.load(k_ptrs, mask=offs_n[:, None] < key_len, other=0.0)
    v = tl.load(v_ptrs, mask=offs_n[:, None] < key_len, other=0.0)

    dk = tl.zeros((block_n, head_dim), tl.float32)
    dv = tl.zeros((block_n, head_dim), tl.float32)
    db = tl.zeros((block_n,), tl.float32)

    if has_bias:
        token_bias = tl.load(bias + batch_id * key_len + offs_n, mask=offs_n < key_len, other=0.0)

    for start_m in range(0, query_len, block_m):
        rows = start_m + offs_m
        q_ptrs = query + (pid_bh * query_len + rows[:, None]) * head_dim + offs_d[None, :]
        do_ptrs = grad_output + (pid_bh * query_len + rows[:, None]) * head_dim + offs_d[None, :]
        q = tl.load(q_ptrs, mask=rows[:, None] < query_len, other=0.0)
        do = tl.load(do_ptrs, mask=rows[:, None] < query_len, other=0.0)
        lse = tl.load(logsumexp + pid_bh * query_len + rows, mask=rows < query_len, other=0.0)
        d = tl.load(delta + pid_bh * query_len + rows, mask=rows < query_len, other=0.0)

        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
        if has_bias:
            scores += token_bias[None, :]
        scores = tl.where((rows[:, None] < query_len) & (offs_n[None, :] < key_len), scores, -float("inf"))
        p = tl.exp(scores - lse[:, None])
        p = tl.where((rows[:, None] < query_len) & (offs_n[None, :] < key_len), p, 0.0)

        dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        ds = p * (dp - d[:, None])
        ds = tl.where((rows[:, None] < query_len) & (offs_n[None, :] < key_len), ds, 0.0)

        dv += tl.dot(tl.trans(p), do, input_precision="ieee")
        dk += tl.dot(tl.trans(ds), q, input_precision="ieee") * sm_scale
        db += tl.sum(ds, axis=0)

    dk_ptrs = grad_key + (pid_bh * key_len + offs_n[:, None]) * head_dim + offs_d[None, :]
    dv_ptrs = grad_value + (pid_bh * key_len + offs_n[:, None]) * head_dim + offs_d[None, :]
    tl.store(dk_ptrs, dk, mask=offs_n[:, None] < key_len)
    tl.store(dv_ptrs, dv, mask=offs_n[:, None] < key_len)

    if has_bias and bias_requires_grad:
        tl.atomic_add(grad_bias + batch_id * key_len + offs_n, db, sem="relaxed", mask=offs_n < key_len)


@triton.jit
def _triton_attention_bwd_dq_kernel(
    query,
    key,
    value,
    bias,
    grad_output,
    logsumexp,
    delta,
    grad_query,
    batch: tl.constexpr,
    heads: tl.constexpr,
    query_len: tl.constexpr,
    key_len: tl.constexpr,
    head_dim: tl.constexpr,
    sm_scale: tl.constexpr,
    has_bias: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_id = pid_bh // heads

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, head_dim)

    q_ptrs = query + (pid_bh * query_len + offs_m[:, None]) * head_dim + offs_d[None, :]
    do_ptrs = grad_output + (pid_bh * query_len + offs_m[:, None]) * head_dim + offs_d[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < query_len, other=0.0)
    do = tl.load(do_ptrs, mask=offs_m[:, None] < query_len, other=0.0)
    lse = tl.load(logsumexp + pid_bh * query_len + offs_m, mask=offs_m < query_len, other=0.0)
    d = tl.load(delta + pid_bh * query_len + offs_m, mask=offs_m < query_len, other=0.0)

    dq = tl.zeros((block_m, head_dim), tl.float32)

    for start_n in range(0, key_len, block_n):
        cols = start_n + offs_n
        k_ptrs = key + (pid_bh * key_len + cols[:, None]) * head_dim + offs_d[None, :]
        v_ptrs = value + (pid_bh * key_len + cols[:, None]) * head_dim + offs_d[None, :]
        k = tl.load(k_ptrs, mask=cols[:, None] < key_len, other=0.0)
        v = tl.load(v_ptrs, mask=cols[:, None] < key_len, other=0.0)

        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
        if has_bias:
            token_bias = tl.load(bias + batch_id * key_len + cols, mask=cols < key_len, other=0.0)
            scores += token_bias[None, :]
        scores = tl.where((offs_m[:, None] < query_len) & (cols[None, :] < key_len), scores, -float("inf"))
        p = tl.exp(scores - lse[:, None])
        p = tl.where((offs_m[:, None] < query_len) & (cols[None, :] < key_len), p, 0.0)

        dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        ds = p * (dp - d[:, None])
        ds = tl.where((offs_m[:, None] < query_len) & (cols[None, :] < key_len), ds, 0.0)
        dq += tl.dot(ds, k, input_precision="ieee") * sm_scale

    dq_ptrs = grad_query + (pid_bh * query_len + offs_m[:, None]) * head_dim + offs_d[None, :]
    tl.store(dq_ptrs, dq, mask=offs_m[:, None] < query_len)


@triton.jit
def _triton_attention_bwd2_preprocess_kernel(
    query,
    key,
    value,
    bias,
    grad_output,
    output,
    logsumexp,
    grad2_query,
    grad2_key,
    grad2_value,
    grad2_bias,
    row_delta,
    row_rbar,
    row_mean,
    batch: tl.constexpr,
    heads: tl.constexpr,
    query_len: tl.constexpr,
    key_len: tl.constexpr,
    head_dim: tl.constexpr,
    sm_scale: tl.constexpr,
    has_bias: tl.constexpr,
    has_grad2_bias: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_id = pid_bh // heads

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, head_dim)

    q = tl.load(query + (pid_bh * query_len + offs_m[:, None]) * head_dim + offs_d[None, :], mask=offs_m[:, None] < query_len, other=0.0)
    go = tl.load(grad_output + (pid_bh * query_len + offs_m[:, None]) * head_dim + offs_d[None, :], mask=offs_m[:, None] < query_len, other=0.0)
    out = tl.load(output + (pid_bh * query_len + offs_m[:, None]) * head_dim + offs_d[None, :], mask=offs_m[:, None] < query_len, other=0.0)
    g2q = tl.load(grad2_query + (pid_bh * query_len + offs_m[:, None]) * head_dim + offs_d[None, :], mask=offs_m[:, None] < query_len, other=0.0)
    lse = tl.load(logsumexp + pid_bh * query_len + offs_m, mask=offs_m < query_len, other=0.0)
    delta = tl.sum(go * out, axis=1)

    rbar = tl.zeros((block_m,), tl.float32)
    prx = tl.zeros((block_m,), tl.float32)
    mean_w = tl.zeros((block_m,), tl.float32)

    for start_n in range(0, key_len, block_n):
        cols = start_n + offs_n
        k = tl.load(key + (pid_bh * key_len + cols[:, None]) * head_dim + offs_d[None, :], mask=cols[:, None] < key_len, other=0.0)
        v = tl.load(value + (pid_bh * key_len + cols[:, None]) * head_dim + offs_d[None, :], mask=cols[:, None] < key_len, other=0.0)
        g2k = tl.load(grad2_key + (pid_bh * key_len + cols[:, None]) * head_dim + offs_d[None, :], mask=cols[:, None] < key_len, other=0.0)
        g2v = tl.load(grad2_value + (pid_bh * key_len + cols[:, None]) * head_dim + offs_d[None, :], mask=cols[:, None] < key_len, other=0.0)

        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
        if has_bias:
            token_bias = tl.load(bias + batch_id * key_len + cols, mask=cols < key_len, other=0.0)
            scores += token_bias[None, :]
        scores = tl.where((offs_m[:, None] < query_len) & (cols[None, :] < key_len), scores, -float("inf"))
        p = tl.exp(scores - lse[:, None])
        p = tl.where((offs_m[:, None] < query_len) & (cols[None, :] < key_len), p, 0.0)

        x = tl.dot(go, tl.trans(v), input_precision="ieee")
        r = tl.dot(g2q, tl.trans(k), input_precision="ieee") * sm_scale
        r += tl.dot(q, tl.trans(g2k), input_precision="ieee") * sm_scale
        if has_grad2_bias:
            r += tl.load(grad2_bias + batch_id * key_len + cols, mask=cols < key_len, other=0.0)[None, :]
        w = tl.dot(go, tl.trans(g2v), input_precision="ieee")

        rbar += tl.sum(p * r, axis=1)
        prx += tl.sum(p * r * x, axis=1)
        mean_w += tl.sum(p * w, axis=1)

    mean = prx - 2.0 * delta * rbar + mean_w
    tl.store(row_delta + pid_bh * query_len + offs_m, delta, mask=offs_m < query_len)
    tl.store(row_rbar + pid_bh * query_len + offs_m, rbar, mask=offs_m < query_len)
    tl.store(row_mean + pid_bh * query_len + offs_m, mean, mask=offs_m < query_len)


@triton.jit
def _triton_attention_bwd2_dq_dgo_kernel(
    query,
    key,
    value,
    bias,
    grad_output,
    logsumexp,
    grad2_query,
    grad2_key,
    grad2_value,
    grad2_bias,
    row_delta,
    row_rbar,
    row_mean,
    grad_query,
    grad_grad_output,
    batch: tl.constexpr,
    heads: tl.constexpr,
    query_len: tl.constexpr,
    key_len: tl.constexpr,
    head_dim: tl.constexpr,
    sm_scale: tl.constexpr,
    has_bias: tl.constexpr,
    has_grad2_bias: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_id = pid_bh // heads

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, head_dim)

    q = tl.load(query + (pid_bh * query_len + offs_m[:, None]) * head_dim + offs_d[None, :], mask=offs_m[:, None] < query_len, other=0.0)
    go = tl.load(grad_output + (pid_bh * query_len + offs_m[:, None]) * head_dim + offs_d[None, :], mask=offs_m[:, None] < query_len, other=0.0)
    g2q = tl.load(grad2_query + (pid_bh * query_len + offs_m[:, None]) * head_dim + offs_d[None, :], mask=offs_m[:, None] < query_len, other=0.0)
    lse = tl.load(logsumexp + pid_bh * query_len + offs_m, mask=offs_m < query_len, other=0.0)
    delta = tl.load(row_delta + pid_bh * query_len + offs_m, mask=offs_m < query_len, other=0.0)
    rbar = tl.load(row_rbar + pid_bh * query_len + offs_m, mask=offs_m < query_len, other=0.0)
    mean = tl.load(row_mean + pid_bh * query_len + offs_m, mask=offs_m < query_len, other=0.0)

    dq = tl.zeros((block_m, head_dim), tl.float32)
    dgo = tl.zeros((block_m, head_dim), tl.float32)

    for start_n in range(0, key_len, block_n):
        cols = start_n + offs_n
        k = tl.load(key + (pid_bh * key_len + cols[:, None]) * head_dim + offs_d[None, :], mask=cols[:, None] < key_len, other=0.0)
        v = tl.load(value + (pid_bh * key_len + cols[:, None]) * head_dim + offs_d[None, :], mask=cols[:, None] < key_len, other=0.0)
        g2k = tl.load(grad2_key + (pid_bh * key_len + cols[:, None]) * head_dim + offs_d[None, :], mask=cols[:, None] < key_len, other=0.0)
        g2v = tl.load(grad2_value + (pid_bh * key_len + cols[:, None]) * head_dim + offs_d[None, :], mask=cols[:, None] < key_len, other=0.0)

        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
        if has_bias:
            token_bias = tl.load(bias + batch_id * key_len + cols, mask=cols < key_len, other=0.0)
            scores += token_bias[None, :]
        scores = tl.where((offs_m[:, None] < query_len) & (cols[None, :] < key_len), scores, -float("inf"))
        p = tl.exp(scores - lse[:, None])
        p = tl.where((offs_m[:, None] < query_len) & (cols[None, :] < key_len), p, 0.0)

        x = tl.dot(go, tl.trans(v), input_precision="ieee")
        r = tl.dot(g2q, tl.trans(k), input_precision="ieee") * sm_scale
        r += tl.dot(q, tl.trans(g2k), input_precision="ieee") * sm_scale
        if has_grad2_bias:
            r += tl.load(grad2_bias + batch_id * key_len + cols, mask=cols < key_len, other=0.0)[None, :]
        w = tl.dot(go, tl.trans(g2v), input_precision="ieee")

        ds = p * (x - delta[:, None])
        t = p * (r - rbar[:, None])
        y = r * (x - delta[:, None]) - rbar[:, None] * x
        s2 = p * (y + w - mean[:, None])
        ds = tl.where((offs_m[:, None] < query_len) & (cols[None, :] < key_len), ds, 0.0)
        t = tl.where((offs_m[:, None] < query_len) & (cols[None, :] < key_len), t, 0.0)
        s2 = tl.where((offs_m[:, None] < query_len) & (cols[None, :] < key_len), s2, 0.0)

        dq += tl.dot(s2, k, input_precision="ieee") * sm_scale
        dq += tl.dot(ds, g2k, input_precision="ieee") * sm_scale
        dgo += tl.dot(t, v, input_precision="ieee")
        dgo += tl.dot(p, g2v, input_precision="ieee")

    tl.store(grad_query + (pid_bh * query_len + offs_m[:, None]) * head_dim + offs_d[None, :], dq, mask=offs_m[:, None] < query_len)
    tl.store(grad_grad_output + (pid_bh * query_len + offs_m[:, None]) * head_dim + offs_d[None, :], dgo, mask=offs_m[:, None] < query_len)


@triton.jit
def _triton_attention_bwd2_dk_dv_db_kernel(
    query,
    key,
    value,
    bias,
    grad_output,
    logsumexp,
    grad2_query,
    grad2_key,
    grad2_value,
    grad2_bias,
    row_delta,
    row_rbar,
    row_mean,
    grad_key,
    grad_value,
    grad_bias,
    batch: tl.constexpr,
    heads: tl.constexpr,
    query_len: tl.constexpr,
    key_len: tl.constexpr,
    head_dim: tl.constexpr,
    sm_scale: tl.constexpr,
    has_bias: tl.constexpr,
    bias_requires_grad: tl.constexpr,
    has_grad2_bias: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_id = pid_bh // heads

    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_m = tl.arange(0, block_m)
    offs_d = tl.arange(0, head_dim)

    k = tl.load(key + (pid_bh * key_len + offs_n[:, None]) * head_dim + offs_d[None, :], mask=offs_n[:, None] < key_len, other=0.0)
    v = tl.load(value + (pid_bh * key_len + offs_n[:, None]) * head_dim + offs_d[None, :], mask=offs_n[:, None] < key_len, other=0.0)
    g2k = tl.load(grad2_key + (pid_bh * key_len + offs_n[:, None]) * head_dim + offs_d[None, :], mask=offs_n[:, None] < key_len, other=0.0)
    g2v = tl.load(grad2_value + (pid_bh * key_len + offs_n[:, None]) * head_dim + offs_d[None, :], mask=offs_n[:, None] < key_len, other=0.0)

    dk = tl.zeros((block_n, head_dim), tl.float32)
    dv = tl.zeros((block_n, head_dim), tl.float32)
    db = tl.zeros((block_n,), tl.float32)
    if has_bias:
        token_bias = tl.load(bias + batch_id * key_len + offs_n, mask=offs_n < key_len, other=0.0)
    if has_grad2_bias:
        g2b = tl.load(grad2_bias + batch_id * key_len + offs_n, mask=offs_n < key_len, other=0.0)

    for start_m in range(0, query_len, block_m):
        rows = start_m + offs_m
        q = tl.load(query + (pid_bh * query_len + rows[:, None]) * head_dim + offs_d[None, :], mask=rows[:, None] < query_len, other=0.0)
        go = tl.load(grad_output + (pid_bh * query_len + rows[:, None]) * head_dim + offs_d[None, :], mask=rows[:, None] < query_len, other=0.0)
        g2q = tl.load(grad2_query + (pid_bh * query_len + rows[:, None]) * head_dim + offs_d[None, :], mask=rows[:, None] < query_len, other=0.0)
        lse = tl.load(logsumexp + pid_bh * query_len + rows, mask=rows < query_len, other=0.0)
        delta = tl.load(row_delta + pid_bh * query_len + rows, mask=rows < query_len, other=0.0)
        rbar = tl.load(row_rbar + pid_bh * query_len + rows, mask=rows < query_len, other=0.0)
        mean = tl.load(row_mean + pid_bh * query_len + rows, mask=rows < query_len, other=0.0)

        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
        if has_bias:
            scores += token_bias[None, :]
        scores = tl.where((rows[:, None] < query_len) & (offs_n[None, :] < key_len), scores, -float("inf"))
        p = tl.exp(scores - lse[:, None])
        p = tl.where((rows[:, None] < query_len) & (offs_n[None, :] < key_len), p, 0.0)

        x = tl.dot(go, tl.trans(v), input_precision="ieee")
        r = tl.dot(g2q, tl.trans(k), input_precision="ieee") * sm_scale
        r += tl.dot(q, tl.trans(g2k), input_precision="ieee") * sm_scale
        if has_grad2_bias:
            r += g2b[None, :]
        w = tl.dot(go, tl.trans(g2v), input_precision="ieee")

        ds = p * (x - delta[:, None])
        t = p * (r - rbar[:, None])
        y = r * (x - delta[:, None]) - rbar[:, None] * x
        s2 = p * (y + w - mean[:, None])
        ds = tl.where((rows[:, None] < query_len) & (offs_n[None, :] < key_len), ds, 0.0)
        t = tl.where((rows[:, None] < query_len) & (offs_n[None, :] < key_len), t, 0.0)
        s2 = tl.where((rows[:, None] < query_len) & (offs_n[None, :] < key_len), s2, 0.0)

        dk += tl.dot(tl.trans(s2), q, input_precision="ieee") * sm_scale
        dk += tl.dot(tl.trans(ds), g2q, input_precision="ieee") * sm_scale
        dv += tl.dot(tl.trans(t), go, input_precision="ieee")
        db += tl.sum(s2, axis=0)

    tl.store(grad_key + (pid_bh * key_len + offs_n[:, None]) * head_dim + offs_d[None, :], dk, mask=offs_n[:, None] < key_len)
    tl.store(grad_value + (pid_bh * key_len + offs_n[:, None]) * head_dim + offs_d[None, :], dv, mask=offs_n[:, None] < key_len)
    if has_bias and bias_requires_grad:
        tl.atomic_add(grad_bias + batch_id * key_len + offs_n, db, sem="relaxed", mask=offs_n < key_len)


class TritonMemoryEfficientAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_weights: torch.Tensor | None,
        block_m: int,
        block_n: int,
    ) -> torch.Tensor:
        if not query.is_cuda:
            raise RuntimeError("Triton attention requires CUDA tensors.")
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise RuntimeError("Expected query/key/value with shape [B, H, L_or_S, D].")
        if query.shape[0] != key.shape[0] or query.shape[1] != key.shape[1] or query.shape[-1] != key.shape[-1]:
            raise RuntimeError("query, key, and value must agree on batch, heads, and head_dim.")
        if key.shape != value.shape:
            raise RuntimeError("key and value must have identical shape.")
        if attention_weights is not None and attention_weights.shape != (query.shape[0], key.shape[2]):
            raise RuntimeError("Triton backend only supports per-token attention_weights with shape [B, S].")

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        has_bias = attention_weights is not None
        bias_requires_grad = has_bias and attention_weights.requires_grad
        if has_bias:
            attention_weights = attention_weights.contiguous()
            bias_for_kernel = attention_weights
        else:
            bias_for_kernel = torch.empty((0,), device=query.device, dtype=query.dtype)

        batch, heads, query_len, head_dim = query.shape
        key_len = key.shape[2]
        sm_scale = 1.0 / math.sqrt(head_dim)
        output = torch.empty_like(query)
        logsumexp = torch.empty((batch, heads, query_len), device=query.device, dtype=torch.float32)

        grid = (triton.cdiv(query_len, block_m), batch * heads)
        _triton_attention_fwd_kernel[grid](
            query,
            key,
            value,
            bias_for_kernel,
            output,
            logsumexp,
            batch,
            heads,
            query_len,
            key_len,
            head_dim,
            sm_scale,
            has_bias,
            block_m,
            block_n,
            num_warps=4,
        )

        ctx.save_for_backward(query, key, value, bias_for_kernel, output, logsumexp)
        ctx.block_m = block_m
        ctx.block_n = block_n
        ctx.sm_scale = sm_scale
        ctx.has_bias = has_bias
        ctx.bias_requires_grad = bias_requires_grad
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        query, key, value, bias_for_kernel, output, logsumexp = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_query, grad_key, grad_value, grad_attention_weights = TritonMemoryEfficientAttentionBackward.apply(
            query,
            key,
            value,
            bias_for_kernel,
            grad_output,
            output,
            logsumexp,
            ctx.has_bias,
            ctx.bias_requires_grad,
            ctx.block_m,
            ctx.block_n,
            ctx.sm_scale,
        )
        if not ctx.has_bias or not ctx.bias_requires_grad:
            grad_attention_weights = None

        return grad_query, grad_key, grad_value, grad_attention_weights, None, None


class TritonMemoryEfficientAttentionBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bias_for_kernel: torch.Tensor,
        grad_output: torch.Tensor,
        output: torch.Tensor,
        logsumexp: torch.Tensor,
        has_bias: bool,
        bias_requires_grad: bool,
        block_m: int,
        block_n: int,
        sm_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, heads, query_len, head_dim = query.shape
        key_len = key.shape[2]

        grad_query = torch.empty_like(query)
        grad_key = torch.empty_like(key)
        grad_value = torch.empty_like(value)
        grad_bias = torch.zeros_like(bias_for_kernel)
        delta = torch.sum(output * grad_output, dim=-1).to(torch.float32)

        grid_kv = (triton.cdiv(key_len, block_n), batch * heads)
        _triton_attention_bwd_dkdv_kernel[grid_kv](
            query,
            key,
            value,
            bias_for_kernel,
            grad_output,
            logsumexp,
            delta,
            grad_key,
            grad_value,
            grad_bias,
            batch,
            heads,
            query_len,
            key_len,
            head_dim,
            sm_scale,
            has_bias,
            bias_requires_grad,
            block_m,
            block_n,
            num_warps=4,
        )

        grid_q = (triton.cdiv(query_len, block_m), batch * heads)
        _triton_attention_bwd_dq_kernel[grid_q](
            query,
            key,
            value,
            bias_for_kernel,
            grad_output,
            logsumexp,
            delta,
            grad_query,
            batch,
            heads,
            query_len,
            key_len,
            head_dim,
            sm_scale,
            has_bias,
            block_m,
            block_n,
            num_warps=4,
        )

        ctx.save_for_backward(query, key, value, bias_for_kernel, grad_output, output, logsumexp)
        ctx.has_bias = has_bias
        ctx.bias_requires_grad = bias_requires_grad
        ctx.block_m = block_m
        ctx.block_n = block_n
        ctx.sm_scale = sm_scale
        return grad_query, grad_key, grad_value, grad_bias

    @staticmethod
    def backward(
        ctx,
        grad2_query: torch.Tensor | None,
        grad2_key: torch.Tensor | None,
        grad2_value: torch.Tensor | None,
        grad2_bias: torch.Tensor | None,
    ):
        query, key, value, bias_for_kernel, grad_output, output, logsumexp = ctx.saved_tensors
        batch, heads, query_len, head_dim = query.shape
        key_len = key.shape[2]
        block_m = ctx.block_m
        block_n = ctx.block_n

        if grad2_query is None:
            grad2_query = torch.zeros_like(query)
        else:
            grad2_query = grad2_query.contiguous()
        if grad2_key is None:
            grad2_key = torch.zeros_like(key)
        else:
            grad2_key = grad2_key.contiguous()
        if grad2_value is None:
            grad2_value = torch.zeros_like(value)
        else:
            grad2_value = grad2_value.contiguous()

        has_grad2_bias = grad2_bias is not None and ctx.has_bias
        if has_grad2_bias:
            grad2_bias_for_kernel = grad2_bias.contiguous()
        else:
            grad2_bias_for_kernel = torch.empty((0,), device=query.device, dtype=query.dtype)

        row_delta = torch.empty((batch, heads, query_len), device=query.device, dtype=torch.float32)
        row_rbar = torch.empty_like(row_delta)
        row_mean = torch.empty_like(row_delta)

        grid_q = (triton.cdiv(query_len, block_m), batch * heads)
        _triton_attention_bwd2_preprocess_kernel[grid_q](
            query,
            key,
            value,
            bias_for_kernel,
            grad_output,
            output,
            logsumexp,
            grad2_query,
            grad2_key,
            grad2_value,
            grad2_bias_for_kernel,
            row_delta,
            row_rbar,
            row_mean,
            batch,
            heads,
            query_len,
            key_len,
            head_dim,
            ctx.sm_scale,
            ctx.has_bias,
            has_grad2_bias,
            block_m,
            block_n,
            num_warps=4,
        )

        grad_query = torch.empty_like(query)
        grad_key = torch.empty_like(key)
        grad_value = torch.empty_like(value)
        grad_grad_output = torch.empty_like(grad_output)
        grad_bias = torch.zeros_like(bias_for_kernel)

        _triton_attention_bwd2_dq_dgo_kernel[grid_q](
            query,
            key,
            value,
            bias_for_kernel,
            grad_output,
            logsumexp,
            grad2_query,
            grad2_key,
            grad2_value,
            grad2_bias_for_kernel,
            row_delta,
            row_rbar,
            row_mean,
            grad_query,
            grad_grad_output,
            batch,
            heads,
            query_len,
            key_len,
            head_dim,
            ctx.sm_scale,
            ctx.has_bias,
            has_grad2_bias,
            block_m,
            block_n,
            num_warps=4,
        )

        grid_kv = (triton.cdiv(key_len, block_n), batch * heads)
        _triton_attention_bwd2_dk_dv_db_kernel[grid_kv](
            query,
            key,
            value,
            bias_for_kernel,
            grad_output,
            logsumexp,
            grad2_query,
            grad2_key,
            grad2_value,
            grad2_bias_for_kernel,
            row_delta,
            row_rbar,
            row_mean,
            grad_key,
            grad_value,
            grad_bias,
            batch,
            heads,
            query_len,
            key_len,
            head_dim,
            ctx.sm_scale,
            ctx.has_bias,
            ctx.bias_requires_grad,
            has_grad2_bias,
            block_m,
            block_n,
            num_warps=4,
        )

        return (
            grad_query,
            grad_key,
            grad_value,
            grad_bias,
            grad_grad_output,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def triton_memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_weights: torch.Tensor | None = None,
    block_m: int = 32,
    block_n: int = 64,
) -> torch.Tensor:
    """Run memory-efficient attention.

    ``attention_weights`` is an optional per-token additive bias with shape
    ``[batch, key_len]``. It is broadcast over heads and query tokens.
    """
    return TritonMemoryEfficientAttention.apply(query, key, value, attention_weights, block_m, block_n)
