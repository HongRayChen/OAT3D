# Copyright 2025 Xunhao Lai & Jianqiao Lu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------
# Copyright 2025 Shuang Wu
# adapted from https://github.com/XunhaoLai/native-sparse-attention-triton/blob/main/native_sparse_attention/ops/triton/topk_sparse_attention.py

# OAT3D adaptation: inference-only selection attention.

import math
from typing import Any, Optional

import torch
import triton
import triton.language as tl


@triton.jit
def forward_kernel(
    q_ptr,  # Q: n x h x d
    k_ptr,  # K: n x kh x d
    v_ptr,  # V: n x kh x d
    t_ptr,  # topk_idx: kh x n x k
    o_ptr,  # O: n x h x d
    lse_ptr,  # LSE: h x n
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    cu_seqblocks,
    cu_block_include_tokens,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    TOPK,
    # q loop num
    num_q_loop,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_lh,
    stride_ln,
    # META parameters
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_b = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_h = pid_kh * NUM_SHARE_Q_HEADS
    pid_q = tl.program_id(2)
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    block_start = tl.load(cu_seqblocks + pid_b)
    block_len = tl.load(cu_seqblocks + pid_b + 1) - block_start
    if pid_q * num_q_loop >= q_len:
        return
    num_q_loop_ = min(num_q_loop, q_len - pid_q * num_q_loop)
    for j in range(num_q_loop_):
        pid_q_j = pid_q * num_q_loop + j
        # init topk idx pointer
        off_t = tl.arange(0, BLOCK_SIZE_T)
        t_ptr_j = t_ptr + (q_start + pid_q_j) * stride_tn + pid_kh * stride_th
        topk_idx = tl.load(t_ptr_j + off_t * stride_tk, mask=off_t < TOPK, other=-1)
        real_topk = tl.sum(
            tl.where((topk_idx >= 0) & (topk_idx < block_len), 1, 0),
            axis=0,
        )
        # init qkv pointer
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + (q_start + pid_q_j) * stride_qn + pid_h * stride_qh,
            shape=(NUM_SHARE_Q_HEADS, HEAD_DIM),
            strides=(stride_qh, stride_qd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        k_ptrs = tl.make_block_ptr(
            base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
            shape=(HEAD_DIM, k_len),
            strides=(stride_kd, stride_kn),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
            order=(0, 1),
        )
        v_ptrs = tl.make_block_ptr(
            base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
            shape=(k_len, HEAD_DIM),
            strides=(stride_vn, stride_vd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
            order=(1, 0),
        )
        # load q
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
        # init statistics
        off_h = tl.arange(0, BLOCK_SIZE_H)
        off_k = tl.arange(0, BLOCK_SIZE_K)
        m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        acc_o = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0, dtype=tl.float32)
        # sparse attention
        for i in range(real_topk):
            # get current block start index
            cur_block_idx = tl.load(t_ptr_j).to(tl.int32)
            cur_token_start = tl.load(cu_block_include_tokens + block_start + cur_block_idx).to(tl.int32)
            cur_block_size = tl.load(cu_block_include_tokens + block_start + cur_block_idx + 1).to(tl.int32) - cur_token_start
            c = cur_token_start - k_start
            t_ptr_j = t_ptr_j + stride_tk
            for b_j in range(0, cur_block_size, BLOCK_SIZE_K):
                # load k
                k = tl.load(
                    tl.advance(k_ptrs, (0, c + b_j)),
                    boundary_check=(1, 0), padding_option="zero",
                )
                # compute qk
                qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
                qk += tl.where(((c + b_j + off_k < k_len) & (b_j + off_k < cur_block_size))[None, :], 0, float("-inf"))
                # [BLOCK_SIZE_H, HEAD_DIM] @ [HEAD_DIM, BLOCK_SIZE_K] -> [BLOCK_SIZE_H, BLOCK_SIZE_K]
                qk += tl.dot(q, k) * qk_scale
                # compute m_ij and l_ij
                m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
                p = tl.exp2(qk - m_ij[:, None])
                l_ij = tl.sum(p, axis=1)
                # scale acc_o
                acc_o_scale = tl.exp2(m_i - m_ij)
                acc_o = acc_o * acc_o_scale[:, None]
                # load v and update acc_o
                v = tl.load(
                    tl.advance(v_ptrs, (c + b_j, 0)),
                    boundary_check=(0, 1), padding_option="zero"
                )
                p = p.to(v.dtype)
                acc_o += tl.dot(p, v)
                # update statistics
                m_i = m_ij
                lse_i = m_ij + tl.math.log2(tl.exp2(lse_i - m_ij) + l_ij)
        # final scale
        acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
        # save output
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + (q_start + pid_q_j) * stride_on + pid_h * stride_oh,
            shape=(NUM_SHARE_Q_HEADS, HEAD_DIM),
            strides=(stride_oh, stride_od),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
        # save lse
        lse_ptrs = (
            lse_ptr + (q_start + pid_q_j) * stride_ln + (pid_h + off_h) * stride_lh
        )
        tl.store(lse_ptrs, lse_i, mask=off_h < NUM_SHARE_Q_HEADS)


def _spatial_selection_attention_fwd(
    q: torch.Tensor,  # [total_len, num_q_heads, head_dim]
    k: torch.Tensor,  # [total_len, num_k_heads, head_dim]
    v: torch.Tensor,  # [total_len, num_k_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_len, topk]
    cu_seqblocks: torch.Tensor,
    cu_block_include_tokens: torch.Tensor,
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
):
    # dtype check
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    # shape
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    v_len, num_v_heads, head_dim = v.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    assert q_len == k_len and k_len == v_len
    topk = topk_idx.shape[-1]
    assert topk_idx.shape[0] == num_k_heads
    assert topk_idx.shape[1] == q_len
    # gqa
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    # output tensor
    o = torch.zeros_like(q)
    lse = torch.zeros(num_q_heads, q_len, dtype=torch.float32, device=q.device)
    # launch kernel
    num_q_loop = (
        cu_seqlens_q[-1].item() // 32768 + 1
    )  # calculate multiple querys in one kernel if seqlence length is too long
    grid = (batch_size, num_k_heads, triton.cdiv(max_seqlen_q, num_q_loop))
    num_warps = 4 if head_dim <= 64 else 8
    num_stages = 3
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    BLOCK_SIZE_H = max(16, triton.next_power_of_2(num_share_q_heads))
    BLOCK_SIZE_T = triton.next_power_of_2(topk)
    forward_kernel[grid](
        q,
        k,
        v,
        topk_idx,
        o,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqblocks,
        cu_block_include_tokens,
        num_k_heads,
        num_share_q_heads,
        head_dim,
        topk,
        num_q_loop,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        lse.stride(0),
        lse.stride(1),
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_T=BLOCK_SIZE_T,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o, lse


def spatial_selection_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_topk: torch.Tensor,
    cu_seqblocks: torch.Tensor,
    cu_block_include_tokens: torch.Tensor,
    block_size: int,
    cu_seqlens: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Spatial selection attention implemented in triton.

    Args:
        q (torch.Tensor): shape [total_len, num_q_heads, head_dim]
        k (torch.Tensor): shape [total_len, num_kv_heads, head_dim]
        v (torch.Tensor): shape [total_len, num_kv_heads, head_dim]
        block_topk (torch.Tensor): topk block idx for each query, shape [num_kv_heads, total_len, topk]. -1 means padding.
        block_size (int): key value block size.
        cu_seqlens (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens in flash_attn_func_varlen.
        cu_block_include_tokens (torch.Tensor) shape [total_block_len]: number of tokens within each block
        softmax_scale (Optional[float], optional): Defaults to None, means 1/sqrt(head_dim).

    Returns:
        torch.Tensor: attention output, shape [total_len, num_q_heads, head_dim]
    """
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(q.shape[-1])
    output, _ = _spatial_selection_attention_fwd(
        q,
        k,
        v,
        block_topk,
        cu_seqblocks,
        cu_block_include_tokens,
        block_size,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        softmax_scale,
    )
    return output
