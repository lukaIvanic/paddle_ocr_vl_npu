#!/usr/bin/env python3

from __future__ import annotations

import torch


def absorb_kv_b_weight(
    weight: torch.Tensor,
    *,
    local_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected = (
        local_heads * (qk_nope_head_dim + v_head_dim),
        kv_lora_rank,
    )
    if tuple(weight.shape) != expected:
        raise ValueError(
            f"Expected KV-B weight {expected}, got {tuple(weight.shape)}"
        )
    per_head = weight.view(
        local_heads,
        qk_nope_head_dim + v_head_dim,
        kv_lora_rank,
    )
    w_uk_t = per_head[:, :qk_nope_head_dim, :].contiguous()
    w_uv = per_head[:, qk_nope_head_dim:, :].transpose(1, 2).contiguous()
    return w_uk_t, w_uv


def materialize_absorbed_kv(
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    w_uk_t: torch.Tensor,
    w_uv: torch.Tensor,
    *,
    used_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    latent = latent_cache[0, :used_length]
    rope = rope_cache[0, :used_length]
    key_nope = torch.matmul(latent, w_uk_t.transpose(1, 2))
    value = torch.matmul(latent, w_uv)
    key_rope = rope.unsqueeze(0).expand(w_uk_t.shape[0], -1, -1)
    key = torch.cat((key_nope, key_rope), dim=-1)
    return key.unsqueeze(0), value.unsqueeze(0)


def manual_absorbed_attention(
    query_nope: torch.Tensor,
    query_rope: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    w_uk_t: torch.Tensor,
    w_uv: torch.Tensor,
    selected: torch.Tensor,
    position: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    local_heads = w_uk_t.shape[0]
    kv_lora_rank = w_uk_t.shape[-1]
    v_head_dim = w_uv.shape[-1]
    selected_latent = torch.index_select(latent_cache, 1, selected)
    selected_rope = torch.index_select(rope_cache, 1, selected)
    sparse_k = selected.shape[0]

    latent_query = torch.bmm(
        query_nope.reshape(local_heads, 1, query_nope.shape[-1]),
        w_uk_t,
    )
    # Keep both score contractions explicitly 2-D. GE can interpret the
    # singleton query axis of [heads, 1, dim] as the contraction dimension.
    latent_scores = torch.matmul(
        latent_query.reshape(local_heads, kv_lora_rank).float(),
        selected_latent.reshape(sparse_k, kv_lora_rank).float().transpose(0, 1),
    ).view(local_heads, 1, sparse_k)
    rope_scores = torch.matmul(
        query_rope.reshape(local_heads, query_rope.shape[-1]).float(),
        selected_rope.reshape(sparse_k, query_rope.shape[-1])
        .float()
        .transpose(0, 1),
    ).view(local_heads, 1, sparse_k)
    scores = (latent_scores + rope_scores).view(1, local_heads, 1, sparse_k)
    scores = scores * scale
    valid = selected.unsqueeze(0) <= position.unsqueeze(1)
    scores = scores.masked_fill(
        ~valid.unsqueeze(1), torch.finfo(scores.dtype).min
    )
    probabilities = torch.softmax(scores, dim=-1).to(selected_latent.dtype)
    latent_output = torch.matmul(
        probabilities.reshape(local_heads, sparse_k),
        selected_latent.reshape(sparse_k, kv_lora_rank),
    ).view(local_heads, 1, kv_lora_rank)
    value_output = torch.bmm(latent_output, w_uv)
    return value_output.transpose(0, 1).reshape(
        1, 1, local_heads * v_head_dim
    )
