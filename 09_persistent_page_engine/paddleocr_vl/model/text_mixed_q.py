"""One packed M16 pass for a B1Q8 verifier and eight B1 draft rows."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .compile_utils import (
    TORCHAIR_EXECUTION_MODE,
    cache_key_part,
    short_file_hash,
    torch_npu_version_label,
    torchair_version_label,
)
from .text_decode import (
    DECODE_LINEAR_WEIGHT_FORMAT,
    DecodeOptimizationConfig,
    _decode_add_with_optional_rms_norm,
    _decode_mlp,
    _decode_rms_norm,
    _linear_tokenwise,
    _lookup_scalar_rotary_factors,
    _prepare_multimodal_rotary_factors,
    resolve_decode_optimization,
    update_decode_kv_cache_,
)
from .text_spec_verify import _update_spec_kv_cache_

if TYPE_CHECKING:
    from .modeling import LocalPaddleOCRVLForConditionalGeneration


MIXED_M16_OPTIMIZATION = "combined_apply_mixed_m16"
VERIFIER_BATCH_SIZE = 1
VERIFIER_QUERY_LENGTH = 8
DRAFT_BATCH_SIZE = 8
DRAFT_QUERY_LENGTH = 1
PACKED_TOKEN_COUNT = 16
MIXED_LAYOUT_SPLIT_LANES_THEN_PACK_VERIFIER = "split_lanes_then_pack_verifier"
MIXED_LAYOUT_PACK_ALL_THEN_SPLIT_LANES = "pack_all_then_split_lanes"
MIXED_LAYOUT_SPLIT_LANES_THEN_TRANSPOSE_QKV = (
    "split_lanes_then_transpose_qkv_separately"
)
MIXED_LAYOUT_PACKED_BSND_PROMPTFA = "packed_bsnd_promptfa"
MIXED_LAYOUT_B2_BSND_PROMPTFA = "b2_bsnd_promptfa"
MIXED_LAYOUT_B9_BSND_PROMPTFA = "b9_bsnd_promptfa"
MIXED_LAYOUT_B16_INCREFA = "b16_replicated_increfa"
MIXED_LAYOUT_B16_INCREFA_SCATTER = "b16_replicated_increfa_scatter"
MIXED_LAYOUT_B16_INCREFA_BSH = "b16_replicated_increfa_bsh"
MIXED_REPLICATED_INCREFA_LAYOUTS = (MIXED_LAYOUT_B16_INCREFA, MIXED_LAYOUT_B16_INCREFA_SCATTER, MIXED_LAYOUT_B16_INCREFA_BSH)
MIXED_SINGLE_ATTENTION_LAYOUTS = (
    MIXED_LAYOUT_PACKED_BSND_PROMPTFA, MIXED_LAYOUT_B2_BSND_PROMPTFA,
    MIXED_LAYOUT_B9_BSND_PROMPTFA,
    MIXED_LAYOUT_B16_INCREFA,
    MIXED_LAYOUT_B16_INCREFA_SCATTER,
    MIXED_LAYOUT_B16_INCREFA_BSH,
)
MIXED_M16_LAYOUTS = (
    MIXED_LAYOUT_SPLIT_LANES_THEN_PACK_VERIFIER,
    MIXED_LAYOUT_PACK_ALL_THEN_SPLIT_LANES,
    MIXED_LAYOUT_SPLIT_LANES_THEN_TRANSPOSE_QKV,
    MIXED_LAYOUT_PACKED_BSND_PROMPTFA,
    MIXED_LAYOUT_B2_BSND_PROMPTFA,
    MIXED_LAYOUT_B9_BSND_PROMPTFA,
    MIXED_LAYOUT_B16_INCREFA,
    MIXED_LAYOUT_B16_INCREFA_SCATTER,
    MIXED_LAYOUT_B16_INCREFA_BSH,
)
DEFAULT_MIXED_M16_LAYOUT = MIXED_LAYOUT_SPLIT_LANES_THEN_PACK_VERIFIER
MIXED_PREFETCH_FULL = "full"
MIXED_PREFETCH_WEIGHTS_ONLY = "weights_only"
MIXED_PREFETCH_KV_ONLY = "kv_only"
MIXED_PREFETCH_NONE = "none"
MIXED_M16_PREFETCH_MODES = (
    MIXED_PREFETCH_FULL,
    MIXED_PREFETCH_WEIGHTS_ONLY,
    MIXED_PREFETCH_KV_ONLY,
    MIXED_PREFETCH_NONE,
)
DEFAULT_MIXED_M16_PREFETCH = MIXED_PREFETCH_FULL
MIXED_ATTENTION_VERIFIER_THEN_DRAFT = "verifier_then_draft"
MIXED_ATTENTION_DRAFT_THEN_VERIFIER = "draft_then_verifier"
MIXED_M16_ATTENTION_ORDERS = (
    MIXED_ATTENTION_VERIFIER_THEN_DRAFT,
    MIXED_ATTENTION_DRAFT_THEN_VERIFIER,
)
DEFAULT_MIXED_M16_ATTENTION_ORDER = MIXED_ATTENTION_VERIFIER_THEN_DRAFT
MIXED_ROTARY_SHARED_M16 = "shared_m16"
MIXED_ROTARY_PER_LANE = "per_lane"
MIXED_M16_ROTARY_MODES = (MIXED_ROTARY_SHARED_M16, MIXED_ROTARY_PER_LANE)
DEFAULT_MIXED_M16_ROTARY_MODE = MIXED_ROTARY_SHARED_M16


def _query_positions(cache_position: torch.Tensor, query_length: int) -> torch.Tensor:
    start = cache_position.reshape(-1).to(dtype=torch.int64)
    offsets = torch.arange(
        int(query_length),
        device=cache_position.device,
        dtype=torch.int64,
    )
    return start.view(-1, 1) + offsets.view(1, -1)


def _manual_verifier_attention(
    attention: nn.Module,
    query_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    legal_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the B1Q8 verifier result before its output projection."""
    import torch_npu

    batch_size = int(query_states.shape[0])
    query_length = int(query_states.shape[2])
    cache_length = int(key_cache.shape[2])
    head_dim = int(attention.head_dim)
    kv_heads = int(attention.num_key_value_heads)
    groups = int(attention.num_key_value_groups)
    query_heads = int(attention.num_heads)
    flattened_rows = batch_size * query_heads * query_length
    if flattened_rows != 128 or cache_length != 4096:
        raise ValueError(
            "mixed M16 verifier attention requires B1Q8 with KV4096"
        )

    grouped_query = query_states.view(
        batch_size, kv_heads, groups, query_length, head_dim
    ).reshape(batch_size * kv_heads, groups * query_length, head_dim)
    grouped_key = key_cache.reshape(
        batch_size * kv_heads, cache_length, head_dim
    )
    grouped_value = value_cache.reshape(
        batch_size * kv_heads, cache_length, head_dim
    )
    scores = torch.bmm(
        grouped_query,
        grouped_key.transpose(1, 2),
    ).view(
        batch_size * kv_heads,
        groups,
        query_length,
        cache_length,
    )
    probabilities = torch_npu.npu_scaled_masked_softmax(
        scores.reshape(1, 1, flattened_rows, cache_length),
        legal_attention_mask,
        float(attention.scaling),
        False,
    ).view_as(scores)
    return torch.bmm(
        probabilities.reshape(
            batch_size * kv_heads,
            groups * query_length,
            cache_length,
        ),
        grouped_value,
    ).view(
        batch_size,
        kv_heads,
        groups,
        query_length,
        head_dim,
    ).reshape(
        batch_size,
        query_heads,
        query_length,
        head_dim,
    )


def _draft_increfa_attention(
    attention: nn.Module,
    query_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the B8Q1 draft result before its output projection."""
    import torch_npu

    if (
        int(query_states.shape[0]) != DRAFT_BATCH_SIZE
        or int(query_states.shape[2]) != DRAFT_QUERY_LENGTH
        or int(key_cache.shape[2]) != 768
    ):
        raise ValueError("mixed M16 draft attention requires B8Q1 with KV768")
    return torch_npu.npu_incre_flash_attention(
        query_states.contiguous(),
        key_cache.contiguous(),
        value_cache.contiguous(),
        atten_mask=attention_mask.contiguous(),
        num_heads=int(attention.num_heads),
        num_key_value_heads=int(attention.num_key_value_heads),
        input_layout="BNSD",
        scale_value=float(attention.scaling),
    )


def _mixed_attention(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    packed_factors: tuple[torch.Tensor, torch.Tensor],
    verifier_key_cache: torch.Tensor,
    verifier_value_cache: torch.Tensor,
    verifier_positions: torch.Tensor,
    verifier_legal_mask: torch.Tensor,
    draft_key_cache: torch.Tensor,
    draft_value_cache: torch.Tensor,
    draft_cache_position: torch.Tensor,
    draft_attention_mask: torch.Tensor,
    optimization: DecodeOptimizationConfig,
    layout: str,
    prefetch_mode: str,
    attention_order: str,
    rotary_mode: str,
    packed_write_indices: torch.Tensor | None = None,
    packed_attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one M16 projection body and two cache-specific attention calls."""
    import torch_npu

    if tuple(hidden_states.shape[:2]) != (1, PACKED_TOKEN_COUNT):
        raise ValueError("mixed M16 hidden states must have shape [1,16,H]")
    if attention_order not in MIXED_M16_ATTENTION_ORDERS:
        raise ValueError(f"unsupported mixed attention order {attention_order!r}")
    if rotary_mode not in MIXED_M16_ROTARY_MODES:
        raise ValueError(f"unsupported mixed rotary mode {rotary_mode!r}")
    if (
        rotary_mode == MIXED_ROTARY_PER_LANE
        and layout == MIXED_LAYOUT_PACK_ALL_THEN_SPLIT_LANES
    ):
        raise ValueError("per-lane rotary does not support pack-all layout")
    query_heads = int(attention.num_heads)
    kv_heads = int(attention.num_key_value_heads)
    head_dim = int(attention.head_dim)
    packed_qkv = _linear_tokenwise(attention.decode_qkv_proj, hidden_states)
    query_flat, key_flat, value_flat = packed_qkv.split(
        (
            query_heads * head_dim,
            kv_heads * head_dim,
            kv_heads * head_dim,
        ),
        dim=-1,
    )
    query_bsnd = query_flat.view(1, PACKED_TOKEN_COUNT, query_heads, head_dim)
    key_bsnd = key_flat.view(1, PACKED_TOKEN_COUNT, kv_heads, head_dim)
    value_bsnd = value_flat.view(1, PACKED_TOKEN_COUNT, kv_heads, head_dim)
    cosine, sine = packed_factors
    if layout in MIXED_SINGLE_ATTENTION_LAYOUTS:
        # All 16 real tokens stay in projection-native BSND order. The caches
        # are persistent BSND segment arenas; no full-cache packing is hidden
        # outside (or repeated inside) the measured forward.
        query_bsnd, key_bsnd = torch_npu.npu_apply_rotary_pos_emb(
            query_bsnd, key_bsnd, cosine, sine,
            layout="BSND", rotary_mode="half",
        )
        if layout == MIXED_LAYOUT_B16_INCREFA_BSH:
            # Select directly from projection order into [B,Q,KV-head*D].
            # Repeating the draft token into seven future slots avoids padding
            # assembly; Q1 masks hide these slots and later steps overwrite them.
            key_updates = key_bsnd.reshape(16, kv_heads * head_dim).index_select(
                0, packed_write_indices
            ).reshape(16, 8, kv_heads * head_dim)
            value_updates = value_bsnd.reshape(16, kv_heads * head_dim).index_select(
                0, packed_write_indices
            ).reshape(16, 8, kv_heads * head_dim)
            starts = torch.cat((verifier_positions[:, 0].expand(8), draft_cache_position.reshape(-1))).contiguous()
            torch_npu.scatter_update_(verifier_key_cache, starts, key_updates, 1)
            torch_npu.scatter_update_(verifier_value_cache, starts, value_updates, 1)
        elif layout == MIXED_LAYOUT_B16_INCREFA_SCATTER:
            # One uniform Q8 write block per physical row permits dedicated
            # Scatter KV writes. Draft rows have one real update and seven
            # zero future updates, always hidden by their Q1 causal mask.
            key_updates = torch.cat((
                key_bsnd[:, :8].transpose(1, 2).expand(8, kv_heads, 8, head_dim),
                torch.cat((key_bsnd[:, 8:].reshape(8, kv_heads, 1, head_dim),
                           key_bsnd.new_zeros((8, kv_heads, 7, head_dim))), dim=2),
            ), dim=0).contiguous()
            value_updates = torch.cat((
                value_bsnd[:, :8].transpose(1, 2).expand(8, kv_heads, 8, head_dim),
                torch.cat((value_bsnd[:, 8:].reshape(8, kv_heads, 1, head_dim),
                           value_bsnd.new_zeros((8, kv_heads, 7, head_dim))), dim=2),
            ), dim=0).contiguous()
            starts = torch.cat((verifier_positions[:, 0].expand(8), draft_cache_position.reshape(-1))).contiguous()
            torch_npu.scatter_update_(verifier_key_cache, starts, key_updates, 2)
            torch_npu.scatter_update_(verifier_value_cache, starts, value_updates, 2)
        elif layout == MIXED_LAYOUT_B16_INCREFA:
            # Eight verifier queries share one logical history. Replicate only
            # the new Q8 block into eight persistent physical histories. Each
            # row's mask imposes its own causal endpoint.
            key_updates = torch.cat((
                key_bsnd[:, :8].expand(8, 8, kv_heads, head_dim).reshape(-1, head_dim),
                key_bsnd[:, 8:].reshape(-1, head_dim),
            ))
            value_updates = torch.cat((
                value_bsnd[:, :8].expand(8, 8, kv_heads, head_dim).reshape(-1, head_dim),
                value_bsnd[:, 8:].reshape(-1, head_dim),
            ))
        else:
            key_updates = key_bsnd.reshape(PACKED_TOKEN_COUNT, kv_heads, head_dim).contiguous()
            value_updates = value_bsnd.reshape(PACKED_TOKEN_COUNT, kv_heads, head_dim).contiguous()
        if layout not in (MIXED_LAYOUT_B16_INCREFA_SCATTER, MIXED_LAYOUT_B16_INCREFA_BSH):
            torch_npu.npu_scatter_nd_update_(verifier_key_cache, packed_write_indices, key_updates)
            torch_npu.npu_scatter_nd_update_(verifier_value_cache, packed_write_indices, value_updates)
        if (
            prefetch_mode in (MIXED_PREFETCH_FULL, MIXED_PREFETCH_KV_ONLY)
            and optimization.post_scatter_kv_prefetch
        ):
            torch_npu.npu_prefetch(
                verifier_key_cache, key_bsnd,
                verifier_key_cache.numel() * verifier_key_cache.element_size(),
            )
            torch_npu.npu_prefetch(
                verifier_value_cache, value_bsnd,
                verifier_value_cache.numel() * verifier_value_cache.element_size(),
            )
        if layout in MIXED_REPLICATED_INCREFA_LAYOUTS:
            use_bsh = layout == MIXED_LAYOUT_B16_INCREFA_BSH
            output = torch_npu.npu_incre_flash_attention(
                query_bsnd.reshape((16, 1, query_heads * head_dim) if use_bsh else (16, query_heads, 1, head_dim)).contiguous(),
                verifier_key_cache, verifier_value_cache,
                atten_mask=packed_attention_mask,
                num_heads=query_heads, num_key_value_heads=kv_heads,
                input_layout="BSH" if use_bsh else "BNSD", scale_value=float(attention.scaling),
            )
            return _linear_tokenwise(attention.o_proj, output.reshape(1, 16, query_heads * head_dim))
        if layout == MIXED_LAYOUT_B9_BSND_PROMPTFA:
            draft_queries = query_bsnd[:, 8:].reshape(8, 1, query_heads, head_dim)
            attention_query = torch.cat((
                query_bsnd[:, :8],
                torch.cat((draft_queries, draft_queries.new_zeros((8, 7, query_heads, head_dim))), dim=1),
            ), dim=0)
        else:
            attention_query = query_bsnd.reshape(
                2 if layout == MIXED_LAYOUT_B2_BSND_PROMPTFA else 1,
                8 if layout == MIXED_LAYOUT_B2_BSND_PROMPTFA else 16,
                query_heads, head_dim,
            )
        packed_output = torch_npu.npu_prompt_flash_attention(
            attention_query.contiguous(), verifier_key_cache, verifier_value_cache,
            atten_mask=packed_attention_mask,
            num_heads=query_heads, num_key_value_heads=kv_heads,
            input_layout="BSND", scale_value=float(attention.scaling),
            pre_tokens=2147483647, next_tokens=2147483647, sparse_mode=0,
        )
        if layout == MIXED_LAYOUT_B9_BSND_PROMPTFA:
            packed_output = torch.cat((
                packed_output[:1], packed_output[1:, :1].reshape(1, 8, query_heads, head_dim),
            ), dim=1)
        return _linear_tokenwise(
            attention.o_proj,
            packed_output.reshape(1, PACKED_TOKEN_COUNT, query_heads * head_dim),
        )
    lanes_are_split = rotary_mode == MIXED_ROTARY_PER_LANE
    if rotary_mode == MIXED_ROTARY_SHARED_M16:
        query_bsnd, key_bsnd = torch_npu.npu_apply_rotary_pos_emb(
            query_bsnd,
            key_bsnd,
            cosine,
            sine,
            layout="BSND",
            rotary_mode="half",
        )
    else:
        verifier_query_bsnd, draft_query_bsnd = query_bsnd.split(
            (VERIFIER_QUERY_LENGTH, DRAFT_BATCH_SIZE), dim=1
        )
        verifier_key_bsnd, draft_key_bsnd = key_bsnd.split(
            (VERIFIER_QUERY_LENGTH, DRAFT_BATCH_SIZE), dim=1
        )
        verifier_value_bsnd, draft_value_bsnd = value_bsnd.split(
            (VERIFIER_QUERY_LENGTH, DRAFT_BATCH_SIZE), dim=1
        )
        verifier_cosine, draft_cosine = cosine.split(
            (VERIFIER_QUERY_LENGTH, DRAFT_BATCH_SIZE), dim=1
        )
        verifier_sine, draft_sine = sine.split(
            (VERIFIER_QUERY_LENGTH, DRAFT_BATCH_SIZE), dim=1
        )
        verifier_query_bsnd, verifier_key_bsnd = (
            torch_npu.npu_apply_rotary_pos_emb(
                verifier_query_bsnd,
                verifier_key_bsnd,
                verifier_cosine,
                verifier_sine,
                layout="BSND",
                rotary_mode="half",
            )
        )
        draft_query, draft_key = torch_npu.npu_apply_rotary_pos_emb(
            draft_query_bsnd.reshape(
                DRAFT_BATCH_SIZE, DRAFT_QUERY_LENGTH, query_heads, head_dim
            ),
            draft_key_bsnd.reshape(
                DRAFT_BATCH_SIZE, DRAFT_QUERY_LENGTH, kv_heads, head_dim
            ),
            draft_cosine.reshape(
                DRAFT_BATCH_SIZE, DRAFT_QUERY_LENGTH, 1, head_dim
            ),
            draft_sine.reshape(
                DRAFT_BATCH_SIZE, DRAFT_QUERY_LENGTH, 1, head_dim
            ),
            layout="BSND",
            rotary_mode="half",
        )
        draft_query_bsnd = draft_query.reshape(
            1, DRAFT_BATCH_SIZE, query_heads, head_dim
        )
        draft_key_bsnd = draft_key.reshape(
            1, DRAFT_BATCH_SIZE, kv_heads, head_dim
        )

    if layout in (
        MIXED_LAYOUT_SPLIT_LANES_THEN_PACK_VERIFIER,
        MIXED_LAYOUT_SPLIT_LANES_THEN_TRANSPOSE_QKV,
    ):
        # Keep the draft lane out of the verifier's QKV packing and transpose.
        if not lanes_are_split:
            verifier_query_bsnd, draft_query_bsnd = query_bsnd.split(
                (VERIFIER_QUERY_LENGTH, DRAFT_BATCH_SIZE), dim=1
            )
            verifier_key_bsnd, draft_key_bsnd = key_bsnd.split(
                (VERIFIER_QUERY_LENGTH, DRAFT_BATCH_SIZE), dim=1
            )
            verifier_value_bsnd, draft_value_bsnd = value_bsnd.split(
                (VERIFIER_QUERY_LENGTH, DRAFT_BATCH_SIZE), dim=1
            )
        if layout == MIXED_LAYOUT_SPLIT_LANES_THEN_PACK_VERIFIER:
            verifier_packed_bsnd = torch.cat(
                (
                    verifier_query_bsnd,
                    verifier_key_bsnd,
                    verifier_value_bsnd,
                ),
                dim=2,
            )
    elif layout == MIXED_LAYOUT_PACK_ALL_THEN_SPLIT_LANES:
        # Test whether one packed lane split is cheaper than separate Q/K/V
        # lane splits. This moves all 16 tokens through the QKV concatenation.
        packed_qkv_bsnd = torch.cat(
            (query_bsnd, key_bsnd, value_bsnd),
            dim=2,
        )
        verifier_packed_bsnd, draft_packed_bsnd = packed_qkv_bsnd.split(
            (VERIFIER_QUERY_LENGTH, DRAFT_BATCH_SIZE), dim=1
        )
        draft_query_bsnd, draft_key_bsnd, draft_value_bsnd = (
            draft_packed_bsnd.split((query_heads, kv_heads, kv_heads), dim=2)
        )
    else:
        raise ValueError(f"unsupported mixed M16 layout {layout!r}")

    draft_output: torch.Tensor | None = None
    if attention_order == MIXED_ATTENTION_DRAFT_THEN_VERIFIER:
        draft_query = draft_query_bsnd.reshape(
            DRAFT_BATCH_SIZE, query_heads, DRAFT_QUERY_LENGTH, head_dim
        )
        draft_key = draft_key_bsnd.reshape(
            DRAFT_BATCH_SIZE, kv_heads, DRAFT_QUERY_LENGTH, head_dim
        )
        draft_value = draft_value_bsnd.reshape(
            DRAFT_BATCH_SIZE, kv_heads, DRAFT_QUERY_LENGTH, head_dim
        )
        draft_key_cache, draft_value_cache = update_decode_kv_cache_(
            draft_key_cache,
            draft_value_cache,
            draft_cache_position,
            draft_key,
            draft_value,
        )
        if (
            prefetch_mode in (MIXED_PREFETCH_FULL, MIXED_PREFETCH_KV_ONLY)
            and optimization.post_scatter_kv_prefetch
        ):
            torch_npu.npu_prefetch(
                draft_key_cache,
                draft_key,
                int(draft_key_cache.numel() * draft_key_cache.element_size()),
            )
            torch_npu.npu_prefetch(
                draft_value_cache,
                draft_value,
                int(draft_value_cache.numel() * draft_value_cache.element_size()),
            )
        draft_output = _draft_increfa_attention(
            attention,
            draft_query,
            draft_key_cache,
            draft_value_cache,
            draft_attention_mask,
        )

    if layout == MIXED_LAYOUT_SPLIT_LANES_THEN_TRANSPOSE_QKV:
        verifier_query = verifier_query_bsnd.transpose(1, 2).contiguous()
        verifier_key = verifier_key_bsnd.transpose(1, 2).contiguous()
        verifier_value = verifier_value_bsnd.transpose(1, 2).contiguous()
    else:
        verifier_packed_qkv = verifier_packed_bsnd.transpose(1, 2).contiguous()
        verifier_query, verifier_key, verifier_value = verifier_packed_qkv.split(
            (query_heads, kv_heads, kv_heads), dim=1
        )
    _update_spec_kv_cache_(
        verifier_key_cache,
        verifier_value_cache,
        verifier_positions,
        verifier_key,
        verifier_value,
    )
    if (
        prefetch_mode in (MIXED_PREFETCH_FULL, MIXED_PREFETCH_KV_ONLY)
        and optimization.post_scatter_kv_prefetch
    ):
        torch_npu.npu_prefetch(
            verifier_key_cache,
            verifier_key,
            int(verifier_key_cache.numel() * verifier_key_cache.element_size()),
        )
        torch_npu.npu_prefetch(
            verifier_value_cache,
            verifier_value,
            int(verifier_value_cache.numel() * verifier_value_cache.element_size()),
        )
    verifier_output = _manual_verifier_attention(
        attention,
        verifier_query,
        verifier_key_cache,
        verifier_value_cache,
        verifier_legal_mask,
    )

    if draft_output is None:
        # Q=1 makes BSND and BNSD physically identical after the batch reshape.
        # Keep these as views instead of concatenating and transposing draft QKV.
        draft_query = draft_query_bsnd.reshape(
            DRAFT_BATCH_SIZE, query_heads, DRAFT_QUERY_LENGTH, head_dim
        )
        draft_key = draft_key_bsnd.reshape(
            DRAFT_BATCH_SIZE, kv_heads, DRAFT_QUERY_LENGTH, head_dim
        )
        draft_value = draft_value_bsnd.reshape(
            DRAFT_BATCH_SIZE, kv_heads, DRAFT_QUERY_LENGTH, head_dim
        )
        draft_key_cache, draft_value_cache = update_decode_kv_cache_(
            draft_key_cache,
            draft_value_cache,
            draft_cache_position,
            draft_key,
            draft_value,
        )
        if (
            prefetch_mode in (MIXED_PREFETCH_FULL, MIXED_PREFETCH_KV_ONLY)
            and optimization.post_scatter_kv_prefetch
        ):
            torch_npu.npu_prefetch(
                draft_key_cache,
                draft_key,
                int(draft_key_cache.numel() * draft_key_cache.element_size()),
            )
            torch_npu.npu_prefetch(
                draft_value_cache,
                draft_value,
                int(draft_value_cache.numel() * draft_value_cache.element_size()),
            )
        draft_output = _draft_increfa_attention(
            attention,
            draft_query,
            draft_key_cache,
            draft_value_cache,
            draft_attention_mask,
        )

    verifier_tokens = verifier_output.transpose(1, 2).contiguous().reshape(
        VERIFIER_QUERY_LENGTH, query_heads * head_dim
    )
    draft_tokens = draft_output.transpose(1, 2).contiguous().reshape(
        DRAFT_BATCH_SIZE, query_heads * head_dim
    )
    packed_output = torch.cat((verifier_tokens, draft_tokens), dim=0).view(
        1, PACKED_TOKEN_COUNT, query_heads * head_dim
    )
    return _linear_tokenwise(attention.o_proj, packed_output)


def run_text_mixed_m16_transformer(
    text_model: nn.Module,
    *,
    verifier_inputs_embeds: torch.Tensor,
    verifier_cache_position: torch.Tensor,
    verifier_rope_deltas: torch.Tensor,
    verifier_key_caches: tuple[torch.Tensor, ...],
    verifier_value_caches: tuple[torch.Tensor, ...],
    draft_inputs_embeds: torch.Tensor,
    draft_cache_position: torch.Tensor,
    draft_rope_deltas: torch.Tensor,
    draft_key_caches: tuple[torch.Tensor, ...],
    draft_value_caches: tuple[torch.Tensor, ...],
    optimization: str | DecodeOptimizationConfig = MIXED_M16_OPTIMIZATION,
    layout: str = DEFAULT_MIXED_M16_LAYOUT,
    prefetch_mode: str = DEFAULT_MIXED_M16_PREFETCH,
    attention_order: str = DEFAULT_MIXED_M16_ATTENTION_ORDER,
    rotary_mode: str = DEFAULT_MIXED_M16_ROTARY_MODE,
) -> torch.Tensor:
    optimization = resolve_decode_optimization(optimization)
    if optimization.name != MIXED_M16_OPTIMIZATION:
        raise ValueError("mixed M16 requires its locked optimization preset")
    if layout not in MIXED_M16_LAYOUTS:
        raise ValueError(f"unsupported mixed M16 layout {layout!r}")
    if prefetch_mode not in MIXED_M16_PREFETCH_MODES:
        raise ValueError(f"unsupported mixed M16 prefetch mode {prefetch_mode!r}")
    if attention_order not in MIXED_M16_ATTENTION_ORDERS:
        raise ValueError(f"unsupported mixed attention order {attention_order!r}")
    if rotary_mode not in MIXED_M16_ROTARY_MODES:
        raise ValueError(f"unsupported mixed rotary mode {rotary_mode!r}")

    verifier_positions = _query_positions(
        verifier_cache_position, VERIFIER_QUERY_LENGTH
    )
    verifier_kv_positions = torch.arange(
        4096,
        device=verifier_inputs_embeds.device,
        dtype=torch.int64,
    )
    verifier_mask = verifier_kv_positions.view(1, 1, 1, 4096) > (
        verifier_positions.view(1, 1, VERIFIER_QUERY_LENGTH, 1)
    )
    attention = text_model.layers[0].self_attn
    verifier_legal_mask = (
        verifier_mask.view(1, 1, 1, VERIFIER_QUERY_LENGTH, 4096)
        .expand(
            1,
            int(attention.num_key_value_heads),
            int(attention.num_key_value_groups),
            VERIFIER_QUERY_LENGTH,
            4096,
        )
        .reshape(1, 1, 128, 4096)
        .contiguous()
    )
    draft_positions = draft_cache_position.reshape(-1).to(dtype=torch.int64)
    draft_kv_positions = torch.arange(
        768,
        device=draft_inputs_embeds.device,
        dtype=torch.int64,
    )
    draft_attention_mask = (
        draft_kv_positions.view(1, 768) > draft_positions.view(-1, 1)
    ).view(DRAFT_BATCH_SIZE, 1, 1, 768)
    packed_write_indices = None
    packed_attention_mask = None
    if layout in MIXED_SINGLE_ATTENTION_LAYOUTS:
        attention_batch = 2 if layout == MIXED_LAYOUT_B2_BSND_PROMPTFA else 1
        attention_kv = 6144 if attention_batch == 2 else 10240
        draft_offsets = (0 if attention_batch == 2 else 4096) + torch.arange(
            DRAFT_BATCH_SIZE, device=draft_inputs_embeds.device, dtype=torch.int64
        ) * 768
        limits = torch.cat((verifier_positions.reshape(-1), draft_offsets + draft_positions))
        starts = torch.cat((torch.zeros_like(verifier_positions.reshape(-1)), draft_offsets))
        slots = torch.arange(attention_kv, device=draft_inputs_embeds.device, dtype=torch.int64)
        packed_attention_mask = (
            (slots.view(1, -1) < starts.view(-1, 1))
            | (slots.view(1, -1) > limits.view(-1, 1))
        ).view(attention_batch, 1, PACKED_TOKEN_COUNT // attention_batch, attention_kv).contiguous()
        batch_indices = torch.cat((
            torch.zeros_like(verifier_positions.reshape(-1)),
            torch.full_like(draft_positions, attention_batch - 1),
        ))
        packed_write_indices = torch.stack((batch_indices, limits), dim=1)
        if layout == MIXED_LAYOUT_B9_BSND_PROMPTFA:
            batch_indices = torch.cat((
                torch.zeros_like(verifier_positions.reshape(-1)),
                torch.arange(1, 9, device=draft_positions.device, dtype=torch.int64),
            ))
            write_positions = torch.cat((verifier_positions.reshape(-1), draft_positions))
            packed_write_indices = torch.stack((batch_indices, write_positions), dim=1)
            query_limits = torch.cat((verifier_positions, draft_positions.view(8, 1).expand(8, 8)))
            # Dummy queries use the same legal prefix as their real draft row.
            # They never enter RoPE, the MLP, the output projection or KV writes.
            packed_attention_mask = (
                verifier_kv_positions.view(1, 1, 1, 4096) > query_limits.view(9, 1, 8, 1)
            ).contiguous()
        if layout in MIXED_REPLICATED_INCREFA_LAYOUTS:
            rows = torch.arange(8, device=draft_positions.device, dtype=torch.int64)
            heads = torch.arange(2, device=draft_positions.device, dtype=torch.int64)
            verifier_indices = torch.stack((
                rows.view(8, 1, 1).expand(8, 8, 2),
                heads.view(1, 1, 2).expand(8, 8, 2),
                verifier_positions.view(1, 8, 1).expand(8, 8, 2),
            ), dim=-1).reshape(-1, 3)
            draft_indices = torch.stack((
                (rows + 8).view(8, 1).expand(8, 2),
                heads.view(1, 2).expand(8, 2),
                draft_positions.view(8, 1).expand(8, 2),
            ), dim=-1).reshape(-1, 3)
            packed_write_indices = torch.cat((verifier_indices, draft_indices))
            query_limits = torch.cat((verifier_positions.reshape(-1), draft_positions))
            packed_attention_mask = (
                verifier_kv_positions.view(1, 1, 1, 4096) > query_limits.view(16, 1, 1, 1)
            ).contiguous()
        if layout == MIXED_LAYOUT_B16_INCREFA_BSH:
            packed_write_indices = torch.cat((
                torch.arange(8, device=draft_positions.device, dtype=torch.int64).repeat(8),
                torch.arange(8, 16, device=draft_positions.device, dtype=torch.int64).repeat_interleave(8),
            ))

    verifier_decode_positions = verifier_positions + verifier_rope_deltas.to(
        device=verifier_inputs_embeds.device,
        dtype=torch.int64,
    )
    verifier_position_ids = verifier_decode_positions.unsqueeze(0).expand(
        3, -1, -1
    )
    verifier_position_embeddings = text_model.rotary_emb(
        verifier_inputs_embeds,
        verifier_position_ids,
    )
    verifier_factors = _prepare_multimodal_rotary_factors(
        verifier_position_embeddings,
        attention.mrope_section,
    )
    verifier_factors = (
        verifier_factors[0].transpose(1, 2).contiguous(),
        verifier_factors[1].transpose(1, 2).contiguous(),
    )
    draft_decode_positions = draft_positions.view(-1, 1) + draft_rope_deltas.to(
        device=draft_inputs_embeds.device,
        dtype=torch.int64,
    )
    draft_factors = _lookup_scalar_rotary_factors(
        text_model.rotary_emb,
        draft_decode_positions,
    )
    packed_factors = (
        torch.cat(
            (
                verifier_factors[0],
                draft_factors[0].reshape(1, DRAFT_BATCH_SIZE, 1, -1),
            ),
            dim=1,
        ),
        torch.cat(
            (
                verifier_factors[1],
                draft_factors[1].reshape(1, DRAFT_BATCH_SIZE, 1, -1),
            ),
            dim=1,
        ),
    )

    hidden_states = torch.cat(
        (
            verifier_inputs_embeds,
            draft_inputs_embeds.reshape(1, DRAFT_BATCH_SIZE, -1),
        ),
        dim=1,
    )
    residual: torch.Tensor | None = None
    for layer_index, layer in enumerate(text_model.layers):
        if (
            prefetch_mode in (MIXED_PREFETCH_FULL, MIXED_PREFETCH_WEIGHTS_ONLY)
            and optimization.complete_layer_prefetch_ahead
        ):
            import torch_npu

            for weight in layer._decode_prefetch_future_layers:
                torch_npu.npu_prefetch(
                    weight,
                    hidden_states,
                    int(weight.numel() * weight.element_size()),
                )
        if residual is None:
            attention_input = _decode_rms_norm(
                layer.input_layernorm,
                hidden_states,
                optimization,
            )
            residual = hidden_states
        else:
            attention_input, residual = _decode_add_with_optional_rms_norm(
                hidden_states,
                residual,
                layer.input_layernorm,
                optimization,
            )
        attention_output = _mixed_attention(
            layer.self_attn,
            attention_input,
            packed_factors,
            verifier_key_caches[layer_index],
            verifier_value_caches[layer_index],
            verifier_positions,
            verifier_legal_mask,
            draft_key_caches[layer_index],
            draft_value_caches[layer_index],
            draft_cache_position,
            draft_attention_mask,
            optimization,
            layout,
            prefetch_mode,
            attention_order,
            rotary_mode,
            packed_write_indices,
            packed_attention_mask,
        )
        mlp_input, residual = _decode_add_with_optional_rms_norm(
            attention_output,
            residual,
            layer.post_attention_layernorm,
            optimization,
        )
        hidden_states = _decode_mlp(layer.mlp, mlp_input, optimization)
    hidden_states, _residual = _decode_add_with_optional_rms_norm(
        hidden_states,
        residual,
        text_model.norm,
        optimization,
    )
    return hidden_states


class TextMixedM16Stage(nn.Module):
    """Static mixed graph for one Q8 verifier and eight Q1 draft rows."""

    def __init__(
        self,
        model: "LocalPaddleOCRVLForConditionalGeneration",
        *,
        optimization: str | DecodeOptimizationConfig = MIXED_M16_OPTIMIZATION,
        layout: str = DEFAULT_MIXED_M16_LAYOUT,
        prefetch_mode: str = DEFAULT_MIXED_M16_PREFETCH,
        attention_order: str = DEFAULT_MIXED_M16_ATTENTION_ORDER,
        rotary_mode: str = DEFAULT_MIXED_M16_ROTARY_MODE,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_layers = int(model.config.text_config.num_hidden_layers)
        self.optimization = resolve_decode_optimization(optimization)
        if self.optimization.name != MIXED_M16_OPTIMIZATION:
            raise ValueError("mixed M16 stage requires its locked preset")
        if layout not in MIXED_M16_LAYOUTS:
            raise ValueError(f"unsupported mixed M16 layout {layout!r}")
        if prefetch_mode not in MIXED_M16_PREFETCH_MODES:
            raise ValueError(
                f"unsupported mixed M16 prefetch mode {prefetch_mode!r}"
            )
        self.layout = layout
        self.prefetch_mode = prefetch_mode
        if attention_order not in MIXED_M16_ATTENTION_ORDERS:
            raise ValueError(f"unsupported mixed attention order {attention_order!r}")
        self.attention_order = attention_order
        if rotary_mode not in MIXED_M16_ROTARY_MODES:
            raise ValueError(f"unsupported mixed rotary mode {rotary_mode!r}")
        self.rotary_mode = rotary_mode
        if layout in MIXED_SINGLE_ATTENTION_LAYOUTS and rotary_mode != MIXED_ROTARY_SHARED_M16:
            raise ValueError("single-attention layouts use shared M16 rotary")
        if not hasattr(model, "decode_token_id_map"):
            raise ValueError("mixed M16 stage requires the compact output vocabulary")

    def forward(
        self,
        verifier_input_ids: torch.Tensor,
        verifier_cache_position: torch.Tensor,
        verifier_rope_deltas: torch.Tensor,
        draft_input_ids: torch.Tensor,
        draft_cache_position: torch.Tensor,
        draft_rope_deltas: torch.Tensor,
        *flat_cache_tensors: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(verifier_input_ids.shape) != (1, VERIFIER_QUERY_LENGTH):
            raise ValueError("verifier input must have shape [1,8]")
        if tuple(draft_input_ids.shape) != (DRAFT_BATCH_SIZE, 1):
            raise ValueError("draft input must have shape [8,1]")
        packed_cache = self.layout in MIXED_SINGLE_ATTENTION_LAYOUTS
        expected_cache_tensors = (2 if packed_cache else 4) * self.num_layers
        if len(flat_cache_tensors) != expected_cache_tensors:
            raise ValueError(
                f"expected {expected_cache_tensors} flat cache tensors, "
                f"got {len(flat_cache_tensors)}"
            )
        verifier_key_caches = flat_cache_tensors[: self.num_layers]
        verifier_value_caches = flat_cache_tensors[
            self.num_layers : 2 * self.num_layers
        ]
        draft_key_caches = flat_cache_tensors[
            2 * self.num_layers : 3 * self.num_layers
        ]
        draft_value_caches = flat_cache_tensors[3 * self.num_layers :]
        if packed_cache:
            # The common layer body takes two cache groups. The packed branch
            # uses only the first; these aliases do not copy or write twice.
            draft_key_caches = verifier_key_caches
            draft_value_caches = verifier_value_caches
        verifier_inputs_embeds = self.model.model.embed_tokens(verifier_input_ids)
        draft_inputs_embeds = self.model.model.embed_tokens(draft_input_ids)
        hidden_states = run_text_mixed_m16_transformer(
            self.model.model,
            verifier_inputs_embeds=verifier_inputs_embeds,
            verifier_cache_position=verifier_cache_position,
            verifier_rope_deltas=verifier_rope_deltas,
            verifier_key_caches=verifier_key_caches,
            verifier_value_caches=verifier_value_caches,
            draft_inputs_embeds=draft_inputs_embeds,
            draft_cache_position=draft_cache_position,
            draft_rope_deltas=draft_rope_deltas,
            draft_key_caches=draft_key_caches,
            draft_value_caches=draft_value_caches,
            optimization=self.optimization,
            layout=self.layout,
            prefetch_mode=self.prefetch_mode,
            attention_order=self.attention_order,
            rotary_mode=self.rotary_mode,
        )
        output_head = self.model.decode_lm_head
        logits = _linear_tokenwise(output_head, hidden_states)
        compact_ids = torch.argmax(logits.float(), dim=-1)
        return self.model.decode_token_id_map.index_select(
            0,
            compact_ids.reshape(-1),
        ).view(1, PACKED_TOKEN_COUNT)


def mixed_m16_source_hash() -> str:
    here = Path(__file__).resolve().parent
    digest = hashlib.sha1()
    for name in (
        "text_prefill.py",
        "text_decode.py",
        "text_spec_verify.py",
        "text_mixed_q.py",
    ):
        path = here / name
        digest.update(name.encode("utf-8"))
        digest.update(short_file_hash(path).encode("utf-8"))
    return digest.hexdigest()[:12]


def torchair_cache_dir_for_mixed_m16(
    cache_root: Path,
    *,
    dtype: torch.dtype,
    device: torch.device,
    model_dir: Path,
    linear_weight_format: str = DECODE_LINEAR_WEIGHT_FORMAT,
    layout: str = DEFAULT_MIXED_M16_LAYOUT,
    prefetch_mode: str = DEFAULT_MIXED_M16_PREFETCH,
    attention_order: str = DEFAULT_MIXED_M16_ATTENTION_ORDER,
    rotary_mode: str = DEFAULT_MIXED_M16_ROTARY_MODE,
) -> Path:
    if layout not in MIXED_M16_LAYOUTS:
        raise ValueError(f"unsupported mixed M16 layout {layout!r}")
    if prefetch_mode not in MIXED_M16_PREFETCH_MODES:
        raise ValueError(f"unsupported mixed M16 prefetch mode {prefetch_mode!r}")
    if attention_order not in MIXED_M16_ATTENTION_ORDERS:
        raise ValueError(f"unsupported mixed attention order {attention_order!r}")
    if rotary_mode not in MIXED_M16_ROTARY_MODES:
        raise ValueError(f"unsupported mixed rotary mode {rotary_mode!r}")
    shape_key = "_".join(
        (
            "text_mixed_m16",
            linear_weight_format,
            "manual_q8_increfa_q1",
            f"opt{MIXED_M16_OPTIMIZATION}",
            f"mode{cache_key_part(TORCHAIR_EXECUTION_MODE)}",
            f"dtype{cache_key_part(dtype)}",
            "verifier_b1q8_kv4096",
            "draft_b8q1_kv768",
            f"layout{cache_key_part(layout)}",
            f"prefetch{cache_key_part(prefetch_mode)}",
            f"order{cache_key_part(attention_order)}",
            f"rotary{cache_key_part(rotary_mode)}",
            f"model{short_file_hash(model_dir / 'config.json')}",
            f"torch{cache_key_part(torch.__version__)}",
            f"torchnpu{torch_npu_version_label(device)}",
            f"torchair{torchair_version_label(device)}",
            f"src{mixed_m16_source_hash()}",
        )
    )
    if len(shape_key.encode("utf-8")) > 240:
        shape_key = f"text_mixed_m16_key{hashlib.sha1(shape_key.encode()).hexdigest()[:20]}"
    return cache_root.expanduser().resolve() / shape_key


def mixed_m16_contract(
    layout: str = DEFAULT_MIXED_M16_LAYOUT,
    prefetch_mode: str = DEFAULT_MIXED_M16_PREFETCH,
    attention_order: str = DEFAULT_MIXED_M16_ATTENTION_ORDER,
    rotary_mode: str = DEFAULT_MIXED_M16_ROTARY_MODE,
) -> dict[str, Any]:
    if layout not in MIXED_M16_LAYOUTS:
        raise ValueError(f"unsupported mixed M16 layout {layout!r}")
    if prefetch_mode not in MIXED_M16_PREFETCH_MODES:
        raise ValueError(f"unsupported mixed M16 prefetch mode {prefetch_mode!r}")
    if attention_order not in MIXED_M16_ATTENTION_ORDERS:
        raise ValueError(f"unsupported mixed attention order {attention_order!r}")
    if rotary_mode not in MIXED_M16_ROTARY_MODES:
        raise ValueError(f"unsupported mixed rotary mode {rotary_mode!r}")
    return {
        "attention_boundary": (
            ("single_increfa_persistent_replicated_b16kv4096" if layout in MIXED_REPLICATED_INCREFA_LAYOUTS
             else "single_promptfa_persistent_bsnd_b9kv4096" if layout == MIXED_LAYOUT_B9_BSND_PROMPTFA
             else "single_promptfa_persistent_bsnd_b2kv6144" if layout == MIXED_LAYOUT_B2_BSND_PROMPTFA
             else "single_promptfa_persistent_bsnd_b1kv10240")
            if layout in MIXED_SINGLE_ATTENTION_LAYOUTS
            else "manual_q8_plus_increfa_q1"
        ),
        "packed_token_count": PACKED_TOKEN_COUNT,
        "verifier": {
            "batch_size": VERIFIER_BATCH_SIZE,
            "query_length": VERIFIER_QUERY_LENGTH,
            "cache_length": 4096,
            "attention": ("increfa" if layout in MIXED_REPLICATED_INCREFA_LAYOUTS else "promptfa" if layout in MIXED_SINGLE_ATTENTION_LAYOUTS
                          else "manual_grouped_legal_scaled_masked_softmax_fp16"),
        },
        "draft": {
            "batch_size": DRAFT_BATCH_SIZE,
            "query_length": DRAFT_QUERY_LENGTH,
            "cache_length": 768,
            "attention": "promptfa" if layout in MIXED_SINGLE_ATTENTION_LAYOUTS and layout not in MIXED_REPLICATED_INCREFA_LAYOUTS else "increfa",
        },
        "optimization": MIXED_M16_OPTIMIZATION,
        "layout": layout,
        "prefetch_mode": prefetch_mode,
        "attention_order": attention_order,
        "rotary_mode": rotary_mode,
    }
