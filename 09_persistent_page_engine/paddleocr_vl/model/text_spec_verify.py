"""Compiled multi-token speculative verification for text decode.

The stage consumes the current token followed by ``draft_length`` proposed
tokens.  Its ``draft_length + 1`` logits predict the draft tokens and one
additional target token.  The graph also writes every query token into the
existing persistent KV arena; the caller decides how much of that tentative
tail to commit by advancing the logical cache position.
"""

from __future__ import annotations

import hashlib
import os
import time
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

import torch
from torch import nn

from .compile_utils import (
    TORCHAIR_EXECUTION_MODE,
    cache_key_part,
    import_torchair,
    short_file_hash,
    torch_npu_version_label,
    torchair_version_label,
)
from .text_decode import (
    DECODE_LINEAR_WEIGHT_FORMAT,
    DecodeOptimizationConfig,
    _apply_decode_rotary,
    _decode_add_with_optional_rms_norm,
    _decode_mlp,
    _decode_rms_norm,
    _linear_tokenwise,
    _prepare_multimodal_rotary_factors,
    _project_decode_qkv,
    prepare_decode_optimization_modules,
    prepare_decode_weight_prefetch,
    resolve_decode_optimization,
)
from .token_selection import (
    TOKEN_SELECTION_GREEDY,
    TOKEN_SELECTION_SUPPRESS_MATH_OPEN_AND_SLASH_GREEDY,
    TOKEN_SELECTION_SUPPRESS_MATH_OPEN_GREEDY,
    select_token_ids,
)
from utils.timing import synchronize

if TYPE_CHECKING:
    from .modeling import LocalPaddleOCRVLForConditionalGeneration


FULL_ATTENTION_TOKENS = (1 << 31) - 1
SPEC_VERIFY_ATTENTION = os.environ.get(
    "SPEC_VERIFY_ATTENTION", "promptfa_gqa"
)
_COMBINED_QKV_LAYOUT_ATTENTION = (
    "manual_grouped_legal_scaled_masked_softmax_fp16_combined_qkv_rotary_mul"
)
_COMBINED_QKV_POST_ROPE_ATTENTION = (
    "manual_grouped_legal_scaled_masked_softmax_fp16_combined_qkv_post_rope"
)
if SPEC_VERIFY_ATTENTION not in (
    "promptfa_gqa",
    "manual_grouped_fp32",
    "manual_grouped_scaled_masked_softmax",
    "manual_grouped_legal_scaled_masked_softmax_fp16",
    "manual_grouped_legal_scaled_masked_softmax_fp32",
    _COMBINED_QKV_LAYOUT_ATTENTION,
    _COMBINED_QKV_POST_ROPE_ATTENTION,
):
    raise ValueError(f"unsupported SPEC_VERIFY_ATTENTION={SPEC_VERIFY_ATTENTION!r}")
SPEC_VERIFY_CACHE_UPDATE = "npu_scatter"
_SCALED_MASKED_SOFTMAX_CONVERTER_REGISTERED = False


def _register_scaled_masked_softmax_torchair_converter() -> None:
    global _SCALED_MASKED_SOFTMAX_CONVERTER_REGISTERED
    if _SCALED_MASKED_SOFTMAX_CONVERTER_REGISTERED:
        return
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge._ge_graph import Tensor, TensorSpec

    op = torch.ops.npu.npu_scaled_masked_softmax.default

    @register_fx_node_ge_converter(op)
    def _convert_scaled_masked_softmax(
        x: Tensor,
        mask: Tensor,
        scale: float = 1.0,
        fixed_triu_mask: bool = False,
        meta_outputs: TensorSpec | None = None,
    ) -> Tensor:
        del meta_outputs
        return ge.ScaledMaskedSoftmax(
            x,
            mask,
            scale=float(scale),
            fixed_triu_mask=bool(fixed_triu_mask),
        )

    _SCALED_MASKED_SOFTMAX_CONVERTER_REGISTERED = True


def _query_positions(
    cache_position: torch.Tensor,
    query_length: int,
) -> torch.Tensor:
    start = cache_position.reshape(-1).to(dtype=torch.int64)
    offsets = torch.arange(
        int(query_length),
        device=cache_position.device,
        dtype=torch.int64,
    )
    return start.view(-1, 1) + offsets.view(1, -1)


def _update_spec_kv_cache_(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    positions: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> None:
    # torch_npu.scatter_update_ lowers to the dedicated Scatter KV-cache
    # template. It takes one start index per batch row, while updates carries
    # the full contiguous Q block written along the sequence axis.
    start_positions = positions[:, 0].to(
        device=key_cache.device,
        dtype=torch.int64,
    ).contiguous()
    if key_cache.device.type == "npu":
        import torch_npu

        torch_npu.scatter_update_(
            key_cache,
            start_positions,
            key_states.contiguous(),
            2,
        )
        torch_npu.scatter_update_(
            value_cache,
            start_positions,
            value_states.contiguous(),
            2,
        )
        return
    for batch_index in range(int(key_cache.shape[0])):
        key_cache[batch_index, :, positions[batch_index], :] = key_states[batch_index]
        value_cache[batch_index, :, positions[batch_index], :] = value_states[batch_index]


def _spec_attention(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    prepared_factors: tuple[torch.Tensor, torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    positions: torch.Tensor,
    attention_mask: torch.Tensor,
    legal_attention_mask: torch.Tensor | None,
    optimization: DecodeOptimizationConfig,
) -> torch.Tensor:
    if SPEC_VERIFY_ATTENTION == _COMBINED_QKV_POST_ROPE_ATTENTION:
        if not optimization.packed_qkv or optimization.rotary != "npu_apply":
            raise ValueError(
                "combined-QKV verifier layout requires packed QKV and NPU ApplyRotary"
            )
        batch_size, query_length, _hidden = hidden_states.shape
        query_heads = int(attention.num_heads)
        kv_heads = int(attention.num_key_value_heads)
        head_dim = int(attention.head_dim)
        packed_qkv = _linear_tokenwise(
            attention.decode_qkv_proj,
            hidden_states,
        )
        query_flat, key_flat, value_flat = packed_qkv.split(
            (
                query_heads * head_dim,
                kv_heads * head_dim,
                kv_heads * head_dim,
            ),
            dim=-1,
        )
        query_bsnd = query_flat.view(
            batch_size, query_length, query_heads, head_dim
        )
        key_bsnd = key_flat.view(
            batch_size, query_length, kv_heads, head_dim
        )
        value_bsnd = value_flat.view(
            batch_size, query_length, kv_heads, head_dim
        )
        import torch_npu

        cos, sin = prepared_factors
        query_bsnd, key_bsnd = torch_npu.npu_apply_rotary_pos_emb(
            query_bsnd,
            key_bsnd,
            cos,
            sin,
            layout="BSND",
            rotary_mode="half",
        )
        packed_head_major = (
            torch.cat((query_bsnd, key_bsnd, value_bsnd), dim=2)
            .transpose(1, 2)
            .contiguous()
        )
        query_states, key_states, value_states = packed_head_major.split(
            (query_heads, kv_heads, kv_heads),
            dim=1,
        )
    elif SPEC_VERIFY_ATTENTION == _COMBINED_QKV_LAYOUT_ATTENTION:
        if int(hidden_states.shape[0]) != 1:
            raise ValueError("combined-QKV verifier layout currently requires B1")
        if not optimization.packed_qkv or optimization.rotary != "npu_apply":
            raise ValueError(
                "combined-QKV verifier layout requires packed QKV and NPU ApplyRotary"
            )
        batch_size, query_length, _hidden = hidden_states.shape
        query_heads = int(attention.num_heads)
        kv_heads = int(attention.num_key_value_heads)
        head_dim = int(attention.head_dim)
        packed_qkv = _linear_tokenwise(
            attention.decode_qkv_proj,
            hidden_states,
        )
        # One materialization converts all packed Q, K, and V heads from
        # token-major to head-major layout. The subsequent B1 head-axis split
        # produces contiguous BNSD Q/K/V views without three separate
        # transposes.
        packed_head_major = (
            packed_qkv.view(
                batch_size,
                query_length,
                query_heads + (2 * kv_heads),
                head_dim,
            )
            .transpose(1, 2)
            .contiguous()
        )
        query_states, key_states, value_states = packed_head_major.split(
            (query_heads, kv_heads, kv_heads),
            dim=1,
        )
        import torch_npu

        cos, sin = prepared_factors
        query_states = torch_npu.npu_rotary_mul(
            query_states.contiguous(),
            cos,
            sin,
            rotary_mode="half",
        )
        key_states = torch_npu.npu_rotary_mul(
            key_states.contiguous(),
            cos,
            sin,
            rotary_mode="half",
        )
    else:
        query_states, key_states, value_states = _project_decode_qkv(
            attention,
            hidden_states,
            optimization,
        )
        query_states, key_states = _apply_decode_rotary(
            attention,
            query_states,
            key_states,
            position_embeddings,
            prepared_factors,
            optimization,
        )
    _update_spec_kv_cache_(
        key_cache,
        value_cache,
        positions,
        key_states,
        value_states,
    )
    if optimization.post_scatter_kv_prefetch:
        import torch_npu

        torch_npu.npu_prefetch(
            key_cache,
            key_states,
            int(key_cache.numel() * key_cache.element_size()),
        )
        torch_npu.npu_prefetch(
            value_cache,
            value_states,
            int(value_cache.numel() * value_cache.element_size()),
        )

    cache_length = int(key_cache.shape[2])
    batch_size = int(hidden_states.shape[0])

    if SPEC_VERIFY_ATTENTION == "promptfa_gqa":
        if query_states.device.type != "npu":
            raise ValueError("PromptFA speculative verification requires an NPU")
        import torch_npu

        attention_output = torch_npu.npu_prompt_flash_attention(
            query_states.contiguous(),
            key_cache.contiguous(),
            value_cache.contiguous(),
            atten_mask=attention_mask.contiguous(),
            num_heads=int(attention.num_heads),
            num_key_value_heads=int(attention.num_key_value_heads),
            input_layout="BNSD",
            scale_value=float(attention.scaling),
            pre_tokens=FULL_ATTENTION_TOKENS,
            next_tokens=FULL_ATTENTION_TOKENS,
            sparse_mode=0,
        )
        query_length = int(hidden_states.shape[1])
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .reshape(
                batch_size,
                query_length,
                attention.num_heads * attention.head_dim,
            )
        )
        return _linear_tokenwise(attention.o_proj, attention_output)

    if query_states.device.type != "npu":
        raise ValueError("group-folded manual attention is an NPU-only lab path")
    import torch_npu

    query_length = int(hidden_states.shape[1])
    head_dim = int(attention.head_dim)
    kv_heads = int(attention.num_key_value_heads)
    groups = int(attention.num_key_value_groups)
    query_heads = int(attention.num_heads)
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
    if SPEC_VERIFY_ATTENTION == "manual_grouped_scaled_masked_softmax":
        probabilities = torch_npu.npu_scaled_masked_softmax(
            scores,
            attention_mask.contiguous(),
            float(attention.scaling),
            False,
        )
    elif SPEC_VERIFY_ATTENTION in (
        "manual_grouped_legal_scaled_masked_softmax_fp16",
        "manual_grouped_legal_scaled_masked_softmax_fp32",
        _COMBINED_QKV_LAYOUT_ATTENTION,
        _COMBINED_QKV_POST_ROPE_ATTENTION,
    ):
        flattened_rows = batch_size * query_heads * query_length
        if (
            flattened_rows < 32
            or flattened_rows > 4096
            or flattened_rows % 32 != 0
            or cache_length < 32
            or cache_length > 4096
            or cache_length % 32 != 0
        ):
            raise ValueError(
                "legal scaled-masked-softmax requires flattened rows and "
                "cache length in [32, 4096] and divisible by 32"
            )
        legal_scores = scores.reshape(1, 1, flattened_rows, cache_length)
        if legal_attention_mask is None:
            raise ValueError("legal attention requires a hoisted legal mask")
        if SPEC_VERIFY_ATTENTION.endswith("_fp32"):
            legal_scores = (
                legal_scores * float(attention.scaling)
            ).float()
            probabilities = torch_npu.npu_scaled_masked_softmax(
                legal_scores,
                legal_attention_mask,
                1.0,
                False,
            ).to(query_states.dtype)
        else:
            probabilities = torch_npu.npu_scaled_masked_softmax(
                legal_scores,
                legal_attention_mask,
                float(attention.scaling),
                False,
            )
        probabilities = probabilities.view_as(scores)
    else:
        scores = scores * float(attention.scaling)
        scores = scores.masked_fill(
            attention_mask,
            torch.finfo(scores.dtype).min,
        )
        probabilities = torch.softmax(scores.float(), dim=-1).to(
            query_states.dtype
        )
    attention_output = torch.bmm(
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

    attention_output = (
        attention_output.transpose(1, 2).contiguous().reshape(
            batch_size,
            query_length,
            attention.num_heads * attention.head_dim,
        )
    )
    return _linear_tokenwise(attention.o_proj, attention_output)


def run_text_spec_verify_transformer(
    text_model: nn.Module,
    *,
    inputs_embeds: torch.Tensor,
    cache_position: torch.Tensor,
    rope_deltas: torch.Tensor,
    key_caches: tuple[torch.Tensor, ...],
    value_caches: tuple[torch.Tensor, ...],
    optimization: str | DecodeOptimizationConfig,
) -> torch.Tensor:
    """Run one B1 multi-token verification pass against a static KV arena."""
    optimization = resolve_decode_optimization(optimization)
    batch_size, query_length, _hidden = inputs_embeds.shape
    if int(cache_position.reshape(-1).numel()) != int(batch_size):
        raise ValueError("cache_position must contain one position per batch row")
    if not optimization.add_rms_norm:
        raise ValueError(
            "text speculative verification requires the optimized add-RMS path"
        )
    if optimization.rotary_factors not in ("mrope", "lookup"):
        raise ValueError(
            "text speculative verification requires MRoPE or RoPE lookup"
        )

    positions = _query_positions(cache_position, int(query_length))
    cache_length = int(key_caches[0].shape[2])
    kv_positions = torch.arange(
        cache_length,
        device=inputs_embeds.device,
        dtype=torch.int64,
    )
    attention_mask = kv_positions.view(1, 1, 1, cache_length) > positions.view(
        batch_size,
        1,
        -1,
        1,
    )
    legal_attention_mask: torch.Tensor | None = None
    if SPEC_VERIFY_ATTENTION in (
        "manual_grouped_legal_scaled_masked_softmax_fp16",
        "manual_grouped_legal_scaled_masked_softmax_fp32",
        _COMBINED_QKV_LAYOUT_ATTENTION,
        _COMBINED_QKV_POST_ROPE_ATTENTION,
    ):
        attention = text_model.layers[0].self_attn
        kv_heads = int(attention.num_key_value_heads)
        groups = int(attention.num_key_value_groups)
        flattened_rows = batch_size * int(attention.num_heads) * query_length
        legal_attention_mask = (
            attention_mask.view(
                batch_size, 1, 1, query_length, cache_length
            )
            .expand(
                batch_size,
                kv_heads,
                groups,
                query_length,
                cache_length,
            )
            .reshape(1, 1, flattened_rows, cache_length)
            .contiguous()
        )
    decode_positions = positions + rope_deltas.to(
        device=inputs_embeds.device,
        dtype=torch.int64,
    )
    if optimization.rotary_factors == "lookup":
        packed_factors = torch.index_select(
            text_model.rotary_emb.decode_rope_factor_lut,
            1,
            decode_positions.reshape(-1),
        )
        cosine, sine = packed_factors.unbind(dim=0)
        prepared_factors = (
            cosine.view(batch_size, query_length, -1).unsqueeze(1),
            sine.view(batch_size, query_length, -1).unsqueeze(1),
        )
        position_embeddings = prepared_factors
    else:
        position_ids = decode_positions.unsqueeze(0).expand(3, -1, -1)
        position_embeddings = text_model.rotary_emb(inputs_embeds, position_ids)
        prepared_factors = _prepare_multimodal_rotary_factors(
            position_embeddings,
            text_model.layers[0].self_attn.mrope_section,
        )
    if (
        optimization.rotary == "npu_apply"
        and SPEC_VERIFY_ATTENTION != _COMBINED_QKV_LAYOUT_ATTENTION
    ):
        # _prepare_multimodal_rotary_factors returns BNSD factors. The public
        # ApplyRotaryPosEmb BSND layout accepts [B,Q,1,D]; Q=1 made these two
        # representations indistinguishable in the ordinary decode graph.
        prepared_factors = (
            prepared_factors[0].transpose(1, 2).contiguous(),
            prepared_factors[1].transpose(1, 2).contiguous(),
        )

    hidden_states = inputs_embeds
    residual: torch.Tensor | None = None
    for layer_idx, layer in enumerate(text_model.layers):
        if optimization.complete_layer_prefetch_ahead:
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
        attention_output = _spec_attention(
            layer.self_attn,
            attention_input,
            position_embeddings,
            prepared_factors,
            key_caches[layer_idx],
            value_caches[layer_idx],
            positions,
            attention_mask,
            legal_attention_mask,
            optimization,
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


class TextSpecVerifyStage(nn.Module):
    """Static batched verifier for exactly ``draft_length`` draft tokens."""

    def __init__(
        self,
        model: "LocalPaddleOCRVLForConditionalGeneration",
        *,
        batch_size: int = 1,
        draft_length: int,
        optimization: str | DecodeOptimizationConfig = "combined_apply",
        token_selection: str = TOKEN_SELECTION_GREEDY,
        preferred_token_id: int | None = None,
        alternate_preferred_token_id: int | None = None,
        cell_start_token_ids: Iterable[int] = (),
    ):
        super().__init__()
        if int(draft_length) <= 0:
            raise ValueError("draft_length must be positive")
        self.model = model
        self.num_layers = int(model.config.text_config.num_hidden_layers)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.draft_length = int(draft_length)
        self.query_length = self.draft_length + 1
        self.optimization = resolve_decode_optimization(optimization)
        self.token_selection = str(token_selection)
        self.preferred_token_id = (
            None if preferred_token_id is None else int(preferred_token_id)
        )
        self.alternate_preferred_token_id = (
            None
            if alternate_preferred_token_id is None
            else int(alternate_preferred_token_id)
        )
        self.cell_start_token_ids = tuple(int(value) for value in cell_start_token_ids)
        self.compact_output_vocab = hasattr(model, "decode_token_id_map")
        if self.compact_output_vocab and self.token_selection != TOKEN_SELECTION_GREEDY:
            raise ValueError(
                "compact verifier output currently supports greedy selection only"
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        rope_deltas: torch.Tensor,
        *flat_cache_tensors: torch.Tensor,
    ) -> torch.Tensor:
        if int(input_ids.shape[0]) != self.batch_size:
            raise ValueError(
                f"expected batch size {self.batch_size}, "
                f"got {int(input_ids.shape[0])}"
            )
        if int(input_ids.shape[1]) != self.query_length:
            raise ValueError(
                f"expected query length {self.query_length}, "
                f"got {int(input_ids.shape[1])}"
            )
        key_caches = flat_cache_tensors[: self.num_layers]
        value_caches = flat_cache_tensors[self.num_layers :]
        inputs_embeds = self.model.model.embed_tokens(input_ids)
        hidden_states = run_text_spec_verify_transformer(
            self.model.model,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            rope_deltas=rope_deltas,
            key_caches=key_caches,
            value_caches=value_caches,
            optimization=self.optimization,
        )
        output_head = getattr(self.model, "decode_lm_head", self.model.lm_head)
        logits = _linear_tokenwise(output_head, hidden_states)
        if self.compact_output_vocab:
            compact_ids = torch.argmax(logits.float(), dim=-1)
            return self.model.decode_token_id_map.index_select(
                0,
                compact_ids.reshape(-1),
            ).view_as(compact_ids)
        cell_start_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.cell_start_token_ids:
            cell_start_mask |= input_ids == int(token_id)
        policy_mask = (
            torch.ones_like(input_ids, dtype=torch.bool)
            if self.token_selection in (
                TOKEN_SELECTION_SUPPRESS_MATH_OPEN_GREEDY,
                TOKEN_SELECTION_SUPPRESS_MATH_OPEN_AND_SLASH_GREEDY,
            )
            else cell_start_mask
        )
        return select_token_ids(
            logits,
            mode=self.token_selection,
            preferred_token_id=self.preferred_token_id,
            alternate_preferred_token_id=self.alternate_preferred_token_id,
            policy_mask=policy_mask,
            legacy_policy_mask=torch.ones_like(input_ids, dtype=torch.bool),
        )


def unique_spec_verify_forward(
    module: TextSpecVerifyStage,
    batch_size: int,
    draft_length: int,
) -> Callable[..., torch.Tensor]:
    """Give each static D shape an independent TorchDynamo code identity."""
    original = module.forward.__func__
    name = f"text_spec_verify_b{int(batch_size)}_draft_{int(draft_length)}"
    code = original.__code__.replace(co_name=name)
    function = types.FunctionType(
        code,
        original.__globals__,
        name,
        original.__defaults__,
        original.__closure__,
    )
    function.__annotations__ = dict(original.__annotations__)
    function.__kwdefaults__ = original.__kwdefaults__
    return types.MethodType(function, module)


def spec_verify_source_hash() -> str:
    here = Path(__file__).resolve().parent
    digest = hashlib.sha1()
    for name in ("text_prefill.py", "text_decode.py", "text_spec_verify.py"):
        path = here / name
        digest.update(name.encode("utf-8"))
        digest.update(short_file_hash(path).encode("utf-8"))
    return digest.hexdigest()[:12]


def bounded_spec_cache_component(shape_key: str, *, max_bytes: int = 240) -> str:
    """Keep one cache path component below common filesystem limits."""

    encoded = str(shape_key).encode("utf-8")
    if len(encoded) <= int(max_bytes):
        return str(shape_key)
    digest = hashlib.sha1(encoded).hexdigest()[:20]
    return f"text_spec_verify_key{digest}"


def torchair_cache_dir_for_spec_shape(
    cache_root: Path,
    *,
    batch_size: int = 1,
    draft_length: int,
    cache_length: int,
    dtype: torch.dtype,
    device: torch.device,
    model_dir: Path,
    linear_weight_format: str = DECODE_LINEAR_WEIGHT_FORMAT,
    optimization: str | DecodeOptimizationConfig = "combined_apply",
    token_selection: str = TOKEN_SELECTION_GREEDY,
    preferred_token_id: int | None = None,
    alternate_preferred_token_id: int | None = None,
    cell_start_token_ids: Iterable[int] = (),
) -> Path:
    optimization = resolve_decode_optimization(optimization)
    cell_start_token_ids = tuple(int(value) for value in cell_start_token_ids)
    shape_key = "_".join(
        [
            "text_spec_verify",
            linear_weight_format,
            SPEC_VERIFY_ATTENTION,
            SPEC_VERIFY_CACHE_UPDATE,
            f"opt{cache_key_part(optimization.name)}",
            f"select{cache_key_part(token_selection)}",
            f"preferred{preferred_token_id if preferred_token_id is not None else 'none'}",
            f"alternate{alternate_preferred_token_id if alternate_preferred_token_id is not None else 'none'}",
            "cellstart" + "-".join(str(value) for value in cell_start_token_ids),
            f"mode{cache_key_part(TORCHAIR_EXECUTION_MODE)}",
            f"dtype{cache_key_part(dtype)}",
            f"batch{int(batch_size)}",
            f"draft{int(draft_length)}",
            f"query{int(draft_length) + 1}",
            f"cache{int(cache_length)}",
            f"model{short_file_hash(model_dir / 'config.json')}",
            f"torch{cache_key_part(torch.__version__)}",
            f"torchnpu{torch_npu_version_label(device)}",
            f"torchair{torchair_version_label(device)}",
            f"src{spec_verify_source_hash()}",
        ]
    )
    return cache_root.expanduser().resolve() / bounded_spec_cache_component(shape_key)


class TextSpecVerifyRuntime:
    """Own one fixed-D compiled verification graph and warm KV arena."""

    def __init__(
        self,
        model: "LocalPaddleOCRVLForConditionalGeneration",
        *,
        batch_size: int = 1,
        device: torch.device,
        cache_root: Path,
        draft_length: int,
        cache_length: int,
        dtype: torch.dtype,
        model_dir: Path,
        linear_weight_format: str,
        optimization: str | DecodeOptimizationConfig = "combined_apply",
        token_selection: str = TOKEN_SELECTION_GREEDY,
        preferred_token_id: int | None = None,
        alternate_preferred_token_id: int | None = None,
        cell_start_token_ids: Iterable[int] = (),
    ):
        self.batch_size = int(batch_size)
        self.draft_length = int(draft_length)
        self.query_length = self.draft_length + 1
        self.cache_length = int(cache_length)
        self.optimization = prepare_decode_optimization_modules(
            model,
            optimization,
        )
        prepare_decode_weight_prefetch(model, self.optimization)
        if SPEC_VERIFY_ATTENTION != "manual_grouped_fp32":
            _register_scaled_masked_softmax_torchair_converter()
        self.token_selection = str(token_selection)
        self.preferred_token_id = (
            None if preferred_token_id is None else int(preferred_token_id)
        )
        self.alternate_preferred_token_id = (
            None
            if alternate_preferred_token_id is None
            else int(alternate_preferred_token_id)
        )
        self.cell_start_token_ids = tuple(int(value) for value in cell_start_token_ids)
        self.stage = TextSpecVerifyStage(
            model,
            batch_size=self.batch_size,
            draft_length=self.draft_length,
            optimization=self.optimization,
            token_selection=self.token_selection,
            preferred_token_id=self.preferred_token_id,
            alternate_preferred_token_id=self.alternate_preferred_token_id,
            cell_start_token_ids=self.cell_start_token_ids,
        ).eval()
        self.entrypoint = unique_spec_verify_forward(
            self.stage,
            self.batch_size,
            self.draft_length,
        )
        cache_dir = torchair_cache_dir_for_spec_shape(
            cache_root,
            batch_size=self.batch_size,
            draft_length=self.draft_length,
            cache_length=self.cache_length,
            dtype=dtype,
            device=device,
            model_dir=model_dir,
            linear_weight_format=linear_weight_format,
            optimization=self.optimization,
            token_selection=self.token_selection,
            preferred_token_id=self.preferred_token_id,
            alternate_preferred_token_id=self.alternate_preferred_token_id,
            cell_start_token_ids=self.cell_start_token_ids,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        torchair, CompilerConfig = import_torchair()
        synchronize(device)
        started = time.perf_counter()
        self.fn = torchair.inference.cache_compile(
            self.entrypoint,
            config=CompilerConfig(),
            dynamic=False,
            cache_dir=str(cache_dir),
            ge_cache=True,
        )
        synchronize(device)
        wrapper_s = time.perf_counter() - started

        self.warm_cache = model.allocate_static_cache(
            batch_size=self.batch_size,
            cache_length=self.cache_length,
            device=device,
            dtype=dtype,
            init_mode="zeros",
        )
        warm_input = torch.zeros(
            (self.batch_size, self.query_length),
            device=device,
            dtype=torch.int64,
        )
        warm_position = torch.ones(
            (self.batch_size,), device=device, dtype=torch.int64
        )
        warm_rope = torch.zeros(
            (self.batch_size, 1), device=device, dtype=torch.int64
        )
        synchronize(device)
        started = time.perf_counter()
        self.fn(
            warm_input,
            warm_position,
            warm_rope,
            *self.warm_cache.flat_tensors(),
        )
        synchronize(device)
        first_call_s = time.perf_counter() - started
        self.metadata: dict[str, Any] = {
            "boundary": "token_embedding_text_transformer_lm_head_argmax",
            "batch_size": self.batch_size,
            "draft_length": self.draft_length,
            "query_length": self.query_length,
            "recoverable_tokens_if_fully_accepted": self.query_length,
            "cache_length": self.cache_length,
            "attention": SPEC_VERIFY_ATTENTION,
            "cache_update": SPEC_VERIFY_CACHE_UPDATE,
            "optimization": self.optimization.name,
            "token_selection": self.token_selection,
            "preferred_token_id": self.preferred_token_id,
            "cell_start_token_ids": list(self.cell_start_token_ids),
            "output_vocab_size": int(
                getattr(model, "decode_lm_head", model.lm_head).out_features
            ),
            "compact_output_vocab": bool(hasattr(model, "decode_token_id_map")),
            "torchair_cache_dir": str(cache_dir),
            "compile_wrapper_s": float(wrapper_s),
            "compile_first_call_s": float(first_call_s),
        }
