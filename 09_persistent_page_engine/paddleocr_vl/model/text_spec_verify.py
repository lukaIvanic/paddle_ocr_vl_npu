"""Compiled multi-token speculative verification for text decode.

The stage consumes the current token followed by ``draft_length`` proposed
tokens.  Its ``draft_length + 1`` logits predict the draft tokens and one
additional target token.  The graph also writes every query token into the
existing persistent KV arena; the caller decides how much of that tentative
tail to commit by advancing the logical cache position.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    resolve_decode_optimization,
)
from utils.timing import synchronize

if TYPE_CHECKING:
    from .modeling import LocalPaddleOCRVLForConditionalGeneration


FULL_ATTENTION_TOKENS = (1 << 31) - 1
SPEC_VERIFY_ATTENTION = "promptfa_gqa"
SPEC_VERIFY_CACHE_UPDATE = "npu_scatter"


def _query_positions(
    cache_position: torch.Tensor,
    query_length: int,
) -> torch.Tensor:
    start = cache_position.reshape(-1).to(dtype=torch.int64)
    if int(start.numel()) != 1:
        raise ValueError("text speculative verification currently requires B=1")
    offsets = torch.arange(
        int(query_length),
        device=cache_position.device,
        dtype=torch.int64,
    )
    return start + offsets


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
    start_positions = positions[:1].to(
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
    key_cache[:, :, positions, :] = key_states
    value_cache[:, :, positions, :] = value_states


def _spec_attention(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    prepared_factors: tuple[torch.Tensor, torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    positions: torch.Tensor,
    optimization: DecodeOptimizationConfig,
) -> torch.Tensor:
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

    cache_length = int(key_cache.shape[2])
    kv_positions = torch.arange(
        cache_length,
        device=hidden_states.device,
        dtype=torch.int64,
    )
    attention_mask = (
        kv_positions.view(1, 1, 1, cache_length)
        > positions.view(1, 1, -1, 1)
    )

    if query_states.device.type != "npu":
        groups = int(attention.num_key_value_groups)
        _batch, kv_heads, _kv_length, head_dim = key_cache.shape
        query_heads = int(attention.num_heads)
        expanded_key = (
            key_cache[:, :, None, :, :]
            .expand(1, kv_heads, groups, cache_length, head_dim)
            .reshape(query_heads, cache_length, head_dim)
        )
        expanded_value = (
            value_cache[:, :, None, :, :]
            .expand(1, kv_heads, groups, cache_length, head_dim)
            .reshape(query_heads, cache_length, head_dim)
        )
        query = query_states.reshape(query_heads, -1, head_dim)
        scores = torch.bmm(query, expanded_key.transpose(1, 2)).view(
            1, query_heads, query.shape[1], cache_length
        ) * float(attention.scaling)
        scores = scores.masked_fill(
            attention_mask,
            torch.finfo(scores.dtype).min,
        )
        probabilities = torch.softmax(scores.float(), dim=-1).to(
            query_states.dtype
        )
        attention_output = torch.bmm(
            probabilities.reshape(query_heads, query.shape[1], cache_length),
            expanded_value,
        ).view(1, query_heads, query.shape[1], head_dim)
    else:
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
        .reshape(1, query_length, attention.num_heads * attention.head_dim)
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
    if int(batch_size) != 1:
        raise ValueError("text speculative verification currently requires B=1")
    if not optimization.add_rms_norm:
        raise ValueError(
            "text speculative verification requires the optimized add-RMS path"
        )
    if optimization.rotary_factors != "mrope":
        raise ValueError("text speculative verification currently requires MRoPE")

    positions = _query_positions(cache_position, int(query_length))
    decode_positions = positions.view(1, query_length) + rope_deltas.to(
        device=inputs_embeds.device,
        dtype=torch.int64,
    )
    position_ids = decode_positions.unsqueeze(0).expand(3, -1, -1)
    position_embeddings = text_model.rotary_emb(inputs_embeds, position_ids)
    prepared_factors = _prepare_multimodal_rotary_factors(
        position_embeddings,
        text_model.layers[0].self_attn.mrope_section,
    )
    if optimization.rotary == "npu_apply":
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
    """Static B1 verifier for exactly ``draft_length`` draft tokens."""

    def __init__(
        self,
        model: "LocalPaddleOCRVLForConditionalGeneration",
        *,
        draft_length: int,
        optimization: str | DecodeOptimizationConfig = "combined_apply",
    ):
        super().__init__()
        if int(draft_length) <= 0:
            raise ValueError("draft_length must be positive")
        self.model = model
        self.num_layers = int(model.config.text_config.num_hidden_layers)
        self.draft_length = int(draft_length)
        self.query_length = self.draft_length + 1
        self.optimization = resolve_decode_optimization(optimization)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        rope_deltas: torch.Tensor,
        *flat_cache_tensors: torch.Tensor,
    ) -> torch.Tensor:
        if int(input_ids.shape[0]) != 1:
            raise ValueError("TextSpecVerifyStage requires batch size 1")
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
        logits = _linear_tokenwise(self.model.lm_head, hidden_states)
        return torch.argmax(logits, dim=-1)


def spec_verify_source_hash() -> str:
    here = Path(__file__).resolve().parent
    digest = hashlib.sha1()
    for name in ("text_prefill.py", "text_decode.py", "text_spec_verify.py"):
        path = here / name
        digest.update(name.encode("utf-8"))
        digest.update(short_file_hash(path).encode("utf-8"))
    return digest.hexdigest()[:12]


def torchair_cache_dir_for_spec_shape(
    cache_root: Path,
    *,
    draft_length: int,
    cache_length: int,
    dtype: torch.dtype,
    device: torch.device,
    model_dir: Path,
    linear_weight_format: str = DECODE_LINEAR_WEIGHT_FORMAT,
    optimization: str | DecodeOptimizationConfig = "combined_apply",
) -> Path:
    optimization = resolve_decode_optimization(optimization)
    shape_key = "_".join(
        [
            "text_spec_verify",
            linear_weight_format,
            SPEC_VERIFY_ATTENTION,
            SPEC_VERIFY_CACHE_UPDATE,
            f"opt{cache_key_part(optimization.name)}",
            f"mode{cache_key_part(TORCHAIR_EXECUTION_MODE)}",
            f"dtype{cache_key_part(dtype)}",
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
    return cache_root.expanduser().resolve() / shape_key


class TextSpecVerifyRuntime:
    """Own one fixed-D compiled verification graph and warm KV arena."""

    def __init__(
        self,
        model: "LocalPaddleOCRVLForConditionalGeneration",
        *,
        device: torch.device,
        cache_root: Path,
        draft_length: int,
        cache_length: int,
        dtype: torch.dtype,
        model_dir: Path,
        linear_weight_format: str,
        optimization: str | DecodeOptimizationConfig = "combined_apply",
    ):
        self.draft_length = int(draft_length)
        self.query_length = self.draft_length + 1
        self.cache_length = int(cache_length)
        self.optimization = prepare_decode_optimization_modules(
            model,
            optimization,
        )
        self.stage = TextSpecVerifyStage(
            model,
            draft_length=self.draft_length,
            optimization=self.optimization,
        ).eval()
        cache_dir = torchair_cache_dir_for_spec_shape(
            cache_root,
            draft_length=self.draft_length,
            cache_length=self.cache_length,
            dtype=dtype,
            device=device,
            model_dir=model_dir,
            linear_weight_format=linear_weight_format,
            optimization=self.optimization,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        torchair, CompilerConfig = import_torchair()
        synchronize(device)
        started = time.perf_counter()
        self.fn = torchair.inference.cache_compile(
            self.stage.forward,
            config=CompilerConfig(),
            dynamic=False,
            cache_dir=str(cache_dir),
            ge_cache=True,
        )
        synchronize(device)
        wrapper_s = time.perf_counter() - started

        self.warm_cache = model.allocate_static_cache(
            batch_size=1,
            cache_length=self.cache_length,
            device=device,
            dtype=dtype,
            init_mode="zeros",
        )
        warm_input = torch.zeros(
            (1, self.query_length),
            device=device,
            dtype=torch.int64,
        )
        warm_position = torch.ones((1,), device=device, dtype=torch.int64)
        warm_rope = torch.zeros((1, 1), device=device, dtype=torch.int64)
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
            "batch_size": 1,
            "draft_length": self.draft_length,
            "query_length": self.query_length,
            "recoverable_tokens_if_fully_accepted": self.query_length,
            "cache_length": self.cache_length,
            "attention": SPEC_VERIFY_ATTENTION,
            "cache_update": SPEC_VERIFY_CACHE_UPDATE,
            "optimization": self.optimization.name,
            "torchair_cache_dir": str(cache_dir),
            "compile_wrapper_s": float(wrapper_s),
            "compile_first_call_s": float(first_call_s),
        }
