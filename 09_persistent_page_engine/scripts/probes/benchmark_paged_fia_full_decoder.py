#!/usr/bin/env python3
"""Compare IncreFA and paged FIA v2 in a complete compiled Paddle decoder.

This is intentionally not an attention microbenchmark. It instantiates the
real PaddleOCR-VL text architecture with random weights and runs the complete
one-token decode step:

    embedding -> 18 decoder layers -> final norm -> LM head -> argmax

Both lanes share the same model weights and TorchAir static compilation. The
only material execution difference is dense-cache IncreFA versus a page-native
KV cache, page scatter, and ``torchair.ops`` FIA v2. Paged-cache scatter and
sequence-length metadata are built once per decode step and shared by all
decoder layers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch_npu
import torchair as tng
from torch import nn

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.compile_utils import import_torchair
from paddleocr_vl.model.config import PaddleOCRTextConfig, PaddleOCRVLConfig
from paddleocr_vl.model.text_decode import (
    LocalPaddleOCRVLStaticCache,
    TextDecodeRuntime,
    TextDecodeStage,
    _apply_decode_rotary,
    _decode_add_rms_norm,
    _decode_mlp,
    _decode_rms_norm,
    _linear_tokenwise,
    _prepare_multimodal_rotary_factors,
    _project_decode_qkv,
    cast_decode_linear_weights_to_nz,
    prepare_decode_optimization_modules,
)
from paddleocr_vl.model.text_prefill import PaddleOCRTextModel
from utils.timing import synchronize


DEFAULT_MODEL = Path("/workspace/models/PaddleOCR-VL-1.6")
DEFAULT_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_torchair"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/text_decode_lab"
    / "paged_fia_full_decoder_b1_k1024.json"
)
OPTIMIZATION = "combined_apply"
PA_NZ_LAST_DIM = 16
CACHE_UPDATE_MODES = (
    "scatter_nd",
    "scatter_pa",
    "scatter_pa_inplace",
)
PAGED_METADATA_MODES = (
    "dynamic_tensor",
    "fixed_bucket_mask",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cache-length", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--paged-cache-update",
        choices=CACHE_UPDATE_MODES,
        default="scatter_nd",
        help=(
            "Paged-cache writer inside the compiled decoder. "
            "scatter_pa uses torch_npu.npu_scatter_pa_kv_cache."
        ),
    )
    parser.add_argument(
        "--paged-metadata",
        choices=PAGED_METADATA_MODES,
        default="dynamic_tensor",
        help=(
            "Represent the valid KV prefix either as TorchAir tensor metadata "
            "or as a fixed host-list bucket plus a runtime attention mask."
        ),
    )
    parser.add_argument(
        "--attention-bucket-length",
        type=int,
        default=None,
        help=(
            "Static valid-prefix bucket used by fixed_bucket_mask. Defaults "
            "to --cache-length, but may be smaller than the physical cache."
        ),
    )
    parser.add_argument(
        "--paged-single-stream",
        action="store_true",
        help=(
            "Compile both comparison lanes with ge.enableSingleStream=true; "
            "TorchAir requires one global stream mode per process."
        ),
    )
    parser.add_argument(
        "--positions",
        default="127,511,768,1023",
        help="Comma-separated zero-based cache positions benchmarked in one graph.",
    )
    parser.add_argument(
        "--batch-positions",
        default=None,
        help=(
            "Optional comma-separated position for every batch row. When set, "
            "run one mixed-length row instead of repeating each --positions "
            "value across the batch."
        ),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--correctness-steps",
        type=int,
        default=1,
        help=(
            "Changing-position decode steps used to verify cache-state "
            "propagation through the compiled graph."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.cache_length <= 0:
        parser.error("--cache-length must be positive")
    if args.block_size <= 0 or args.cache_length % args.block_size:
        parser.error("--block-size must evenly divide --cache-length")
    if args.attention_bucket_length is None:
        args.attention_bucket_length = args.cache_length
    if (
        args.attention_bucket_length <= 0
        or args.attention_bucket_length > args.cache_length
    ):
        parser.error(
            "--attention-bucket-length must be positive and no larger than "
            "--cache-length"
        )
    if (
        args.warmup < 0
        or args.repeats <= 0
        or args.correctness_steps <= 0
    ):
        parser.error(
            "--warmup must be non-negative; --repeats and "
            "--correctness-steps must be positive"
        )
    try:
        args.positions = tuple(
            int(value.strip())
            for value in args.positions.split(",")
            if value.strip()
        )
    except ValueError as exc:
        parser.error(f"invalid --positions: {exc}")
    if not args.positions:
        parser.error("--positions must contain at least one position")
    if min(args.positions) < 0 or max(args.positions) >= args.cache_length:
        parser.error("--positions must be within the selected cache capacity")
    if args.batch_positions is not None:
        try:
            args.batch_positions = tuple(
                int(value.strip())
                for value in args.batch_positions.split(",")
                if value.strip()
            )
        except ValueError as exc:
            parser.error(f"invalid --batch-positions: {exc}")
        if len(args.batch_positions) != args.batch_size:
            parser.error(
                "--batch-positions must contain exactly --batch-size values"
            )
        if (
            min(args.batch_positions) < 0
            or max(args.batch_positions) >= args.cache_length
        ):
            parser.error(
                "--batch-positions must be within the selected cache capacity"
            )
        args.positions = (max(args.batch_positions),)
    if max(args.positions) >= args.attention_bucket_length:
        parser.error(
            "all runtime positions must fit inside --attention-bucket-length"
        )
    multistep_start = (
        max(args.batch_positions)
        if args.batch_positions is not None
        else args.positions[0]
    )
    if multistep_start + args.correctness_steps > args.attention_bucket_length:
        parser.error(
            "--correctness-steps would cross --attention-bucket-length"
        )
    return args


class RandomPaddleTextForCausalLM(nn.Module):
    """Text-only PaddleOCR-VL model with the real production dimensions."""

    def __init__(self, config: PaddleOCRVLConfig):
        super().__init__()
        self.config = config
        self.model = PaddleOCRTextModel(config.text_config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )

    def allocate_static_cache(
        self,
        *,
        batch_size: int,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
        init_mode: str = "zeros",
    ) -> LocalPaddleOCRVLStaticCache:
        return LocalPaddleOCRVLStaticCache.allocate(
            self.config.text_config,
            batch_size=batch_size,
            cache_length=cache_length,
            device=device,
            dtype=dtype,
            init_mode=init_mode,
        )


@dataclass
class PagedCache:
    key_caches: tuple[torch.Tensor, ...]
    value_caches: tuple[torch.Tensor, ...]
    block_table: torch.Tensor
    block_size: int
    cache_length: int

    def flat_tensors(self) -> tuple[torch.Tensor, ...]:
        return (*self.key_caches, *self.value_caches)


def _script_hash() -> str:
    digest = hashlib.sha1()
    with Path(__file__).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _parameter_counts(
    model: RandomPaddleTextForCausalLM,
) -> dict[str, int]:
    return {
        "decoder_layers": sum(
            parameter.numel()
            for parameter in model.model.layers.parameters()
        ),
        "token_embedding": model.model.embed_tokens.weight.numel(),
        "final_norm": sum(
            parameter.numel() for parameter in model.model.norm.parameters()
        ),
        "lm_head": model.lm_head.weight.numel(),
        "allocated_total": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }


def _create_random_model(
    config: PaddleOCRVLConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> RandomPaddleTextForCausalLM:
    torch.manual_seed(seed)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = RandomPaddleTextForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    return model.to(device).eval()


def _pa_nz_scatter_indices(
    slot_mapping: torch.Tensor,
    block_size: int,
    num_hidden_tiles: int,
) -> torch.Tensor:
    slots = slot_mapping.reshape(-1).to(dtype=torch.int64)
    physical_blocks = torch.div(
        slots,
        block_size,
        rounding_mode="floor",
    )
    offsets = torch.remainder(slots, block_size)
    hidden_tiles = torch.arange(
        num_hidden_tiles,
        device=slot_mapping.device,
        dtype=torch.int64,
    )
    batch_size = slots.shape[0]
    return torch.stack(
        (
            physical_blocks.view(-1, 1).expand(
                batch_size,
                num_hidden_tiles,
            ),
            hidden_tiles.view(1, -1).expand(
                batch_size,
                num_hidden_tiles,
            ),
            offsets.view(-1, 1).expand(
                batch_size,
                num_hidden_tiles,
            ),
        ),
        dim=-1,
    ).reshape(-1, 3)


def _pa_slot_mapping(
    cache_position: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    positions = cache_position.reshape(-1).to(dtype=torch.int64)
    logical_blocks = torch.div(
        positions,
        block_size,
        rounding_mode="floor",
    )
    physical_blocks = torch.gather(
        block_table,
        1,
        logical_blocks.view(-1, 1),
    ).reshape(-1)
    offsets = torch.remainder(positions, block_size)
    return physical_blocks * block_size + offsets


def _paged_decode_attention(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    prepared_factors: tuple[torch.Tensor, torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_update_metadata: torch.Tensor,
    actual_seq_qlen: torch.Tensor,
    actual_seq_kvlen: torch.Tensor,
    attention_mask: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    cache_update_mode: str,
    optimization: Any,
    native_fia: bool = False,
    fixed_actual_kv_lengths: tuple[int, ...] | None = None,
    native_fia_runner: Callable[..., torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    batch_size = query_states.shape[0]
    num_hidden_tiles = key_cache.shape[1]
    if cache_update_mode == "scatter_pa":
        updated_key_cache, updated_value_cache = (
            torch_npu.npu_scatter_pa_kv_cache_functional(
                key=key_states.squeeze(2).contiguous(),
                value=value_states.squeeze(2).contiguous(),
                key_cache=key_cache,
                value_cache=value_cache,
                slot_mapping=cache_update_metadata.contiguous(),
            )
        )
    elif cache_update_mode == "scatter_pa_inplace":
        torch_npu.npu_scatter_pa_kv_cache(
            key=key_states.squeeze(2).contiguous(),
            value=value_states.squeeze(2).contiguous(),
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=cache_update_metadata.contiguous(),
            cache_mode="PA_NZ",
        )
        updated_key_cache = key_cache
        updated_value_cache = value_cache
    else:
        key_updates = (
            key_states.squeeze(2)
            .contiguous()
            .view(batch_size, num_hidden_tiles, PA_NZ_LAST_DIM)
            .reshape(-1, PA_NZ_LAST_DIM)
        )
        value_updates = (
            value_states.squeeze(2)
            .contiguous()
            .view(batch_size, num_hidden_tiles, PA_NZ_LAST_DIM)
            .reshape(-1, PA_NZ_LAST_DIM)
        )
        torch_npu.npu_scatter_nd_update_(
            key_cache,
            cache_update_metadata,
            key_updates,
        )
        torch_npu.npu_scatter_nd_update_(
            value_cache,
            cache_update_metadata,
            value_updates,
        )
        updated_key_cache = key_cache
        updated_value_cache = value_cache
    key_cache_fia = updated_key_cache.view(
        updated_key_cache.shape[0],
        attention.num_key_value_heads,
        attention.head_dim // PA_NZ_LAST_DIM,
        block_size,
        PA_NZ_LAST_DIM,
    )
    value_cache_fia = updated_value_cache.view_as(key_cache_fia)
    if native_fia_runner is not None:
        if (
            fixed_actual_kv_lengths is None
            or len(fixed_actual_kv_lengths) != batch_size
        ):
            raise ValueError(
                "native FIA graph-task capture requires one fixed KV length "
                "for every batch row"
            )
        attention_output = native_fia_runner(
            query=query_states.contiguous(),
            key=key_cache_fia,
            value=value_cache_fia,
            block_table=block_table,
            block_size=block_size,
            actual_seq_kv_lengths=fixed_actual_kv_lengths,
            num_query_heads=int(attention.num_heads),
            num_key_value_heads=int(attention.num_key_value_heads),
            softmax_scale=float(attention.scaling),
        )
    elif native_fia:
        if (
            fixed_actual_kv_lengths is None
            or len(fixed_actual_kv_lengths) != batch_size
        ):
            raise ValueError(
                "native FIA graph capture requires one fixed KV length "
                "for every batch row"
            )
        attention_output = torch_npu.npu_fused_infer_attention_score_v2(
            query_states.contiguous(),
            key_cache_fia,
            value_cache_fia,
            num_query_heads=int(attention.num_heads),
            num_key_value_heads=int(attention.num_key_value_heads),
            input_layout="BNSD",
            softmax_scale=float(attention.scaling),
            atten_mask=attention_mask,
            actual_seq_qlen=[1] * batch_size,
            actual_seq_kvlen=list(fixed_actual_kv_lengths),
            block_table=block_table,
            block_size=block_size,
            sparse_mode=0,
            inner_precise=1,
        )[0]
    else:
        attention_output = tng.ops.npu_fused_infer_attention_score_v2(
            query_states.contiguous(),
            key_cache_fia,
            value_cache_fia,
            num_query_heads=int(attention.num_heads),
            num_key_value_heads=int(attention.num_key_value_heads),
            input_layout="BNSD",
            softmax_scale=float(attention.scaling),
            atten_mask=attention_mask,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            block_table=block_table,
            block_size=block_size,
            sparse_mode=0,
            inner_precise=1,
        )[0]
    attention_output = (
        attention_output.transpose(1, 2)
        .contiguous()
        .reshape(
            batch_size,
            1,
            attention.num_heads * attention.head_dim,
        )
    )
    return (
        _linear_tokenwise(attention.o_proj, attention_output),
        updated_key_cache,
        updated_value_cache,
    )


class PagedFIATextDecodeStage(nn.Module):
    """Complete production-shaped text decode with page-native attention."""

    def __init__(
        self,
        model: RandomPaddleTextForCausalLM,
        *,
        block_size: int,
        cache_update_mode: str,
        optimization: Any,
        native_fia: bool = False,
        fixed_actual_kv_lengths: Sequence[int] | None = None,
        native_fia_runner: Callable[..., torch.Tensor] | None = None,
    ):
        super().__init__()
        self.model = model
        self.block_size = int(block_size)
        self.cache_update_mode = cache_update_mode
        self.num_layers = int(model.config.text_config.num_hidden_layers)
        self.optimization = optimization
        self.native_fia = native_fia
        self.native_fia_runner = native_fia_runner
        self.fixed_actual_kv_lengths = (
            tuple(int(length) for length in fixed_actual_kv_lengths)
            if fixed_actual_kv_lengths is not None
            else None
        )
        if self.native_fia and self.fixed_actual_kv_lengths is None:
            raise ValueError(
                "native FIA requires fixed per-row KV bucket lengths"
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        rope_deltas: torch.Tensor,
        block_table: torch.Tensor,
        *flat_cache_tensors: torch.Tensor,
    ) -> Any:
        key_caches = flat_cache_tensors[: self.num_layers]
        value_caches = flat_cache_tensors[self.num_layers :]
        inputs_embeds = self.model.model.embed_tokens(input_ids)
        batch_size, sequence_length, _hidden_size = inputs_embeds.shape
        if sequence_length != 1:
            raise ValueError("paged decode expects one token per batch row")
        cache_position = cache_position.reshape(-1).to(
            device=inputs_embeds.device,
            dtype=torch.int64,
        )
        if cache_position.numel() == 1:
            cache_position = cache_position.expand(batch_size)
        position_ids = cache_position.view(batch_size, 1) + rope_deltas.to(
            device=inputs_embeds.device,
            dtype=torch.int64,
        )
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        position_embeddings = self.model.model.rotary_emb(
            inputs_embeds,
            position_ids,
        )
        prepared_factors = _prepare_multimodal_rotary_factors(
            position_embeddings,
            self.model.model.layers[0].self_attn.mrope_section,
        )
        slot_mapping = _pa_slot_mapping(
            cache_position,
            block_table,
            self.block_size,
        )
        if self.cache_update_mode in ("scatter_pa", "scatter_pa_inplace"):
            cache_update_metadata = slot_mapping
        else:
            cache_update_metadata = _pa_nz_scatter_indices(
                slot_mapping,
                self.block_size,
                key_caches[0].shape[1],
            )
        actual_seq_qlen = torch.ones_like(
            cache_position,
            dtype=torch.int64,
        )
        actual_seq_kvlen = cache_position + 1
        attention_mask = None
        if self.native_fia:
            mask_length = block_table.shape[1] * self.block_size
            logical_positions = torch.arange(
                mask_length,
                dtype=torch.int64,
                device=inputs_embeds.device,
            ).view(1, 1, 1, mask_length)
            attention_mask = logical_positions >= actual_seq_kvlen.view(
                batch_size,
                1,
                1,
                1,
            )

        hidden_states = inputs_embeds
        residual: torch.Tensor | None = None
        updated_key_caches: list[torch.Tensor] = []
        updated_value_caches: list[torch.Tensor] = []
        for layer_index, layer in enumerate(self.model.model.layers):
            if residual is None:
                attention_input = _decode_rms_norm(
                    layer.input_layernorm,
                    hidden_states,
                    self.optimization,
                )
                residual = hidden_states
            else:
                attention_input, residual = _decode_add_rms_norm(
                    hidden_states,
                    residual,
                    layer.input_layernorm,
                )
            (
                attention_output,
                updated_key_cache,
                updated_value_cache,
            ) = _paged_decode_attention(
                layer.self_attn,
                attention_input,
                position_embeddings,
                prepared_factors,
                key_caches[layer_index],
                value_caches[layer_index],
                cache_update_metadata,
                actual_seq_qlen,
                actual_seq_kvlen,
                attention_mask,
                block_table,
                self.block_size,
                self.cache_update_mode,
                self.optimization,
                self.native_fia,
                self.fixed_actual_kv_lengths,
                self.native_fia_runner,
            )
            if self.cache_update_mode == "scatter_pa":
                updated_key_caches.append(updated_key_cache)
                updated_value_caches.append(updated_value_cache)
            mlp_input, residual = _decode_add_rms_norm(
                attention_output,
                residual,
                layer.post_attention_layernorm,
            )
            hidden_states = _decode_mlp(
                layer.mlp,
                mlp_input,
                self.optimization,
            )
        hidden_states, _residual = _decode_add_rms_norm(
            hidden_states,
            residual,
            self.model.model.norm,
        )
        logits = _linear_tokenwise(
            self.model.lm_head,
            hidden_states[:, -1:, :],
        )
        if self.cache_update_mode == "scatter_pa":
            return (
                logits,
                *updated_key_caches,
                *updated_value_caches,
            )
        return logits


def _dense_to_paged_nz(
    tensor: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    batch_size, num_kv_heads, cache_length, head_dim = tensor.shape
    blocks_per_request = cache_length // block_size
    hidden_size = num_kv_heads * head_dim
    if hidden_size % PA_NZ_LAST_DIM:
        raise ValueError(
            "paged-NZ hidden size must be divisible by "
            f"{PA_NZ_LAST_DIM}: {hidden_size}"
        )
    normal_pages = (
        tensor.permute(0, 2, 1, 3)
        .contiguous()
        .view(
            batch_size,
            blocks_per_request,
            block_size,
            hidden_size,
        )
        .reshape(
            batch_size * blocks_per_request,
            block_size,
            hidden_size,
        )
    )
    return (
        normal_pages.view(
            batch_size * blocks_per_request,
            block_size,
            hidden_size // PA_NZ_LAST_DIM,
            PA_NZ_LAST_DIM,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def _allocate_matching_caches(
    config: PaddleOCRTextConfig,
    *,
    batch_size: int,
    cache_length: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[LocalPaddleOCRVLStaticCache, PagedCache]:
    torch.manual_seed(seed)
    dense_keys = []
    dense_values = []
    page_keys = []
    page_values = []
    shape = (
        batch_size,
        config.num_key_value_heads,
        cache_length,
        config.head_dim,
    )
    for _layer_index in range(config.num_hidden_layers):
        dense_key = torch.randn(shape, device=device, dtype=dtype) * 0.02
        dense_value = torch.randn(shape, device=device, dtype=dtype) * 0.02
        dense_keys.append(dense_key)
        dense_values.append(dense_value)
        page_keys.append(
            _dense_to_paged_nz(
                dense_key,
                block_size=block_size,
            ).clone()
        )
        page_values.append(
            _dense_to_paged_nz(
                dense_value,
                block_size=block_size,
            ).clone()
        )
    blocks_per_request = cache_length // block_size
    block_table = torch.arange(
        batch_size * blocks_per_request,
        device=device,
        dtype=torch.int32,
    ).view(batch_size, blocks_per_request)
    dense = LocalPaddleOCRVLStaticCache(
        tuple(dense_keys),
        tuple(dense_values),
        cache_length,
    )
    paged = PagedCache(
        tuple(page_keys),
        tuple(page_values),
        block_table,
        block_size,
        cache_length,
    )
    return dense, paged


def _page_cache_written_values(
    cache: PagedCache,
    cache_position: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    positions = cache_position.reshape(-1).to(torch.int64)
    logical_blocks = torch.div(
        positions,
        cache.block_size,
        rounding_mode="floor",
    )
    physical_blocks = torch.gather(
        cache.block_table,
        1,
        logical_blocks.view(-1, 1),
    ).reshape(-1)
    offsets = torch.remainder(positions, cache.block_size)
    outputs = []
    for tensor in cache.flat_tensors():
        blocks = tensor.index_select(0, physical_blocks)
        gather_index = offsets.view(-1, 1, 1, 1).expand(
            -1,
            blocks.shape[1],
            1,
            blocks.shape[3],
        )
        values = blocks.gather(2, gather_index).squeeze(2)
        outputs.append(values.reshape(positions.shape[0], -1))
    return tuple(outputs)


def _dense_cache_written_values(
    cache: LocalPaddleOCRVLStaticCache,
    cache_position: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    batch_indices = torch.arange(
        cache_position.numel(),
        device=cache_position.device,
        dtype=torch.int64,
    )
    return tuple(
        tensor[batch_indices, :, cache_position, :].reshape(
            cache_position.numel(),
            -1,
        )
        for tensor in cache.flat_tensors()
    )


def _delta_stats(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(delta.max().cpu()),
        "mean_abs": float(delta.mean().cpu()),
    }


def _cache_delta_stats(
    dense_values: tuple[torch.Tensor, ...],
    page_values: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    maximum = 0.0
    absolute_sum = 0.0
    count = 0
    per_tensor = []
    for tensor_index, (dense, paged) in enumerate(
        zip(dense_values, page_values)
    ):
        delta = (dense.float() - paged.float()).abs()
        tensor_maximum = float(delta.max().cpu())
        tensor_mean = float(delta.mean().cpu())
        maximum = max(maximum, tensor_maximum)
        absolute_sum += tensor_mean * delta.numel()
        count += delta.numel()
        per_tensor.append(
            {
                "tensor_index": tensor_index,
                "max_abs": tensor_maximum,
                "mean_abs": tensor_mean,
            }
        )
    return {
        "max_abs": maximum,
        "mean_abs": absolute_sum / count,
        "per_tensor": per_tensor,
    }


def _timed_decode(
    fn: Callable[..., Any],
    args: tuple[torch.Tensor, ...],
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
    state_prefix: int | None = None,
) -> dict[str, float]:
    current_args = args
    for _ in range(warmup):
        output = fn(*current_args)
        if state_prefix is None:
            logits = output
        else:
            logits = output[0]
            current_args = (*current_args[:state_prefix], *output[1:])
        torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
    synchronize(device)
    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        output = fn(*current_args)
        if state_prefix is None:
            logits = output
        else:
            logits = output[0]
            current_args = (*current_args[:state_prefix], *output[1:])
        torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
    end.record()
    end.synchronize()
    total_s = float(start.elapsed_time(end)) / 1000.0
    return {
        "total_s": total_s,
        "mean_ms": total_s * 1000.0 / repeats,
        "raw_tokens_per_s": repeats * int(args[0].shape[0]) / total_s,
    }


def _compile_paged_stage(
    stage: PagedFIATextDecodeStage,
    *,
    cache_root: Path,
    batch_size: int,
    cache_length: int,
    attention_bucket_length: int,
    block_size: int,
    cache_update_mode: str,
    metadata_mode: str,
    single_stream: bool,
) -> tuple[Callable[..., torch.Tensor], dict[str, Any]]:
    torchair, CompilerConfig = import_torchair()
    compiler_config = CompilerConfig()
    compiler_config.ge_config.enable_single_stream = bool(single_stream)
    stream_mode = "single_stream" if single_stream else "multi_stream"
    cache_dir = (
        cache_root.expanduser().resolve()
        / (
            f"paged_fia_v2_{OPTIMIZATION}_b{batch_size}_"
            f"k{cache_length}_a{attention_bucket_length}_"
            f"block{block_size}_{cache_update_mode}_"
            f"{metadata_mode}_"
            f"{stream_mode}_"
            f"src{_script_hash()}"
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    bucket_forward_name = (
        f"forward_b{batch_size}_k{cache_length}_"
        f"a{attention_bucket_length}_{metadata_mode}"
    )
    original_forward = stage.forward.__func__
    bucket_forward_code = original_forward.__code__.replace(
        co_name=bucket_forward_name,
        co_qualname=(
            f"{type(stage).__qualname__}.{bucket_forward_name}"
        ),
    )
    bucket_forward_function = types.FunctionType(
        bucket_forward_code,
        original_forward.__globals__,
        bucket_forward_name,
        original_forward.__defaults__,
        original_forward.__closure__,
    )
    bucket_forward_function.__kwdefaults__ = original_forward.__kwdefaults__
    bucket_forward = types.MethodType(bucket_forward_function, stage)
    started = time.perf_counter()
    fn = torchair.inference.cache_compile(
        bucket_forward,
        config=compiler_config,
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
    )
    return fn, {
        "compile_wrapper_s": time.perf_counter() - started,
        "cache_dir": str(cache_dir),
        "api": "torchair.inference.cache_compile",
        "dynamic": False,
        "cache_update_mode": cache_update_mode,
        "metadata_mode": metadata_mode,
        "attention_bucket_length": attention_bucket_length,
        "single_stream": bool(single_stream),
        "ge_enable_single_stream": bool(single_stream),
    }


@dataclass
class _ProbeTextDecodeRuntime:
    fn: Callable[..., torch.Tensor]
    metadata: dict[str, Any]
    setup_timing_s: dict[str, float]


def _single_stream_incre_runtime(
    model: nn.Module,
    *,
    device: torch.device,
    cache_root: Path,
    batch_size: int,
    cache_length: int,
    dtype: torch.dtype,
    optimization: Any,
) -> _ProbeTextDecodeRuntime:
    stage = TextDecodeStage(model, optimization=optimization).eval()
    torchair, CompilerConfig = import_torchair()
    compiler_config = CompilerConfig()
    compiler_config.ge_config.enable_single_stream = True
    cache_dir = (
        cache_root.expanduser().resolve()
        / (
            f"increfa_{OPTIMIZATION}_b{batch_size}_k{cache_length}_"
            f"single_stream_src{_script_hash()}"
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    wrapper_started = time.perf_counter()
    fn = torchair.inference.cache_compile(
        stage.forward,
        config=compiler_config,
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
    )
    compile_wrapper_s = time.perf_counter() - wrapper_started

    warm_cache = model.allocate_static_cache(
        batch_size=batch_size,
        cache_length=cache_length,
        device=device,
        dtype=dtype,
        init_mode="zeros",
    )
    warm_input = torch.zeros((batch_size, 1), device=device, dtype=torch.int64)
    warm_position = torch.ones((batch_size,), device=device, dtype=torch.int64)
    warm_rope = torch.zeros((batch_size, 1), device=device, dtype=torch.int64)
    synchronize(device)
    first_call_started = time.perf_counter()
    fn(
        warm_input,
        warm_position,
        warm_rope,
        *warm_cache.flat_tensors(),
    )
    synchronize(device)
    compile_first_call_s = time.perf_counter() - first_call_started
    return _ProbeTextDecodeRuntime(
        fn=fn,
        metadata={
            "backend": "torchair",
            "compile_api": "torchair.inference.cache_compile",
            "cache_dir": str(cache_dir),
            "dynamic": False,
            "single_stream": True,
            "ge_enable_single_stream": True,
        },
        setup_timing_s={
            "compile_wrapper": float(compile_wrapper_s),
            "compile_first_call": float(compile_first_call_s),
        },
    )


def _run_multistep_correctness(
    *,
    config: PaddleOCRVLConfig,
    incre_fn: Callable[..., torch.Tensor],
    paged_fn: Callable[..., torch.Tensor],
    batch_size: int,
    cache_length: int,
    block_size: int,
    initial_positions: Sequence[int],
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, Any]:
    dense_cache, paged_cache = _allocate_matching_caches(
        config.text_config,
        batch_size=batch_size,
        cache_length=cache_length,
        block_size=block_size,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    dense_input = (
        torch.arange(batch_size, device=device, dtype=torch.int64)
        .add_(17)
        .view(batch_size, 1)
    )
    paged_input = dense_input.clone()
    dense_positions = torch.tensor(
        initial_positions,
        device=device,
        dtype=torch.int64,
    )
    paged_positions = dense_positions.clone()
    dense_rope = torch.zeros(
        (batch_size, 1),
        device=device,
        dtype=torch.int64,
    )
    paged_rope = dense_rope.clone()
    rows = []
    passed = True
    for step in range(steps):
        dense_logits = incre_fn(
            dense_input,
            dense_positions,
            dense_rope,
            *dense_cache.flat_tensors(),
        )
        paged_logits = paged_fn(
            paged_input,
            paged_positions,
            paged_rope,
            paged_cache.block_table,
            *paged_cache.flat_tensors(),
        )
        synchronize(device)
        dense_tokens = torch.argmax(
            dense_logits[:, -1, :].float(),
            dim=-1,
            keepdim=True,
        )
        paged_tokens = torch.argmax(
            paged_logits[:, -1, :].float(),
            dim=-1,
            keepdim=True,
        )
        logits_delta = _delta_stats(paged_logits, dense_logits)
        cache_delta = _cache_delta_stats(
            _dense_cache_written_values(dense_cache, dense_positions),
            _page_cache_written_values(paged_cache, paged_positions),
        )
        argmax_matches = int((dense_tokens == paged_tokens).sum().cpu())
        step_passed = (
            argmax_matches == batch_size
            and cache_delta["mean_abs"] < 1e-3
        )
        passed = passed and step_passed
        rows.append(
            {
                "step": step,
                "positions": [
                    int(position)
                    for position in dense_positions.cpu().tolist()
                ],
                "argmax_matches": argmax_matches,
                "argmax_total": batch_size,
                "logits": logits_delta,
                "written_kv": {
                    "max_abs": cache_delta["max_abs"],
                    "mean_abs": cache_delta["mean_abs"],
                },
                "passed": step_passed,
            }
        )
        dense_input.copy_(dense_tokens)
        paged_input.copy_(paged_tokens)
        dense_positions.add_(1)
        paged_positions.add_(1)
    return {
        "passed": passed,
        "steps": steps,
        "initial_positions": list(initial_positions),
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.npu.is_available():
        raise RuntimeError("benchmark requires an available Ascend NPU")
    if not hasattr(tng.ops, "npu_fused_infer_attention_score_v2"):
        raise RuntimeError("TorchAir FIA v2 graph operation is unavailable")
    device = torch.device("npu:0")
    dtype = torch.float16
    torch.npu.set_compile_mode(jit_compile=False)
    config = PaddleOCRVLConfig.from_model_dir(args.model)

    synchronize(device)
    setup_started = time.perf_counter()
    model_started = time.perf_counter()
    model = _create_random_model(
        config,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )
    model_create_s = time.perf_counter() - model_started
    parameters_before_packing = _parameter_counts(model)
    optimization = prepare_decode_optimization_modules(
        model,
        OPTIMIZATION,
    )
    parameters_after_packing = _parameter_counts(model)
    format_started = time.perf_counter()
    weight_format = cast_decode_linear_weights_to_nz(model)
    synchronize(device)
    weight_format_s = time.perf_counter() - format_started
    linear_weight_format = str(weight_format["effective_mode"])

    incre_started = time.perf_counter()
    if args.paged_single_stream:
        incre_runtime = _single_stream_incre_runtime(
            model,
            device=device,
            cache_root=args.cache_dir,
            batch_size=args.batch_size,
            cache_length=args.cache_length,
            dtype=dtype,
            optimization=optimization,
        )
    else:
        incre_runtime = TextDecodeRuntime(
            model,
            backend="torchair",
            device=device,
            cache_root=args.cache_dir,
            batch_size=args.batch_size,
            cache_length=args.cache_length,
            dtype=dtype,
            model_dir=args.model,
            linear_weight_format=linear_weight_format,
            optimization=optimization,
        )
    incre_setup_s = time.perf_counter() - incre_started

    paged_stage = PagedFIATextDecodeStage(
        model,
        block_size=args.block_size,
        cache_update_mode=args.paged_cache_update,
        optimization=optimization,
        native_fia=args.paged_metadata == "fixed_bucket_mask",
        fixed_actual_kv_lengths=(
            [args.attention_bucket_length] * args.batch_size
            if args.paged_metadata == "fixed_bucket_mask"
            else None
        ),
    ).eval()
    paged_fn, paged_compile = _compile_paged_stage(
        paged_stage,
        cache_root=args.cache_dir,
        batch_size=args.batch_size,
        cache_length=args.cache_length,
        attention_bucket_length=args.attention_bucket_length,
        block_size=args.block_size,
        cache_update_mode=args.paged_cache_update,
        metadata_mode=args.paged_metadata,
        single_stream=args.paged_single_stream,
    )
    _warm_dense, warm_paged = _allocate_matching_caches(
        config.text_config,
        batch_size=args.batch_size,
        cache_length=args.cache_length,
        block_size=args.block_size,
        device=device,
        dtype=dtype,
        seed=args.seed + 1,
    )
    warm_input_ids = torch.zeros(
        (args.batch_size, 1),
        device=device,
        dtype=torch.int64,
    )
    warm_positions = torch.ones(
        (args.batch_size,),
        device=device,
        dtype=torch.int64,
    )
    warm_rope_deltas = torch.zeros(
        (args.batch_size, 1),
        device=device,
        dtype=torch.int64,
    )
    synchronize(device)
    paged_first_call_started = time.perf_counter()
    paged_fn(
        warm_input_ids,
        warm_positions,
        warm_rope_deltas,
        warm_paged.block_table,
        *warm_paged.flat_tensors(),
    )
    synchronize(device)
    paged_compile["first_call_s"] = (
        time.perf_counter() - paged_first_call_started
    )
    setup_s = time.perf_counter() - setup_started
    multistep_initial_positions = (
        args.batch_positions
        if args.batch_positions is not None
        else (args.positions[0],) * args.batch_size
    )
    if args.paged_cache_update == "scatter_pa":
        multistep_correctness = {
            "passed": True,
            "skipped": True,
            "reason": (
                "functional scatter_pa returns explicit cache state; this "
                "probe currently validates in-place cache writers"
            ),
        }
    else:
        multistep_correctness = _run_multistep_correctness(
            config=config,
            incre_fn=incre_runtime.fn,
            paged_fn=paged_fn,
            batch_size=args.batch_size,
            cache_length=args.cache_length,
            block_size=args.block_size,
            initial_positions=multistep_initial_positions,
            steps=args.correctness_steps,
            device=device,
            dtype=dtype,
            seed=args.seed + 50_000,
        )

    rows = []
    for position in args.positions:
        row_positions = (
            args.batch_positions
            if args.batch_positions is not None
            else (position,) * args.batch_size
        )
        dense_cache, paged_cache = _allocate_matching_caches(
            config.text_config,
            batch_size=args.batch_size,
            cache_length=args.cache_length,
            block_size=args.block_size,
            device=device,
            dtype=dtype,
            seed=args.seed + 1000 + sum(row_positions),
        )
        input_ids = (
            torch.arange(
                args.batch_size,
                device=device,
                dtype=torch.int64,
            ).view(-1, 1)
            + 17
        )
        cache_position = torch.tensor(
            row_positions,
            device=device,
            dtype=torch.int64,
        )
        rope_deltas = torch.zeros(
            (args.batch_size, 1),
            device=device,
            dtype=torch.int64,
        )
        page_values_before = _page_cache_written_values(
            paged_cache,
            cache_position,
        )

        incre_logits = incre_runtime.fn(
            input_ids,
            cache_position,
            rope_deltas,
            *dense_cache.flat_tensors(),
        )
        paged_output = paged_fn(
            input_ids,
            cache_position,
            rope_deltas,
            paged_cache.block_table,
            *paged_cache.flat_tensors(),
        )
        if args.paged_cache_update == "scatter_pa":
            paged_logits = paged_output[0]
            returned_cache_tensors = paged_output[1:]
            paged_cache_after = PagedCache(
                key_caches=tuple(
                    returned_cache_tensors[
                        : config.text_config.num_hidden_layers
                    ]
                ),
                value_caches=tuple(
                    returned_cache_tensors[
                        config.text_config.num_hidden_layers :
                    ]
                ),
                block_table=paged_cache.block_table,
                block_size=paged_cache.block_size,
                cache_length=paged_cache.cache_length,
            )
        else:
            paged_logits = paged_output
            paged_cache_after = paged_cache
        synchronize(device)
        incre_tokens = torch.argmax(
            incre_logits[:, -1, :].float(),
            dim=-1,
        )
        paged_tokens = torch.argmax(
            paged_logits[:, -1, :].float(),
            dim=-1,
        )
        logits_delta = _delta_stats(paged_logits, incre_logits)
        cache_delta = _cache_delta_stats(
            _dense_cache_written_values(dense_cache, cache_position),
            _page_cache_written_values(paged_cache_after, cache_position),
        )
        input_page_pool_change = _cache_delta_stats(
            _page_cache_written_values(paged_cache_after, cache_position),
            page_values_before,
        )
        first_layer_delta = {
            "key": cache_delta["per_tensor"][0],
            "value": cache_delta["per_tensor"][
                config.text_config.num_hidden_layers
            ],
        }

        incre_timing = _timed_decode(
            incre_runtime.fn,
            (
                input_ids,
                cache_position,
                rope_deltas,
                *dense_cache.flat_tensors(),
            ),
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )
        paged_timing = _timed_decode(
            paged_fn,
            (
                input_ids,
                cache_position,
                rope_deltas,
                paged_cache.block_table,
                *paged_cache.flat_tensors(),
            ),
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
            state_prefix=(
                4 if args.paged_cache_update == "scatter_pa" else None
            ),
        )
        rows.append(
            {
                "cache_position": position,
                "actual_kv_length": position + 1,
                "cache_positions": list(row_positions),
                "actual_kv_lengths": [
                    row_position + 1 for row_position in row_positions
                ],
                "increfa": incre_timing,
                "paged_fia_v2": paged_timing,
                "paged_vs_increfa_speedup": (
                    incre_timing["mean_ms"] / paged_timing["mean_ms"]
                ),
                "correctness": {
                    "logits": logits_delta,
                    "argmax_matches": int(
                        (incre_tokens == paged_tokens).sum().cpu()
                    ),
                    "argmax_total": args.batch_size,
                    "written_kv": cache_delta,
                    "first_layer_written_kv": first_layer_delta,
                    "input_page_pool_change": input_page_pool_change,
                },
            }
        )

    anchor = next(
        (
            row
            for row in rows
            if int(row["cache_position"]) == 768
        ),
        rows[-1],
    )
    result = {
        "schema_version": 1,
        "kind": "random_full_decoder_paged_fia_benchmark",
        "passed": (
            multistep_correctness["passed"]
            and all(
                row["correctness"]["first_layer_written_kv"]["key"][
                    "max_abs"
                ]
                == 0.0
                and row["correctness"]["first_layer_written_kv"]["value"][
                    "max_abs"
                ]
                == 0.0
                and row["correctness"]["input_page_pool_change"]["max_abs"]
                > 0.0
                and row["correctness"]["argmax_matches"]
                == row["correctness"]["argmax_total"]
                for row in rows
            )
        ),
        "configuration": {
            "model_config": str(args.model / "config.json"),
            "random_weights": True,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "cache_length": args.cache_length,
            "attention_bucket_length": args.attention_bucket_length,
            "block_size": args.block_size,
            "paged_cache_layout": "PA_NZ",
            "paged_cache_shape": (
                "[num_blocks,Nkv*head_dim/16,block_size,16]"
            ),
            "paged_fia_view": (
                "[num_blocks,Nkv,head_dim/16,block_size,16]"
            ),
            "paged_cache_state": (
                (
                    "explicit functional cache outputs from "
                    "npu_scatter_pa_kv_cache_functional"
                )
                if args.paged_cache_update == "scatter_pa"
                else (
                    "in-place graph inputs updated by "
                    "npu_scatter_nd_update_"
                )
            ),
            "paged_metadata": args.paged_metadata,
            "paged_cache_update": args.paged_cache_update,
            "paged_metadata_scope": (
                "once_per_decode_step_before_18_layer_loop"
            ),
            "paged_single_stream": args.paged_single_stream,
            "positions": list(args.positions),
            "batch_positions": (
                list(args.batch_positions)
                if args.batch_positions is not None
                else None
            ),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "correctness_steps": args.correctness_steps,
            "dtype": str(dtype),
            "optimization": OPTIMIZATION,
            "full_step": (
                "embedding_18_layers_final_norm_lm_head_argmax"
            ),
        },
        "architecture": {
            "hidden_size": config.text_config.hidden_size,
            "intermediate_size": config.text_config.intermediate_size,
            "num_hidden_layers": config.text_config.num_hidden_layers,
            "num_attention_heads": config.text_config.num_attention_heads,
            "num_key_value_heads": config.text_config.num_key_value_heads,
            "head_dim": config.text_config.head_dim,
            "vocab_size": config.text_config.vocab_size,
            "parameters_before_packed_qkv": parameters_before_packing,
            "parameters_after_packed_qkv": parameters_after_packing,
        },
        "versions": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "torchair": getattr(tng, "__version__", "unknown"),
        },
        "setup_s": {
            "total": setup_s,
            "random_model_create_and_transfer": model_create_s,
            "weight_format": weight_format_s,
            "increfa_runtime": incre_setup_s,
        },
        "weight_format": weight_format,
        "compile": {
            "increfa": incre_runtime.metadata,
            "increfa_setup_detail_s": incre_runtime.setup_timing_s,
            "paged_fia_v2": paged_compile,
        },
        "anchor": {
            "saved_b1_k1024_full_production_step_tok_per_s": 742.6,
            "saved_b1_k1024_model_and_argmax_tok_per_s": 749.6,
            "benchmark_boundary": "model_and_argmax",
            "selected_position": anchor["cache_position"],
            "measured_increfa_tok_per_s": anchor["increfa"][
                "raw_tokens_per_s"
            ],
        },
        "rows": rows,
        "multistep_correctness": multistep_correctness,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
