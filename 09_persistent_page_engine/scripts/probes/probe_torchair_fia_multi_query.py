#!/usr/bin/env python3
"""Validate reusable, multi-query paged FIA in a full compiled decoder.

The existing paged-FIA benchmark validates one query token per request.  This
probe exercises the stronger serving contract: every request may contribute
multiple consecutive query tokens in the same model invocation, while all KV
state remains in a page-native PA_NZ cache.

The compiled graph has fixed ``[B, Q_bucket]`` inputs.  Runtime query lengths
select the useful prefix in each row.  Unused query slots write to private
scratch pages that are absent from every request block table, and a runtime
causal mask prevents them from affecting useful outputs.  Consequently the
same graph can execute changing ragged query patterns without FIA graph-task
updates.

Correctness is checked against sequential one-token IncreFA execution using the
same random production-sized PaddleOCR-VL decoder weights and matching initial
KV state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch_npu
from torch import nn

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENT_ROOT))

import benchmark_paged_fia_full_decoder as base

from paddleocr_vl.model.compile_utils import import_torchair
from paddleocr_vl.model.text_decode import (
    LocalPaddleOCRVLStaticCache,
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
from utils.timing import synchronize


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/text_decode_lab"
    / "torchair_fia_multi_query_b4_q8.json"
)


def _parse_patterns(
    value: str,
    *,
    batch_size: int,
    query_bucket: int,
) -> tuple[tuple[int, ...], ...]:
    patterns = []
    for raw_pattern in value.split(";"):
        pattern = tuple(
            int(item.strip())
            for item in raw_pattern.split(",")
            if item.strip()
        )
        if len(pattern) != batch_size:
            raise ValueError(
                "each query pattern must contain exactly "
                f"{batch_size} lengths, got {pattern}"
            )
        if min(pattern) <= 0 or max(pattern) > query_bucket:
            raise ValueError(
                "query lengths must be in [1, query_bucket], got "
                f"{pattern} for bucket {query_bucket}"
            )
        patterns.append(pattern)
    if not patterns:
        raise ValueError("at least one query pattern is required")
    return tuple(patterns)


def _parse_int_tuple(
    value: str,
    *,
    expected: int,
    label: str,
) -> tuple[int, ...]:
    parsed = tuple(
        int(item.strip()) for item in value.split(",") if item.strip()
    )
    if len(parsed) != expected:
        raise ValueError(
            f"{label} must contain exactly {expected} values, got {parsed}"
        )
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=base.DEFAULT_MODEL)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=base.DEFAULT_CACHE_ROOT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--query-bucket", type=int, default=8)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--attention-bucket-length", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--query-patterns",
        default="1,2,4,8;8,4,2,1;3,1,5,2",
        help=(
            "Semicolon-separated runtime query-length patterns. Every "
            "pattern must have --batch-size entries."
        ),
    )
    parser.add_argument(
        "--initial-positions",
        default="63,127,254,508",
        help="Comma-separated zero-based first query position per request.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.query_bucket <= 0:
        parser.error("--batch-size and --query-bucket must be positive")
    if args.cache_length <= 0:
        parser.error("--cache-length must be positive")
    if args.block_size <= 0 or args.cache_length % args.block_size:
        parser.error("--block-size must evenly divide --cache-length")
    if (
        args.attention_bucket_length <= 0
        or args.attention_bucket_length > args.cache_length
    ):
        parser.error(
            "--attention-bucket-length must be positive and no larger than "
            "--cache-length"
        )
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats positive")
    try:
        args.query_patterns = _parse_patterns(
            args.query_patterns,
            batch_size=args.batch_size,
            query_bucket=args.query_bucket,
        )
        args.initial_positions = _parse_int_tuple(
            args.initial_positions,
            expected=args.batch_size,
            label="--initial-positions",
        )
    except ValueError as exc:
        parser.error(str(exc))
    if min(args.initial_positions) < 0:
        parser.error("--initial-positions must be non-negative")
    final_positions = [
        args.initial_positions[row]
        + sum(pattern[row] for pattern in args.query_patterns)
        - 1
        for row in range(args.batch_size)
    ]
    if max(final_positions) >= args.attention_bucket_length:
        parser.error(
            "the multi-step query patterns exceed the selected attention "
            f"bucket: final positions {final_positions}"
        )
    return args


def _source_hash() -> str:
    digest = hashlib.sha1()
    for path in (
        Path(__file__),
        HERE / "benchmark_paged_fia_full_decoder.py",
        EXPERIMENT_ROOT / "paddleocr_vl/model/text_decode.py",
        EXPERIMENT_ROOT / "paddleocr_vl/model/text_prefill.py",
    ):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:12]


class MultiQueryPagedFIAStage(nn.Module):
    """Production-sized decoder over a reusable ragged-query bucket."""

    def __init__(
        self,
        model: base.RandomPaddleTextForCausalLM,
        *,
        batch_size: int,
        query_bucket: int,
        cache_length: int,
        attention_bucket_length: int,
        block_size: int,
        dummy_slot_base: int,
        optimization: Any,
    ):
        super().__init__()
        self.model = model
        self.batch_size = int(batch_size)
        self.query_bucket = int(query_bucket)
        self.cache_length = int(cache_length)
        self.attention_bucket_length = int(attention_bucket_length)
        self.block_size = int(block_size)
        self.dummy_slot_base = int(dummy_slot_base)
        self.num_layers = int(model.config.text_config.num_hidden_layers)
        self.optimization = optimization
        self.fixed_actual_query_lengths = tuple(
            [self.query_bucket] * self.batch_size
        )
        self.fixed_actual_kv_lengths = tuple(
            [self.attention_bucket_length] * self.batch_size
        )

    def _slot_mapping(
        self,
        positions: torch.Tensor,
        query_lengths: torch.Tensor,
        block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_indices = torch.arange(
            self.query_bucket,
            device=positions.device,
            dtype=torch.int64,
        ).view(1, -1)
        real_query_mask = query_indices < query_lengths.view(-1, 1)
        logical_blocks = torch.div(
            positions,
            self.block_size,
            rounding_mode="floor",
        )
        physical_blocks = torch.gather(
            block_table,
            1,
            logical_blocks,
        ).to(torch.int64)
        real_slots = (
            physical_blocks * self.block_size
            + torch.remainder(positions, self.block_size)
        )
        dummy_slots = (
            torch.arange(
                self.batch_size * self.query_bucket,
                device=positions.device,
                dtype=torch.int64,
            ).view(self.batch_size, self.query_bucket)
            + self.dummy_slot_base
        )
        return (
            torch.where(real_query_mask, real_slots, dummy_slots),
            real_query_mask,
        )

    def _attention_mask(
        self,
        positions: torch.Tensor,
        real_query_mask: torch.Tensor,
    ) -> torch.Tensor:
        kv_positions = torch.arange(
            self.cache_length,
            device=positions.device,
            dtype=torch.int64,
        ).view(1, 1, 1, self.cache_length)
        useful_allowed = kv_positions <= positions.view(
            self.batch_size,
            1,
            self.query_bucket,
            1,
        )
        # Avoid a fully masked softmax row for padded query slots. Their
        # outputs are discarded, and their KV writes target scratch pages.
        dummy_allowed = kv_positions == 0
        allowed = torch.where(
            real_query_mask.view(
                self.batch_size,
                1,
                self.query_bucket,
                1,
            ),
            useful_allowed,
            dummy_allowed,
        )
        return ~allowed

    def _attention(
        self,
        attention: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        prepared_factors: tuple[torch.Tensor, torch.Tensor],
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_update_indices: torch.Tensor,
        attention_mask: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        query_states, key_states, value_states = _project_decode_qkv(
            attention,
            hidden_states,
            self.optimization,
        )
        query_states, key_states = _apply_decode_rotary(
            attention,
            query_states,
            key_states,
            position_embeddings,
            prepared_factors,
            self.optimization,
        )
        token_count = self.batch_size * self.query_bucket
        hidden_tiles = key_cache.shape[1]
        key_updates = (
            key_states.transpose(1, 2)
            .contiguous()
            .view(token_count, hidden_tiles, base.PA_NZ_LAST_DIM)
            .reshape(-1, base.PA_NZ_LAST_DIM)
        )
        value_updates = (
            value_states.transpose(1, 2)
            .contiguous()
            .view(token_count, hidden_tiles, base.PA_NZ_LAST_DIM)
            .reshape(-1, base.PA_NZ_LAST_DIM)
        )
        torch_npu.npu_scatter_nd_update_(
            key_cache,
            cache_update_indices,
            key_updates,
        )
        torch_npu.npu_scatter_nd_update_(
            value_cache,
            cache_update_indices,
            value_updates,
        )
        key_cache_fia = key_cache.view(
            key_cache.shape[0],
            attention.num_key_value_heads,
            attention.head_dim // base.PA_NZ_LAST_DIM,
            self.block_size,
            base.PA_NZ_LAST_DIM,
        )
        value_cache_fia = value_cache.view_as(key_cache_fia)
        attention_output = torch_npu.npu_fused_infer_attention_score_v2(
            query_states.contiguous(),
            key_cache_fia,
            value_cache_fia,
            num_query_heads=int(attention.num_heads),
            num_key_value_heads=int(attention.num_key_value_heads),
            input_layout="BNSD",
            softmax_scale=float(attention.scaling),
            atten_mask=attention_mask,
            actual_seq_qlen=list(self.fixed_actual_query_lengths),
            actual_seq_kvlen=list(self.fixed_actual_kv_lengths),
            block_table=block_table,
            block_size=self.block_size,
            sparse_mode=0,
            inner_precise=1,
        )[0]
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .reshape(
                self.batch_size,
                self.query_bucket,
                attention.num_heads * attention.head_dim,
            )
        )
        return _linear_tokenwise(attention.o_proj, attention_output)

    def forward(
        self,
        input_ids: torch.Tensor,
        start_positions: torch.Tensor,
        query_lengths: torch.Tensor,
        rope_deltas: torch.Tensor,
        block_table: torch.Tensor,
        *flat_cache_tensors: torch.Tensor,
    ) -> torch.Tensor:
        key_caches = flat_cache_tensors[: self.num_layers]
        value_caches = flat_cache_tensors[self.num_layers :]
        inputs_embeds = self.model.model.embed_tokens(input_ids)
        query_indices = torch.arange(
            self.query_bucket,
            device=inputs_embeds.device,
            dtype=torch.int64,
        ).view(1, -1)
        positions = (
            start_positions.reshape(-1, 1).to(torch.int64)
            + query_indices
        )
        slot_mapping, real_query_mask = self._slot_mapping(
            positions,
            query_lengths.to(torch.int64),
            block_table,
        )
        cache_update_indices = base._pa_nz_scatter_indices(
            slot_mapping.reshape(-1),
            self.block_size,
            key_caches[0].shape[1],
        )
        attention_mask = self._attention_mask(
            positions,
            real_query_mask,
        )
        position_ids = (
            positions + rope_deltas.to(torch.int64)
        ).unsqueeze(0).expand(3, -1, -1)
        position_embeddings = self.model.model.rotary_emb(
            inputs_embeds,
            position_ids,
        )
        prepared_factors = _prepare_multimodal_rotary_factors(
            position_embeddings,
            self.model.model.layers[0].self_attn.mrope_section,
        )
        if self.optimization.rotary == "npu_apply":
            # The shared decode helper only sees Q=1 in production today, so
            # its [B,1,Q,D] and the BSND operator's required [B,Q,1,D]
            # factors are accidentally indistinguishable there.
            prepared_factors = (
                prepared_factors[0].transpose(1, 2).contiguous(),
                prepared_factors[1].transpose(1, 2).contiguous(),
            )

        hidden_states = inputs_embeds
        residual: torch.Tensor | None = None
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
            attention_output = self._attention(
                layer.self_attn,
                attention_input,
                position_embeddings,
                prepared_factors,
                key_caches[layer_index],
                value_caches[layer_index],
                cache_update_indices,
                attention_mask,
                block_table,
            )
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
        last_indices = (
            query_lengths.to(torch.int64) - 1
        ).view(self.batch_size, 1, 1).expand(
            self.batch_size,
            1,
            hidden_states.shape[-1],
        )
        last_hidden = torch.gather(hidden_states, 1, last_indices)
        return _linear_tokenwise(self.model.lm_head, last_hidden)


def _allocate_matching_multi_query_caches(
    config: base.PaddleOCRVLConfig,
    *,
    batch_size: int,
    query_bucket: int,
    cache_length: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[
    tuple[LocalPaddleOCRVLStaticCache, ...],
    base.PagedCache,
    int,
]:
    dense, paged = base._allocate_matching_caches(
        config.text_config,
        batch_size=batch_size,
        cache_length=cache_length,
        block_size=block_size,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    dense_rows = tuple(
        LocalPaddleOCRVLStaticCache(
            tuple(
                tensor[row : row + 1].clone()
                for tensor in dense.key_caches
            ),
            tuple(
                tensor[row : row + 1].clone()
                for tensor in dense.value_caches
            ),
            cache_length,
        )
        for row in range(batch_size)
    )
    real_page_count = int(paged.key_caches[0].shape[0])
    dummy_page_count = math.ceil(
        batch_size * query_bucket / block_size
    )

    def with_dummy_pages(tensor: torch.Tensor) -> torch.Tensor:
        dummy = torch.zeros(
            (dummy_page_count, *tensor.shape[1:]),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        return torch.cat((tensor, dummy), dim=0).contiguous()

    paged_with_scratch = base.PagedCache(
        tuple(with_dummy_pages(tensor) for tensor in paged.key_caches),
        tuple(with_dummy_pages(tensor) for tensor in paged.value_caches),
        paged.block_table,
        block_size,
        cache_length,
    )
    return (
        dense_rows,
        paged_with_scratch,
        real_page_count * block_size,
    )


def _compile_stage(
    stage: MultiQueryPagedFIAStage,
    *,
    cache_root: Path,
) -> tuple[Callable[..., torch.Tensor], dict[str, Any]]:
    torchair, CompilerConfig = import_torchair()
    compiler_config = CompilerConfig()
    compiler_config.ge_config.enable_single_stream = False
    cache_dir = (
        cache_root.expanduser().resolve()
        / (
            f"paged_fia_multi_query_{base.OPTIMIZATION}_"
            f"b{stage.batch_size}_q{stage.query_bucket}_"
            f"k{stage.cache_length}_a{stage.attention_bucket_length}_"
            f"block{stage.block_size}_src{_source_hash()}"
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    forward_name = (
        f"forward_b{stage.batch_size}_q{stage.query_bucket}_"
        f"k{stage.cache_length}_a{stage.attention_bucket_length}"
    )
    original_forward = stage.forward.__func__
    forward_code = original_forward.__code__.replace(
        co_name=forward_name,
        co_qualname=f"{type(stage).__qualname__}.{forward_name}",
    )
    forward_function = types.FunctionType(
        forward_code,
        original_forward.__globals__,
        forward_name,
        original_forward.__defaults__,
        original_forward.__closure__,
    )
    forward_function.__kwdefaults__ = original_forward.__kwdefaults__
    bucket_forward = types.MethodType(forward_function, stage)
    wrapper_started = time.perf_counter()
    fn = torchair.inference.cache_compile(
        bucket_forward,
        config=compiler_config,
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
    )
    return fn, {
        "api": "torchair.inference.cache_compile",
        "cache_dir": str(cache_dir),
        "dynamic": False,
        "compile_wrapper_s": time.perf_counter() - wrapper_started,
        "ge_enable_single_stream": False,
        "source_hash": _source_hash(),
    }


def _input_ids(
    *,
    batch_size: int,
    query_bucket: int,
    vocab_size: int,
    step: int,
    device: torch.device,
) -> torch.Tensor:
    return (
        torch.arange(
            batch_size * query_bucket,
            device=device,
            dtype=torch.int64,
        ).view(batch_size, query_bucket)
        .add_(17 + step * 97)
        .remainder_(vocab_size)
    )


def _run_sequential_reference(
    stage: TextDecodeStage,
    *,
    input_ids: torch.Tensor,
    start_positions: torch.Tensor,
    query_lengths: Sequence[int],
    rope_deltas: torch.Tensor,
    dense_rows: Sequence[LocalPaddleOCRVLStaticCache],
) -> torch.Tensor:
    last_logits = []
    for row, query_length in enumerate(query_lengths):
        logits = None
        for query_index in range(query_length):
            logits = stage(
                input_ids[row : row + 1, query_index : query_index + 1],
                start_positions[row : row + 1] + query_index,
                rope_deltas[row : row + 1],
                *dense_rows[row].flat_tensors(),
            )
        if logits is None:
            raise AssertionError("query lengths are required to be positive")
        last_logits.append(logits)
    return torch.cat(last_logits, dim=0)


def _written_cache_values(
    *,
    dense_rows: Sequence[LocalPaddleOCRVLStaticCache],
    paged_cache: base.PagedCache,
    start_positions: Sequence[int],
    query_lengths: Sequence[int],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    dense_outputs: list[torch.Tensor] = []
    paged_outputs: list[torch.Tensor] = []
    row_indices = []
    positions = []
    for row, query_length in enumerate(query_lengths):
        row_indices.extend([row] * query_length)
        positions.extend(
            range(
                start_positions[row],
                start_positions[row] + query_length,
            )
        )
    row_tensor = torch.tensor(
        row_indices,
        device=paged_cache.block_table.device,
        dtype=torch.int64,
    )
    position_tensor = torch.tensor(
        positions,
        device=paged_cache.block_table.device,
        dtype=torch.int64,
    )
    logical_blocks = torch.div(
        position_tensor,
        paged_cache.block_size,
        rounding_mode="floor",
    )
    physical_blocks = paged_cache.block_table[
        row_tensor,
        logical_blocks,
    ].to(torch.int64)
    offsets = torch.remainder(
        position_tensor,
        paged_cache.block_size,
    )
    for cache_index, paged_tensor in enumerate(
        paged_cache.flat_tensors()
    ):
        is_value = cache_index >= len(paged_cache.key_caches)
        layer_index = (
            cache_index - len(paged_cache.key_caches)
            if is_value
            else cache_index
        )
        dense_values_for_cache = []
        for row, position in zip(row_indices, positions):
            source = (
                dense_rows[row].value_caches[layer_index]
                if is_value
                else dense_rows[row].key_caches[layer_index]
            )
            dense_values_for_cache.append(
                source[0, :, position, :].reshape(-1)
            )
        dense_outputs.append(torch.stack(dense_values_for_cache))
        blocks = paged_tensor.index_select(0, physical_blocks)
        gather_index = offsets.view(-1, 1, 1, 1).expand(
            -1,
            blocks.shape[1],
            1,
            blocks.shape[3],
        )
        paged_outputs.append(
            blocks.gather(2, gather_index).squeeze(2).reshape(
                len(positions),
                -1,
            )
        )
    return tuple(dense_outputs), tuple(paged_outputs)


def _timed_graph(
    fn: Callable[..., torch.Tensor],
    args: tuple[torch.Tensor, ...],
    *,
    physical_tokens: int,
    effective_tokens: int,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict[str, float]:
    for _ in range(warmup):
        logits = fn(*args)
        torch.argmax(logits[:, -1, :].float(), dim=-1)
    synchronize(device)
    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        logits = fn(*args)
        torch.argmax(logits[:, -1, :].float(), dim=-1)
    end.record()
    end.synchronize()
    total_s = float(start.elapsed_time(end)) / 1000.0
    return {
        "total_s": total_s,
        "mean_ms": total_s * 1000.0 / repeats,
        "calls_per_s": repeats / total_s,
        "physical_query_tokens_per_s": (
            repeats * physical_tokens / total_s
        ),
        "effective_query_tokens_per_s": (
            repeats * effective_tokens / total_s
        ),
        "useful_fraction": effective_tokens / physical_tokens,
    }


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.npu.is_available():
        raise RuntimeError("probe requires an available Ascend NPU")
    device = torch.device("npu:0")
    dtype = torch.float16
    torch.npu.set_compile_mode(jit_compile=False)
    config = base.PaddleOCRVLConfig.from_model_dir(args.model)

    setup_started = time.perf_counter()
    model = base._create_random_model(
        config,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )
    optimization = prepare_decode_optimization_modules(
        model,
        base.OPTIMIZATION,
    )
    weight_format = cast_decode_linear_weights_to_nz(model)
    synchronize(device)

    _warm_dense, warm_paged, dummy_slot_base = (
        _allocate_matching_multi_query_caches(
            config,
            batch_size=args.batch_size,
            query_bucket=args.query_bucket,
            cache_length=args.cache_length,
            block_size=args.block_size,
            device=device,
            dtype=dtype,
            seed=args.seed + 1,
        )
    )
    stage = MultiQueryPagedFIAStage(
        model,
        batch_size=args.batch_size,
        query_bucket=args.query_bucket,
        cache_length=args.cache_length,
        attention_bucket_length=args.attention_bucket_length,
        block_size=args.block_size,
        dummy_slot_base=dummy_slot_base,
        optimization=optimization,
    ).eval()
    fn, compile_metadata = _compile_stage(
        stage,
        cache_root=args.cache_dir,
    )
    warm_input_ids = _input_ids(
        batch_size=args.batch_size,
        query_bucket=args.query_bucket,
        vocab_size=config.text_config.vocab_size,
        step=0,
        device=device,
    )
    warm_positions = torch.tensor(
        args.initial_positions,
        device=device,
        dtype=torch.int64,
    )
    warm_lengths = torch.tensor(
        args.query_patterns[0],
        device=device,
        dtype=torch.int64,
    )
    warm_rope = torch.zeros(
        (args.batch_size, 1),
        device=device,
        dtype=torch.int64,
    )
    synchronize(device)
    first_call_started = time.perf_counter()
    fn(
        warm_input_ids,
        warm_positions,
        warm_lengths,
        warm_rope,
        warm_paged.block_table,
        *warm_paged.flat_tensors(),
    )
    synchronize(device)
    compile_metadata["first_call_s"] = (
        time.perf_counter() - first_call_started
    )
    setup_s = time.perf_counter() - setup_started

    dense_rows, paged_cache, validation_dummy_slot_base = (
        _allocate_matching_multi_query_caches(
            config,
            batch_size=args.batch_size,
            query_bucket=args.query_bucket,
            cache_length=args.cache_length,
            block_size=args.block_size,
            device=device,
            dtype=dtype,
            seed=args.seed + 50_000,
        )
    )
    if validation_dummy_slot_base != dummy_slot_base:
        raise AssertionError("dummy page layout changed across allocations")
    reference_stage = TextDecodeStage(
        model,
        optimization=optimization,
    ).eval()
    rope_deltas = torch.zeros(
        (args.batch_size, 1),
        device=device,
        dtype=torch.int64,
    )
    current_positions = torch.tensor(
        args.initial_positions,
        device=device,
        dtype=torch.int64,
    )
    correctness_rows = []
    passed = True
    for step, query_pattern in enumerate(args.query_patterns):
        input_ids = _input_ids(
            batch_size=args.batch_size,
            query_bucket=args.query_bucket,
            vocab_size=config.text_config.vocab_size,
            step=step,
            device=device,
        )
        query_lengths = torch.tensor(
            query_pattern,
            device=device,
            dtype=torch.int64,
        )
        start_positions = tuple(
            int(value) for value in current_positions.cpu().tolist()
        )
        reference_logits = _run_sequential_reference(
            reference_stage,
            input_ids=input_ids,
            start_positions=current_positions,
            query_lengths=query_pattern,
            rope_deltas=rope_deltas,
            dense_rows=dense_rows,
        )
        paged_logits = fn(
            input_ids,
            current_positions,
            query_lengths,
            rope_deltas,
            paged_cache.block_table,
            *paged_cache.flat_tensors(),
        )
        synchronize(device)
        reference_tokens = torch.argmax(
            reference_logits[:, -1, :].float(),
            dim=-1,
        )
        paged_tokens = torch.argmax(
            paged_logits[:, -1, :].float(),
            dim=-1,
        )
        dense_values, paged_values = _written_cache_values(
            dense_rows=dense_rows,
            paged_cache=paged_cache,
            start_positions=start_positions,
            query_lengths=query_pattern,
        )
        logits_delta = base._delta_stats(
            paged_logits,
            reference_logits,
        )
        cache_delta = base._cache_delta_stats(
            dense_values,
            paged_values,
        )
        argmax_matches = int(
            (reference_tokens == paged_tokens).sum().cpu()
        )
        step_passed = (
            argmax_matches == args.batch_size
            and cache_delta["mean_abs"] < 1e-3
        )
        passed = passed and step_passed
        correctness_rows.append(
            {
                "step": step,
                "query_lengths": list(query_pattern),
                "start_positions": list(start_positions),
                "end_positions": [
                    start_positions[row] + query_pattern[row] - 1
                    for row in range(args.batch_size)
                ],
                "effective_query_tokens": sum(query_pattern),
                "physical_query_tokens": (
                    args.batch_size * args.query_bucket
                ),
                "argmax_matches": argmax_matches,
                "argmax_total": args.batch_size,
                "logits": logits_delta,
                "written_kv": {
                    "max_abs": cache_delta["max_abs"],
                    "mean_abs": cache_delta["mean_abs"],
                },
                "passed": step_passed,
            }
        )
        current_positions.add_(query_lengths)

    timing_pattern = args.query_patterns[0]
    timing_dense_rows, timing_paged, _timing_dummy_slot_base = (
        _allocate_matching_multi_query_caches(
            config,
            batch_size=args.batch_size,
            query_bucket=args.query_bucket,
            cache_length=args.cache_length,
            block_size=args.block_size,
            device=device,
            dtype=dtype,
            seed=args.seed + 90_000,
        )
    )
    del timing_dense_rows
    timing_input_ids = _input_ids(
        batch_size=args.batch_size,
        query_bucket=args.query_bucket,
        vocab_size=config.text_config.vocab_size,
        step=100,
        device=device,
    )
    timing_positions = torch.tensor(
        args.initial_positions,
        device=device,
        dtype=torch.int64,
    )
    timing_lengths = torch.tensor(
        timing_pattern,
        device=device,
        dtype=torch.int64,
    )
    timing = _timed_graph(
        fn,
        (
            timing_input_ids,
            timing_positions,
            timing_lengths,
            rope_deltas,
            timing_paged.block_table,
            *timing_paged.flat_tensors(),
        ),
        physical_tokens=args.batch_size * args.query_bucket,
        effective_tokens=sum(timing_pattern),
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )

    result = {
        "schema_version": 1,
        "kind": "full_decoder_paged_fia_multi_query_probe",
        "passed": passed,
        "configuration": {
            "model_config": str(args.model / "config.json"),
            "random_weights": True,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "query_bucket": args.query_bucket,
            "query_patterns": [
                list(pattern) for pattern in args.query_patterns
            ],
            "initial_positions": list(args.initial_positions),
            "cache_length": args.cache_length,
            "attention_bucket_length": (
                args.attention_bucket_length
            ),
            "block_size": args.block_size,
            "input_layout": "BNSD",
            "paged_cache_layout": "PA_NZ",
            "cache_writer": "torch_npu.npu_scatter_nd_update_",
            "runtime_query_lengths": True,
            "fixed_fia_query_bucket_lengths": (
                list(stage.fixed_actual_query_lengths)
            ),
            "fixed_fia_kv_bucket_lengths": (
                list(stage.fixed_actual_kv_lengths)
            ),
            "padding_kv_destination": (
                "scratch pages absent from every request block table"
            ),
            "attention_mask": (
                "runtime per-request causal mask over physical KV capacity"
            ),
            "optimization": base.OPTIMIZATION,
            "linear_weight_format": weight_format,
        },
        "model_parameters": base._parameter_counts(model),
        "setup_timing_s": {
            "total": setup_s,
            "compile_wrapper": compile_metadata["compile_wrapper_s"],
            "compile_first_call": compile_metadata["first_call_s"],
        },
        "compile": compile_metadata,
        "correctness": {
            "passed": passed,
            "reference": (
                "same-weight sequential one-token IncreFA with matching "
                "initial dense KV"
            ),
            "graph_reuse_patterns": len(args.query_patterns),
            "rows": correctness_rows,
        },
        "throughput": {
            "query_pattern": list(timing_pattern),
            "warmup": args.warmup,
            "repeats": args.repeats,
            **timing,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"OUTPUT_JSON={args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
