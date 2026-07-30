"""Packed B=1 text-prefill graphs for multiple isolated requests.

This module is deliberately separate from text_prefill.py so changing the
packed path does not invalidate the established normal text-prefill caches.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

import torch
import torch.nn.functional as F

from .compile_utils import (
    TORCHAIR_EXECUTION_MODE,
    cache_key_part,
    import_torchair,
    short_file_hash,
    torch_npu_version_label,
    torchair_version_label,
)
from .text_decode import LocalPaddleOCRVLStaticCache
from .text_prefill import (
    TextPrefillStage,
    get_text_softmax_dtype_mode,
    parse_text_buckets,
    select_text_bucket,
    unique_bucket_forward,
)
from utils.timing import synchronize

if TYPE_CHECKING:
    from .modeling import LocalPaddleOCRVLForConditionalGeneration


PACKED_GATHER_SYNC_DIAGNOSTIC_ENV = (
    "PADDLE_OCR_VL_PACKED_GATHER_SYNC_DIAGNOSTIC"
)


def packed_text_source_hash() -> str:
    return short_file_hash(Path(__file__))


@dataclass(frozen=True)
class PreparedPackedTextPrefill:
    """One B=1 physical sequence containing isolated request segments."""

    inputs_embeds: torch.Tensor
    position_ids: torch.Tensor
    segment_ids: torch.Tensor
    local_positions: torch.Tensor
    last_token_indices: torch.Tensor
    segment_lengths: tuple[int, ...]
    segment_offsets: tuple[int, ...]
    real_seq_len: int
    physical_seq_len: int


class PackedTextPrefillStage(TextPrefillStage):
    """Compiled B=1 text prefill with block-diagonal causal attention."""

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        local_positions: torch.Tensor,
        *flat_cache_tensors: torch.Tensor,
    ) -> torch.Tensor:
        key_caches = tuple(flat_cache_tensors[: self.num_layers])
        value_caches = tuple(flat_cache_tensors[self.num_layers :])
        valid = segment_ids >= 0
        allowed = (
            valid[:, None]
            & valid[None, :]
            & (segment_ids[:, None] == segment_ids[None, :])
            & (local_positions[:, None] >= local_positions[None, :])
        )
        attention_mask = torch.zeros(
            (1, 1, inputs_embeds.shape[1], inputs_embeds.shape[1]),
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        ).masked_fill(
            ~allowed[None, None],
            torch.finfo(inputs_embeds.dtype).min,
        )
        position_embeddings = self.text_model.rotary_emb(
            inputs_embeds,
            position_ids,
        )
        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.text_model.layers):
            residual = hidden_states
            attention_input = layer.input_layernorm(hidden_states)
            attention_output = self._attention(
                layer.self_attn,
                attention_input,
                attention_mask,
                position_embeddings,
                key_caches[layer_idx],
                value_caches[layer_idx],
            )
            hidden_states = layer.apply_blocks(residual, attention_output)
        return self.text_model.norm(hidden_states)


def packed_text_cache_dir_for_bucket(
    cache_root: Path,
    *,
    bucket: int,
    max_members: int,
    dtype: torch.dtype,
    device: torch.device,
    model_dir: Path,
    linear_weight_format: str,
) -> Path:
    key = "_".join(
        [
            "text_packed_block_causal",
            f"mode{cache_key_part(TORCHAIR_EXECUTION_MODE)}",
            f"softmax{cache_key_part(get_text_softmax_dtype_mode())}",
            "bs1",
            f"seq{int(bucket)}",
            f"members{int(max_members)}",
            f"cache{int(bucket)}",
            f"weights{cache_key_part(linear_weight_format)}",
            f"dtype{cache_key_part(dtype)}",
            f"model{short_file_hash(model_dir / 'config.json')}",
            f"torch{cache_key_part(torch.__version__)}",
            f"torchnpu{torch_npu_version_label(device)}",
            f"torchair{torchair_version_label(device)}",
            f"src{packed_text_source_hash()}",
        ]
    )
    return cache_root.expanduser().resolve() / key


class PackedTextPrefillRuntime:
    """Static packed-text graphs and scratch KV redistribution."""

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        *,
        buckets: str | Iterable[int],
        max_members: int,
        cache_root: Path,
        destination_cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
        model_dir: Path,
        linear_weight_format: str,
    ):
        if device.type != "npu":
            raise ValueError("packed text prefill requires an NPU device")
        self.model = model
        self.buckets = parse_text_buckets(buckets)
        self.max_members = int(max_members)
        self.cache_root = cache_root.expanduser().resolve()
        self.destination_cache_length = int(destination_cache_length)
        self.device = device
        self.dtype = dtype
        if self.max_members <= 0:
            raise ValueError("packed text max_members must be positive")
        if self.buckets[-1] > self.destination_cache_length:
            raise ValueError(
                "largest packed text bucket exceeds destination cache length: "
                f"bucket={self.buckets[-1]} "
                f"cache={self.destination_cache_length}"
            )

        torchair, CompilerConfig = import_torchair()
        hidden_size = int(model.config.text_config.hidden_size)
        self.compiled: dict[int, Callable[..., torch.Tensor]] = {}
        self.modules: dict[int, PackedTextPrefillStage] = {}
        self.scratch_caches: dict[int, LocalPaddleOCRVLStaticCache] = {}
        per_bucket: dict[str, Any] = {}
        wrapper_total_s = 0.0
        first_call_total_s = 0.0
        for bucket in self.buckets:
            module = PackedTextPrefillStage(model).eval()
            entrypoint = unique_bucket_forward(module, bucket)
            cache_dir = packed_text_cache_dir_for_bucket(
                self.cache_root,
                bucket=bucket,
                max_members=self.max_members,
                dtype=dtype,
                device=device,
                model_dir=model_dir,
                linear_weight_format=linear_weight_format,
            )
            cache_was_warm = cache_dir.is_dir() and any(cache_dir.iterdir())
            cache_dir.mkdir(parents=True, exist_ok=True)
            synchronize(device)
            wrapper_started = time.perf_counter()
            compiled = torchair.inference.cache_compile(
                entrypoint,
                config=CompilerConfig(),
                dynamic=False,
                cache_dir=str(cache_dir),
                ge_cache=True,
            )
            synchronize(device)
            wrapper_s = time.perf_counter() - wrapper_started

            warm_embeds = torch.zeros(
                (1, bucket, hidden_size),
                device=device,
                dtype=dtype,
            )
            warm_positions = torch.zeros(
                (3, 1, bucket),
                device=device,
                dtype=torch.int64,
            )
            warm_segments = torch.zeros(
                (bucket,),
                device=device,
                dtype=torch.int64,
            )
            warm_local_positions = torch.arange(
                bucket,
                device=device,
                dtype=torch.int64,
            )
            scratch_cache = model.allocate_static_cache(
                batch_size=1,
                cache_length=bucket,
                device=device,
                dtype=dtype,
                init_mode="empty",
            )
            synchronize(device)
            first_call_started = time.perf_counter()
            warm_output = compiled(
                warm_embeds,
                warm_positions,
                warm_segments,
                warm_local_positions,
                *scratch_cache.flat_tensors(),
            )
            synchronize(device)
            first_call_s = time.perf_counter() - first_call_started
            expected_shape = (1, bucket, hidden_size)
            if tuple(warm_output.shape) != expected_shape:
                raise RuntimeError(
                    "packed text graph returned the wrong shape: "
                    f"expected={expected_shape} got={tuple(warm_output.shape)}"
                )
            self.modules[bucket] = module
            self.compiled[bucket] = compiled
            self.scratch_caches[bucket] = scratch_cache
            wrapper_total_s += wrapper_s
            first_call_total_s += first_call_s
            per_bucket[str(bucket)] = {
                "cache_dir": str(cache_dir),
                "cache_was_warm": cache_was_warm,
                "compile_wrapper_s": wrapper_s,
                "compile_first_call_s": first_call_s,
            }
            del (
                warm_output,
                warm_embeds,
                warm_positions,
                warm_segments,
                warm_local_positions,
            )
        self.metadata = {
            "enabled": True,
            "boundary": (
                "block_diagonal_text_transformer_plus_packed_scratch_kv_writes"
            ),
            "buckets": list(self.buckets),
            "max_members": self.max_members,
            "destination_cache_length": self.destination_cache_length,
            "attention": "block_diagonal_causal_manual",
            "compile_api": "torchair.inference.cache_compile",
            "dynamic": False,
            "fullgraph": True,
            "torchair_ge_cache": True,
            "compile_wrapper_total_s": wrapper_total_s,
            "compile_first_call_total_s": first_call_total_s,
            "per_bucket": per_bucket,
        }

    def route(self, segment_lengths: Iterable[int]) -> dict[str, Any]:
        lengths = tuple(int(length) for length in segment_lengths)
        if not lengths or any(length <= 0 for length in lengths):
            raise ValueError("packed text segments must have positive lengths")
        if len(lengths) > self.max_members:
            raise ValueError(
                f"packed text graph supports at most {self.max_members} members, "
                f"got {len(lengths)}"
            )
        real_seq_len = sum(lengths)
        bucket = select_text_bucket(real_seq_len, self.buckets)
        if bucket is None:
            raise ValueError(
                f"packed text sequence {real_seq_len} exceeds largest bucket "
                f"{self.buckets[-1]}"
            )
        return {
            "execution": "compiled_packed",
            "real_text_tokens": real_seq_len,
            "physical_text_tokens": bucket,
            "padding_text_tokens": bucket - real_seq_len,
            "useful_token_fraction": real_seq_len / bucket,
            "bucket": bucket,
            "pack_members": len(lengths),
            "segment_lengths": list(lengths),
        }

    def prepare(
        self,
        inputs_embeds: list[torch.Tensor],
        position_ids: list[torch.Tensor],
        *,
        route: dict[str, Any],
    ) -> PreparedPackedTextPrefill:
        if len(inputs_embeds) != len(position_ids):
            raise ValueError("packed text embeddings and positions must align")
        if not inputs_embeds or len(inputs_embeds) > self.max_members:
            raise ValueError(
                f"packed text member count must be 1..{self.max_members}"
            )
        lengths = tuple(int(tensor.shape[1]) for tensor in inputs_embeds)
        expected_lengths = tuple(int(value) for value in route["segment_lengths"])
        if lengths != expected_lengths:
            raise ValueError(
                f"packed text route lengths {expected_lengths} do not match {lengths}"
            )
        for index, (embeds, positions) in enumerate(
            zip(inputs_embeds, position_ids)
        ):
            if tuple(embeds.shape[:1]) != (1,) or embeds.ndim != 3:
                raise ValueError(
                    f"packed text member {index} embeddings have shape "
                    f"{tuple(embeds.shape)}"
                )
            if tuple(positions.shape) != (3, 1, lengths[index]):
                raise ValueError(
                    f"packed text member {index} positions have shape "
                    f"{tuple(positions.shape)}"
                )
        real_seq_len = sum(lengths)
        physical_seq_len = int(route["physical_text_tokens"])
        pad_tokens = physical_seq_len - real_seq_len
        packed_embeds = F.pad(
            torch.cat(inputs_embeds, dim=1),
            (0, 0, 0, pad_tokens),
        ).contiguous()
        packed_positions = F.pad(
            torch.cat(position_ids, dim=2),
            (0, pad_tokens),
            value=1,
        ).contiguous()
        segment_ids = torch.cat(
            [
                torch.full(
                    (length,),
                    segment,
                    device=packed_embeds.device,
                    dtype=torch.int64,
                )
                for segment, length in enumerate(lengths)
            ]
        )
        local_positions = torch.cat(
            [
                torch.arange(
                    length,
                    device=packed_embeds.device,
                    dtype=torch.int64,
                )
                for length in lengths
            ]
        )
        if pad_tokens:
            segment_ids = F.pad(segment_ids, (0, pad_tokens), value=-1)
            local_positions = F.pad(local_positions, (0, pad_tokens), value=0)
        offsets: list[int] = []
        last_indices: list[int] = []
        offset = 0
        for length in lengths:
            offsets.append(offset)
            offset += length
            last_indices.append(offset - 1)
        last_token_indices = torch.zeros(
            (self.max_members,),
            device=packed_embeds.device,
            dtype=torch.int64,
        )
        last_token_indices[: len(last_indices)] = torch.tensor(
            last_indices,
            device=packed_embeds.device,
            dtype=torch.int64,
        )
        return PreparedPackedTextPrefill(
            inputs_embeds=packed_embeds,
            position_ids=packed_positions,
            segment_ids=segment_ids.contiguous(),
            local_positions=local_positions.contiguous(),
            last_token_indices=last_token_indices,
            segment_lengths=lengths,
            segment_offsets=tuple(offsets),
            real_seq_len=real_seq_len,
            physical_seq_len=physical_seq_len,
        )

    def run_prepared(
        self,
        prepared: PreparedPackedTextPrefill,
    ) -> torch.Tensor:
        scratch_cache = self.scratch_caches[prepared.physical_seq_len]
        hidden_states = self.compiled[prepared.physical_seq_len](
            prepared.inputs_embeds,
            prepared.position_ids,
            prepared.segment_ids,
            prepared.local_positions,
            *scratch_cache.flat_tensors(),
        )
        diagnose_gather = (
            os.environ.get(PACKED_GATHER_SYNC_DIAGNOSTIC_ENV, "0") == "1"
        )
        if diagnose_gather:
            print(
                "[packed-gather-sync] compiled_graph_enqueued",
                flush=True,
            )
            try:
                synchronize(self.device)
            except BaseException as exception:
                raise RuntimeError(
                    "packed text compiled graph synchronization failed "
                    "before eager final-token gather"
                ) from exception
            print(
                "[packed-gather-sync] compiled_graph_sync_passed",
                flush=True,
            )
        gathered = torch.index_select(
            hidden_states,
            1,
            prepared.last_token_indices,
        )
        if diagnose_gather:
            print(
                "[packed-gather-sync] eager_gather_enqueued",
                flush=True,
            )
            try:
                synchronize(self.device)
            except BaseException as exception:
                raise RuntimeError(
                    "eager final-token gather synchronization failed "
                    "after compiled graph completed"
                ) from exception
            print(
                "[packed-gather-sync] eager_gather_sync_passed",
                flush=True,
            )
        return gathered

    def redistribute_cache(
        self,
        prepared: PreparedPackedTextPrefill,
        destinations: list[LocalPaddleOCRVLStaticCache],
    ) -> int:
        if len(destinations) != len(prepared.segment_lengths):
            raise ValueError("packed text cache destinations do not align")
        scratch = self.scratch_caches[prepared.physical_seq_len]
        copied_bytes = 0
        for destination, offset, length in zip(
            destinations,
            prepared.segment_offsets,
            prepared.segment_lengths,
        ):
            for source_tensor, destination_tensor in zip(
                scratch.flat_tensors(),
                destination.flat_tensors(),
            ):
                source = source_tensor[:, :, offset : offset + length, :]
                destination_tensor[:, :, :length, :].copy_(source)
                copied_bytes += source.numel() * source.element_size()
        return copied_bytes
