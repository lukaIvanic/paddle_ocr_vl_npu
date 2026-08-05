#!/usr/bin/env python3
"""Packed, bucketed TorchAir text prefill for the local MinerU model.

Multiple independent prompts are concatenated into one physical B=1 sequence.
A block-diagonal causal mask keeps the requests isolated.  The compiled graph
writes a scratch KV cache; only each request's valid prefix is copied into its
decode-arena slot after the graph completes.
"""

from __future__ import annotations

import hashlib
import importlib
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from local_modeling_mineru import (
    LocalMinerUStaticCache,
    linear_last_dim,
    repeat_kv,
)


DEFAULT_TEXT_PREFILL_BUCKETS = (128, 256, 512, 1024)


def parse_text_prefill_buckets(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        buckets = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        buckets = tuple(int(item) for item in value)
    if not buckets or any(bucket <= 0 for bucket in buckets):
        raise ValueError("text-prefill buckets must contain positive integers")
    if tuple(sorted(set(buckets))) != buckets:
        raise ValueError("text-prefill buckets must be unique and increasing")
    return buckets


def select_text_prefill_bucket(real_tokens: int, buckets: Sequence[int]) -> int | None:
    return next((bucket for bucket in buckets if real_tokens <= bucket), None)


def _sync(device: torch.device) -> None:
    if device.type == "npu":
        import torch_npu

        torch_npu.npu.synchronize()


def _import_torchair():
    try:
        import torchair

        CompilerConfig = torchair.CompilerConfig
    except Exception as direct_error:
        try:
            from torch_npu.dynamo import torchair
            from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig
        except Exception as fallback_error:
            raise RuntimeError(
                "TorchAir is unavailable: direct import failed with "
                f"{direct_error!r}; fallback failed with {fallback_error!r}"
            ) from fallback_error
    if not hasattr(torchair, "inference"):
        torchair.inference = importlib.import_module(f"{torchair.__name__}.inference")
    return torchair, CompilerConfig


def _short_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _unique_bucket_forward(module: nn.Module, bucket: int) -> Callable[..., torch.Tensor]:
    original = module.forward.__func__
    name = f"mineru_packed_text_prefill_bucket_{int(bucket)}"
    function = types.FunctionType(
        original.__code__.replace(co_name=name),
        original.__globals__,
        name,
        original.__defaults__,
        original.__closure__,
    )
    function.__annotations__ = dict(original.__annotations__)
    function.__kwdefaults__ = original.__kwdefaults__
    return types.MethodType(function, module)


@dataclass(frozen=True)
class PreparedTextMember:
    inputs_embeds: torch.Tensor
    position_ids: torch.Tensor
    rope_delta: torch.Tensor
    sequence_length: int
    raw_vision_tokens: int
    merged_vision_tokens: int


@dataclass(frozen=True)
class PreparedPackedTextPrefill:
    inputs_embeds: torch.Tensor
    position_ids: torch.Tensor
    segment_ids: torch.Tensor
    local_positions: torch.Tensor
    lengths: tuple[int, ...]
    offsets: tuple[int, ...]
    real_tokens: int
    physical_tokens: int


class PackedMinerUTextPrefillStage(nn.Module):
    """Compiler-safe packed text transformer with scratch KV writes."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.text_model = model.model
        self.num_layers = len(self.text_model.layers)

    @staticmethod
    def _attention(
        attention: Any,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        batch, sequence_length, _hidden = hidden_states.shape
        query_states, key_states, value_states = attention.project_qkv(hidden_states)
        query_states, key_states = attention.apply_rotary(
            query_states,
            key_states,
            position_embeddings,
        )
        key_cache.copy_(key_states.contiguous())
        value_cache.copy_(value_states.contiguous())

        key_for_attention = repeat_kv(key_states, attention.num_key_value_groups)
        value_for_attention = repeat_kv(value_states, attention.num_key_value_groups)
        flat_batch = batch * attention.num_heads
        flat_query = query_states.reshape(flat_batch, sequence_length, attention.head_dim)
        flat_key = key_for_attention.reshape(flat_batch, sequence_length, attention.head_dim)
        scores = torch.bmm(flat_query, flat_key.transpose(1, 2)).reshape(
            batch,
            attention.num_heads,
            sequence_length,
            sequence_length,
        ) * attention.scaling
        scores = scores + attention_mask
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        flat_probabilities = probabilities.reshape(
            flat_batch,
            sequence_length,
            sequence_length,
        )
        flat_value = value_for_attention.reshape(
            flat_batch,
            sequence_length,
            attention.head_dim,
        )
        output = torch.bmm(flat_probabilities, flat_value).reshape(
            batch,
            attention.num_heads,
            sequence_length,
            attention.head_dim,
        )
        output = output.transpose(1, 2).contiguous().reshape(batch, sequence_length, -1)
        return linear_last_dim(attention.o_proj, output)

    @staticmethod
    def _apply_blocks(layer: Any, residual: torch.Tensor, attention_output: torch.Tensor) -> torch.Tensor:
        return layer.apply_blocks(residual, attention_output)

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
        ).masked_fill(~allowed[None, None], torch.finfo(inputs_embeds.dtype).min)
        position_embeddings = self.text_model.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer_index, layer in enumerate(self.text_model.layers):
            residual = hidden_states
            attention_input = layer.input_layernorm(hidden_states)
            attention_output = self._attention(
                layer.self_attn,
                attention_input,
                attention_mask,
                position_embeddings,
                key_caches[layer_index],
                value_caches[layer_index],
            )
            hidden_states = self._apply_blocks(layer, residual, attention_output)
        return self.text_model.norm(hidden_states)


class MinerUPackedTextPrefillRuntime:
    """Pack prompts, run static graphs, and redistribute valid KV prefixes."""

    def __init__(
        self,
        model: Any,
        *,
        buckets: str | Iterable[int] = DEFAULT_TEXT_PREFILL_BUCKETS,
        max_members: int = 32,
        cache_root: Path,
        model_dir: Path,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if device.type != "npu":
            raise ValueError("compiled packed text prefill requires an NPU")
        self.model = model
        self.buckets = parse_text_prefill_buckets(buckets)
        self.max_members = int(max_members)
        self.cache_root = cache_root.expanduser().resolve()
        self.model_dir = model_dir.expanduser().resolve()
        self.device = device
        self.dtype = dtype
        self.compiled: dict[int, Callable[..., torch.Tensor]] = {}
        self.modules: dict[int, PackedMinerUTextPrefillStage] = {}
        self.scratch_caches: dict[int, LocalMinerUStaticCache] = {}
        self.compile_records: dict[str, dict[str, Any]] = {}
        self.route_counts: dict[str, int] = {}
        self.real_tokens = 0
        self.physical_tokens = 0
        self.cache_copy_bytes = 0
        self.pack_count = 0
        if self.max_members <= 0:
            raise ValueError("max_members must be positive")

    def _cache_dir(self, bucket: int) -> Path:
        source = Path(__file__).resolve()
        config = self.model_dir / "config.json"
        try:
            import torch_npu

            torch_npu_version = str(torch_npu.__version__)
        except Exception:
            torch_npu_version = "unknown"
        key = "_".join(
            (
                "mineru_text_packed_block_causal_stock_projections",
                "bs1",
                f"seq{bucket}",
                f"members{self.max_members}",
                f"dtype{str(self.dtype).replace('torch.', '')}",
                f"model{_short_hash(config)}",
                f"torch{torch.__version__}",
                f"torchnpu{torch_npu_version}",
                f"src{_short_hash(source)}",
            )
        )
        return self.cache_root / key.replace("/", "_")

    def _compiled_for_bucket(self, bucket: int) -> Callable[..., torch.Tensor]:
        if bucket in self.compiled:
            return self.compiled[bucket]
        torchair, CompilerConfig = _import_torchair()
        module = PackedMinerUTextPrefillStage(self.model).eval()
        entrypoint = _unique_bucket_forward(module, bucket)
        cache_dir = self._cache_dir(bucket)
        cache_was_warm = cache_dir.is_dir() and any(cache_dir.iterdir())
        cache_dir.mkdir(parents=True, exist_ok=True)
        _sync(self.device)
        started = time.perf_counter()
        compiled = torchair.inference.cache_compile(
            entrypoint,
            config=CompilerConfig(),
            dynamic=False,
            cache_dir=str(cache_dir),
            ge_cache=True,
            fullgraph=True,
        )
        _sync(self.device)
        scratch = self.model.allocate_static_cache(
            batch_size=1,
            cache_length=bucket,
            device=self.device,
            dtype=self.dtype,
            init_mode="empty",
        )
        self.modules[bucket] = module
        self.compiled[bucket] = compiled
        self.scratch_caches[bucket] = scratch
        self.compile_records[str(bucket)] = {
            "cache_dir": str(cache_dir),
            "cache_was_warm": cache_was_warm,
            "compile_wrapper_s": float(time.perf_counter() - started),
            "first_call_s": None,
        }
        return compiled

    def pack_indices(self, lengths: Sequence[int]) -> tuple[list[list[int]], list[int]]:
        """Best-fit decreasing into the largest supported physical bucket."""
        maximum = self.buckets[-1]
        overflow = [index for index, length in enumerate(lengths) if length > maximum]
        eligible = sorted(
            (index for index, length in enumerate(lengths) if length <= maximum),
            key=lambda index: (-int(lengths[index]), index),
        )
        packs: list[list[int]] = []
        totals: list[int] = []
        for index in eligible:
            length = int(lengths[index])
            candidates = [
                pack_index
                for pack_index, total in enumerate(totals)
                if len(packs[pack_index]) < self.max_members and total + length <= maximum
            ]
            if candidates:
                target = min(candidates, key=lambda pack_index: maximum - totals[pack_index] - length)
                packs[target].append(index)
                totals[target] += length
            else:
                packs.append([index])
                totals.append(length)
        return packs, overflow

    def prepare(self, members: Sequence[PreparedTextMember]) -> PreparedPackedTextPrefill:
        lengths = tuple(int(member.sequence_length) for member in members)
        real_tokens = sum(lengths)
        bucket = select_text_prefill_bucket(real_tokens, self.buckets)
        if bucket is None:
            raise ValueError(f"packed prompt length {real_tokens} exceeds {self.buckets[-1]}")
        if len(members) > self.max_members:
            raise ValueError("packed prompt member count exceeds max_members")
        padding = bucket - real_tokens
        inputs_embeds = F.pad(
            torch.cat([member.inputs_embeds for member in members], dim=1),
            (0, 0, 0, padding),
        ).contiguous()
        position_ids = F.pad(
            torch.cat([member.position_ids for member in members], dim=2),
            (0, padding),
            value=1,
        ).contiguous()
        segment_ids = torch.cat(
            [
                torch.full(
                    (length,),
                    segment,
                    device=self.device,
                    dtype=torch.int64,
                )
                for segment, length in enumerate(lengths)
            ]
        )
        local_positions = torch.cat(
            [torch.arange(length, device=self.device, dtype=torch.int64) for length in lengths]
        )
        if padding:
            segment_ids = F.pad(segment_ids, (0, padding), value=-1)
            local_positions = F.pad(local_positions, (0, padding), value=0)
        offsets: list[int] = []
        offset = 0
        for length in lengths:
            offsets.append(offset)
            offset += length
        return PreparedPackedTextPrefill(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            segment_ids=segment_ids.contiguous(),
            local_positions=local_positions.contiguous(),
            lengths=lengths,
            offsets=tuple(offsets),
            real_tokens=real_tokens,
            physical_tokens=bucket,
        )

    def run_prepared(self, prepared: PreparedPackedTextPrefill) -> torch.Tensor:
        compiled = self._compiled_for_bucket(prepared.physical_tokens)
        scratch = self.scratch_caches[prepared.physical_tokens]
        record = self.compile_records[str(prepared.physical_tokens)]
        first_call = record["first_call_s"] is None
        if first_call:
            _sync(self.device)
            started = time.perf_counter()
        hidden_states = compiled(
            prepared.inputs_embeds,
            prepared.position_ids,
            prepared.segment_ids,
            prepared.local_positions,
            *scratch.flat_tensors(),
        )
        if first_call:
            _sync(self.device)
            record["first_call_s"] = float(time.perf_counter() - started)
        self.pack_count += 1
        self.real_tokens += prepared.real_tokens
        self.physical_tokens += prepared.physical_tokens
        key = str(prepared.physical_tokens)
        self.route_counts[key] = self.route_counts.get(key, 0) + 1
        return torch.cat(
            [
                hidden_states[:, offset + length - 1 : offset + length, :]
                for offset, length in zip(prepared.offsets, prepared.lengths)
            ],
            dim=1,
        )

    def redistribute_cache(
        self,
        prepared: PreparedPackedTextPrefill,
        destinations: Sequence[LocalMinerUStaticCache],
    ) -> int:
        if len(destinations) != len(prepared.lengths):
            raise ValueError("packed cache destinations do not match members")
        scratch = self.scratch_caches[prepared.physical_tokens]
        copied = 0
        for destination, offset, length in zip(
            destinations,
            prepared.offsets,
            prepared.lengths,
        ):
            for source, target in zip(scratch.flat_tensors(), destination.flat_tensors()):
                prefix = source[:, :, offset : offset + length, :]
                target[:, :, :length, :].copy_(prefix)
                copied += prefix.numel() * prefix.element_size()
        self.cache_copy_bytes += copied
        return copied

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "boundary": "packed_block_diagonal_text_transformer",
            "buckets": list(self.buckets),
            "max_members": self.max_members,
            "packed_qkv": False,
            "packed_gate_up": False,
            "attention": "manual_block_diagonal_causal",
            "route_counts": dict(self.route_counts),
            "pack_count": self.pack_count,
            "real_tokens": self.real_tokens,
            "physical_tokens": self.physical_tokens,
            "useful_token_fraction": (
                self.real_tokens / self.physical_tokens if self.physical_tokens else None
            ),
            "cache_copy_bytes": self.cache_copy_bytes,
            "compile_records": dict(self.compile_records),
        }
