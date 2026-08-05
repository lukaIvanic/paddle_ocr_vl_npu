"""Static B1 S1024 packed cross-KV projection for UniRec."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


PACKED_TEXT_PREFILL_BUCKET = 1024


def _import_cache_compile() -> tuple[Any, str]:
    try:
        from torch_npu.dynamo.torchair.inference import cache_compile

        return cache_compile, "torch_npu.dynamo.torchair.inference.cache_compile"
    except Exception as first_exc:
        try:
            from torchair.inference import cache_compile

            return cache_compile, "torchair.inference.cache_compile"
        except Exception as second_exc:
            raise RuntimeError(
                "Failed to import TorchAir cache_compile for packed UniRec text "
                f"prefill. first_error={first_exc!r}; "
                f"second_error={second_exc!r}"
            ) from second_exc


def _source_hash() -> str:
    payload = Path(__file__).read_bytes()
    payload += Path(__file__).with_name("modeling_optimized_unirec.py").read_bytes()
    return hashlib.sha256(payload).hexdigest()[:12]


class PackedUniRecTextPrefillStage(nn.Module):
    """Project one concatenated physical sequence into decoder cross K/V."""

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        packed_encoder_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        cross_keys, cross_values = self.decoder.build_cross_attention_cache(
            packed_encoder_hidden_states
        )
        return (*cross_keys, *cross_values)


@dataclass(frozen=True)
class PackedUniRecTextPrefillOutput:
    cross_key_cache: tuple[tuple[torch.Tensor, ...], ...]
    cross_value_cache: tuple[tuple[torch.Tensor, ...], ...]
    segment_lengths: tuple[int, ...]
    real_source_tokens: int
    physical_source_tokens: int


class PackedUniRecTextPrefillRuntime:
    """Greedy-pack-ready S1024 graph plus eager segment redistribution."""

    def __init__(
        self,
        decoder: nn.Module,
        *,
        cache_root: Path,
        dtype_name: str,
    ) -> None:
        self.stage = PackedUniRecTextPrefillStage(decoder).eval()
        self.num_layers = len(decoder.layers)
        self.bucket = PACKED_TEXT_PREFILL_BUCKET
        self.cache_dir = (
            cache_root.expanduser().resolve()
            / (
                f"text_prefill_packed_b1_s{self.bucket}_{dtype_name}_"
                f"src{_source_hash()}"
            )
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

        config = CompilerConfig()
        config.mode.value = "max-autotune"
        cache_compile, import_path = _import_cache_compile()
        self.compiled = cache_compile(
            self.stage.forward,
            config=config,
            dynamic=False,
            cache_dir=str(self.cache_dir),
            ge_cache=True,
            fullgraph=True,
        )
        self.metadata = {
            "execution": "compiled_packed_s1024",
            "batch_size": 1,
            "source_bucket": self.bucket,
            "dynamic": False,
            "fullgraph": True,
            "compile_api": import_path,
            "torchair_cache_dir": str(self.cache_dir),
        }
        self._first_call = True

    def run(
        self,
        *,
        encoder_hidden_states: list[torch.Tensor],
    ) -> PackedUniRecTextPrefillOutput:
        if not encoder_hidden_states:
            raise ValueError("packed UniRec text prefill requires at least one segment")
        hidden_size = int(encoder_hidden_states[0].shape[-1])
        lengths = []
        for index, tensor in enumerate(encoder_hidden_states):
            if tensor.ndim != 3 or tensor.shape[0] != 1:
                raise ValueError(
                    "packed UniRec encoder states must each have shape [1, S, H], "
                    f"segment={index} shape={tuple(tensor.shape)}"
                )
            if int(tensor.shape[-1]) != hidden_size:
                raise ValueError("packed UniRec encoder hidden sizes must match")
            lengths.append(int(tensor.shape[1]))
        segment_lengths = tuple(lengths)
        real_source_tokens = sum(segment_lengths)
        if real_source_tokens > self.bucket:
            raise ValueError(
                f"packed source length {real_source_tokens} exceeds {self.bucket}"
            )

        packed = torch.cat(encoder_hidden_states, dim=1)
        packed = F.pad(
            packed,
            (0, 0, 0, self.bucket - real_source_tokens),
        ).contiguous()
        if self._first_call:
            print(
                "UNIREC_PACKED_TEXT_PREFILL_FIRST_CALL_BEGIN "
                f"bucket={self.bucket} cache_dir={self.cache_dir}",
                flush=True,
            )
            first_call_started = time.perf_counter()
        flat_outputs = self.compiled(packed)
        if self._first_call:
            print(
                "UNIREC_PACKED_TEXT_PREFILL_FIRST_CALL_RETURN "
                f"wall_s={time.perf_counter() - first_call_started:.3f}",
                flush=True,
            )
            self._first_call = False

        padded_cross_keys = tuple(flat_outputs[: self.num_layers])
        padded_cross_values = tuple(flat_outputs[self.num_layers :])
        key_segments: list[list[torch.Tensor]] = [
            [] for _ in encoder_hidden_states
        ]
        value_segments: list[list[torch.Tensor]] = [
            [] for _ in encoder_hidden_states
        ]
        offset = 0
        for member, length in enumerate(segment_lengths):
            end = offset + length
            for tensor in padded_cross_keys:
                key_segments[member].append(
                    tensor[:, :, offset:end, :].contiguous()
                )
            for tensor in padded_cross_values:
                value_segments[member].append(
                    tensor[:, :, offset:end, :].contiguous()
                )
            offset = end

        return PackedUniRecTextPrefillOutput(
            cross_key_cache=tuple(tuple(group) for group in key_segments),
            cross_value_cache=tuple(tuple(group) for group in value_segments),
            segment_lengths=segment_lengths,
            real_source_tokens=real_source_tokens,
            physical_source_tokens=self.bucket,
        )
