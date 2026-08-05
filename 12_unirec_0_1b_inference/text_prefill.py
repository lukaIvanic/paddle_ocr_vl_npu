"""Static B1 text-prefill graph for UniRec encoder outputs."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


TEXT_PREFILL_BUCKET = 512


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
                "Failed to import TorchAir cache_compile for UniRec text prefill. "
                f"first_error={first_exc!r}; second_error={second_exc!r}"
            ) from second_exc


def _source_hash() -> str:
    payload = Path(__file__).read_bytes()
    payload += Path(__file__).with_name("modeling_optimized_unirec.py").read_bytes()
    return hashlib.sha256(payload).hexdigest()[:12]


class UniRecTextPrefillStage(nn.Module):
    """Cross-KV projection at one static encoder-source length."""

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder
        self.num_layers = len(decoder.layers)

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        cross_key_cache, cross_value_cache = self.decoder.build_cross_attention_cache(
            encoder_hidden_states
        )
        return (*cross_key_cache, *cross_value_cache)


@dataclass(frozen=True)
class UniRecTextPrefillOutput:
    cross_key_cache: tuple[torch.Tensor, ...]
    cross_value_cache: tuple[torch.Tensor, ...]
    real_source_tokens: int
    physical_source_tokens: int


class UniRecTextPrefillRuntime:
    """Pad to S=512, replay one graph, then remove source padding eagerly."""

    def __init__(
        self,
        decoder: nn.Module,
        *,
        cache_root: Path,
        dtype_name: str,
    ) -> None:
        self.stage = UniRecTextPrefillStage(decoder).eval()
        self.num_layers = len(decoder.layers)
        self.bucket = TEXT_PREFILL_BUCKET
        self.cache_dir = (
            cache_root.expanduser().resolve()
            / (
                f"text_prefill_b1_s{self.bucket}_{dtype_name}_"
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
            "execution": "compiled_s512",
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
        encoder_hidden_states: torch.Tensor,
    ) -> UniRecTextPrefillOutput:
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != 1:
            raise ValueError(
                "compiled UniRec text prefill expects encoder states [1, S, H], "
                f"got {tuple(encoder_hidden_states.shape)}"
            )
        real_source_tokens = int(encoder_hidden_states.shape[1])
        if real_source_tokens > self.bucket:
            raise ValueError(
                f"encoder source length {real_source_tokens} exceeds "
                f"compiled bucket {self.bucket}"
            )
        pad_tokens = self.bucket - real_source_tokens
        padded_hidden_states = F.pad(
            encoder_hidden_states,
            (0, 0, 0, pad_tokens),
        ).contiguous()
        if self._first_call:
            print(
                "UNIREC_TEXT_PREFILL_FIRST_CALL_BEGIN "
                f"bucket={self.bucket} cache_dir={self.cache_dir}",
                flush=True,
            )
            first_call_started = time.perf_counter()
        flat_outputs = self.compiled(padded_hidden_states)
        if self._first_call:
            print(
                "UNIREC_TEXT_PREFILL_FIRST_CALL_RETURN "
                f"wall_s={time.perf_counter() - first_call_started:.3f}",
                flush=True,
            )
            self._first_call = False

        offset = 0
        padded_cross_keys = tuple(
            flat_outputs[offset : offset + self.num_layers]
        )
        offset += self.num_layers
        padded_cross_values = tuple(
            flat_outputs[offset : offset + self.num_layers]
        )

        # The graph sees one static source shape. Only the real prefix crosses
        # back into the normal cache construction and continuous decode path.
        cross_key_cache = tuple(
            tensor[:, :, :real_source_tokens, :].contiguous()
            for tensor in padded_cross_keys
        )
        cross_value_cache = tuple(
            tensor[:, :, :real_source_tokens, :].contiguous()
            for tensor in padded_cross_values
        )
        return UniRecTextPrefillOutput(
            cross_key_cache=cross_key_cache,
            cross_value_cache=cross_value_cache,
            real_source_tokens=real_source_tokens,
            physical_source_tokens=self.bucket,
        )
