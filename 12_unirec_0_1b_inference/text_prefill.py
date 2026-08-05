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
    return hashlib.sha256(payload).hexdigest()[:12]


class UniRecTextPrefillStage(nn.Module):
    """Cross-KV projection and one-token decoder prefill at a static source S."""

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder
        self.num_layers = len(decoder.layers)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        decoder = self.decoder
        hidden_states = decoder.build_decoder_input_hidden_states(decoder_input_ids)
        self_attention_mask = decoder.build_decoder_attention_mask(decoder_input_ids)
        cross_attention_mask = decoder.build_cross_attention_mask(
            encoder_attention_mask=encoder_attention_mask,
            target_length=decoder_input_ids.shape[1],
        )
        cross_key_cache, cross_value_cache = decoder.build_cross_attention_cache(
            encoder_hidden_states
        )

        layer_keys = []
        layer_values = []
        for layer in decoder.layers:
            hidden_states, key_states, value_states = layer.forward_prefill(
                hidden_states=hidden_states,
                self_attention_mask=self_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=cross_attention_mask,
            )
            layer_keys.append(key_states)
            layer_values.append(value_states)

        return (
            decoder.layer_norm(hidden_states),
            *layer_keys,
            *layer_values,
            *cross_key_cache,
            *cross_value_cache,
        )


@dataclass(frozen=True)
class UniRecTextPrefillOutput:
    decoder_output: torch.Tensor
    layer_keys: tuple[torch.Tensor, ...]
    layer_values: tuple[torch.Tensor, ...]
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
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> UniRecTextPrefillOutput:
        if tuple(decoder_input_ids.shape) != (1, 1):
            raise ValueError(
                "compiled UniRec text prefill expects decoder_input_ids [1, 1], "
                f"got {tuple(decoder_input_ids.shape)}"
            )
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
        if tuple(encoder_attention_mask.shape) != (1, real_source_tokens):
            raise ValueError(
                "encoder_attention_mask must match the real source length, got "
                f"{tuple(encoder_attention_mask.shape)}"
            )

        pad_tokens = self.bucket - real_source_tokens
        padded_hidden_states = F.pad(
            encoder_hidden_states,
            (0, 0, 0, pad_tokens),
        ).contiguous()
        padded_attention_mask = F.pad(
            encoder_attention_mask,
            (0, pad_tokens),
            value=0,
        ).contiguous()
        if self._first_call:
            print(
                "UNIREC_TEXT_PREFILL_FIRST_CALL_BEGIN "
                f"bucket={self.bucket} cache_dir={self.cache_dir}",
                flush=True,
            )
            first_call_started = time.perf_counter()
        flat_outputs = self.compiled(
            decoder_input_ids,
            padded_hidden_states,
            padded_attention_mask,
        )
        if self._first_call:
            print(
                "UNIREC_TEXT_PREFILL_FIRST_CALL_RETURN "
                f"wall_s={time.perf_counter() - first_call_started:.3f}",
                flush=True,
            )
            self._first_call = False

        offset = 1
        layer_keys = tuple(flat_outputs[offset : offset + self.num_layers])
        offset += self.num_layers
        layer_values = tuple(flat_outputs[offset : offset + self.num_layers])
        offset += self.num_layers
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
            decoder_output=flat_outputs[0],
            layer_keys=layer_keys,
            layer_values=layer_values,
            cross_key_cache=cross_key_cache,
            cross_value_cache=cross_value_cache,
            real_source_tokens=real_source_tokens,
            physical_source_tokens=self.bucket,
        )
