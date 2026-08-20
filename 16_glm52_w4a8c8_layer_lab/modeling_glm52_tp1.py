#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from modeling_glm52_dense_tp import (
    GLM52DenseTPDecoderLayer,
    GLM52DenseTPMLP,
)
from modeling_glm52_layer import (
    BF16Linear,
    GLM52Config,
    GLM52SharedExpert,
    GLM52W4A8Experts,
    ShardedSafetensorReader,
    W8A8DynamicLinear,
)
from modeling_glm52_stack import GLM52DSAIndexer


class GLM52MoETP1MLP(nn.Module):
    def __init__(
        self,
        routed: GLM52W4A8Experts,
        shared: GLM52SharedExpert,
    ):
        super().__init__()
        self.routed = routed
        self.shared = shared

    @classmethod
    def from_checkpoint(
        cls,
        reader: ShardedSafetensorReader,
        config: GLM52Config,
        layer_index: int,
        *,
        device: torch.device,
        progress=None,
        w4_weight_format: str = "fractal_nz",
    ) -> "GLM52MoETP1MLP":
        return cls(
            GLM52W4A8Experts.from_checkpoint(
                reader,
                layer_index,
                config,
                device=device,
                progress=progress,
                w4_weight_format=w4_weight_format,
            ),
            GLM52SharedExpert.from_checkpoint(
                reader,
                layer_index,
                device=device,
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.routed(hidden_states) + self.shared(hidden_states)


def expected_indexer_type(config: GLM52Config, layer_index: int) -> str:
    return "full" if config.layer_uses_dsa_topk(layer_index) else "shared"


class GLM52OptimizedTP1Stack(nn.Module):
    """Optimized contiguous TP1 decoder layers with DSA top-k reuse."""

    def __init__(
        self,
        layers: list[GLM52DenseTPDecoderLayer],
        *,
        first_layer: int,
        last_layer: int,
        cache_length: int,
    ):
        super().__init__()
        if not layers:
            raise ValueError("layers must not be empty")
        self.layers = nn.ModuleList(layers)
        self.first_layer = int(first_layer)
        self.last_layer = int(last_layer)
        self.cache_length = int(cache_length)
        self.config = layers[0].config
        self.top_k = min(2048, self.cache_length)

    @classmethod
    def from_checkpoint(
        cls,
        model_dir: str | Path,
        *,
        first_layer: int,
        last_layer: int,
        cache_length: int,
        device: torch.device,
        progress=None,
        w4_weight_format: str = "fractal_nz",
    ) -> "GLM52OptimizedTP1Stack":
        if first_layer < 0 or last_layer < first_layer:
            raise ValueError("invalid layer range")
        config = GLM52Config.from_model_dir(model_dir)
        if last_layer >= config.num_hidden_layers:
            raise ValueError("layer range exceeds the model")
        with (Path(model_dir) / "config.json").open() as handle:
            raw = json.load(handle)
        indexer_types = list(raw["indexer_types"])
        if len(indexer_types) != config.num_hidden_layers:
            raise ValueError("indexer_types length does not match layer count")

        reader = ShardedSafetensorReader(model_dir)
        layers = []
        for layer_index in range(first_layer, last_layer + 1):
            if progress is not None:
                progress(f"loading optimized TP1 layer {layer_index}")
            expected = expected_indexer_type(config, layer_index)
            actual = str(indexer_types[layer_index])
            if actual != expected:
                raise ValueError(
                    f"layer {layer_index} indexer type {actual!r} != {expected!r}"
                )
            attn = f"model.layers.{layer_index}.self_attn"
            layer = f"model.layers.{layer_index}"
            if layer_index < config.first_k_dense_replace:
                mlp: nn.Module = GLM52DenseTPMLP.from_checkpoint(
                    reader,
                    layer_index,
                    rank=0,
                    world_size=1,
                    device=device,
                )
            else:
                mlp = GLM52MoETP1MLP.from_checkpoint(
                    reader,
                    config,
                    layer_index,
                    device=device,
                    progress=(
                        None
                        if progress is None
                        else lambda message, index=layer_index: progress(
                            f"layer {index}: {message}"
                        )
                    ),
                    w4_weight_format=w4_weight_format,
                )
            indexer = None
            if actual == "full":
                indexer = GLM52DSAIndexer.from_checkpoint(
                    reader,
                    layer_index,
                    num_heads=int(raw["index_n_heads"]),
                    head_dim=int(raw["index_head_dim"]),
                    rope_dim=config.qk_rope_head_dim,
                    top_k=int(raw["index_topk"]),
                    cache_length=cache_length,
                    device=device,
                    rope_path="interleave",
                )
            layers.append(
                GLM52DenseTPDecoderLayer(
                    layer_index=layer_index,
                    config=config,
                    cache_length=cache_length,
                    rank=0,
                    world_size=1,
                    fused_qkv_a=W8A8DynamicLinear.from_checkpoint(
                        reader,
                        [attn + ".q_a_proj", attn + ".kv_a_proj_with_mqa"],
                        device=device,
                    ),
                    q_b_proj=W8A8DynamicLinear.from_checkpoint(
                        reader,
                        [attn + ".q_b_proj"],
                        device=device,
                    ),
                    kv_b_proj=BF16Linear.from_checkpoint(
                        reader,
                        attn + ".kv_b_proj",
                        device=device,
                    ),
                    o_proj=W8A8DynamicLinear.from_checkpoint(
                        reader,
                        [attn + ".o_proj"],
                        device=device,
                    ),
                    mlp=mlp,
                    indexer=indexer,
                    input_norm=reader.tensor(layer + ".input_layernorm.weight").to(
                        device=device,
                        dtype=torch.bfloat16,
                    ),
                    post_attention_norm=reader.tensor(
                        layer + ".post_attention_layernorm.weight"
                    ).to(device=device, dtype=torch.bfloat16),
                    q_a_norm=reader.tensor(attn + ".q_a_layernorm.weight").to(
                        device=device,
                        dtype=torch.bfloat16,
                    ),
                    kv_a_norm=reader.tensor(attn + ".kv_a_layernorm.weight").to(
                        device=device,
                        dtype=torch.bfloat16,
                    ),
                )
            )
            if progress is not None:
                progress(f"loaded optimized TP1 layer {layer_index}")
        return cls(
            layers,
            first_layer=first_layer,
            last_layer=last_layer,
            cache_length=cache_length,
        )

    @property
    def full_indexer_layers(self) -> list[int]:
        return [
            layer.layer_index for layer in self.layers if layer.indexer is not None
        ]

    @property
    def shared_indexer_layers(self) -> list[int]:
        return [
            layer.layer_index for layer in self.layers if layer.indexer is None
        ]

    def initial_topk(self, *, device: torch.device) -> torch.Tensor:
        return torch.arange(self.top_k, device=device, dtype=torch.int64)

    def make_cache(self, *, device: torch.device):
        primary_shape = (1, self.cache_length, self.config.kv_lora_rank)
        secondary_shape = (1, self.cache_length, self.config.qk_rope_head_dim)
        primary = tuple(
            torch.zeros(primary_shape, dtype=torch.bfloat16, device=device)
            for _ in self.layers
        )
        secondary = tuple(
            torch.zeros(secondary_shape, dtype=torch.bfloat16, device=device)
            for _ in self.layers
        )
        indices = tuple(
            torch.zeros(
                (1, self.cache_length, 128),
                dtype=torch.bfloat16,
                device=device,
            )
            if layer.indexer is not None
            else torch.zeros((1, 1, 1), dtype=torch.bfloat16, device=device)
            for layer in self.layers
        )
        return primary, secondary, indices

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        cache_position: torch.Tensor,
        primary_caches: tuple[torch.Tensor, ...],
        secondary_caches: tuple[torch.Tensor, ...],
        index_caches: tuple[torch.Tensor, ...],
        shared_topk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for index, layer in enumerate(self.layers):
            hidden_states, shared_topk = layer.forward_decode_with_topk(
                hidden_states,
                cache_position,
                primary_caches[index],
                secondary_caches[index],
                index_caches[index],
                shared_topk,
            )
        return hidden_states, shared_topk
