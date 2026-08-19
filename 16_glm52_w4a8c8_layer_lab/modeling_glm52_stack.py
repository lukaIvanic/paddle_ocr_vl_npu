#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from modeling_glm52_layer import (
    BF16Linear,
    GLM52Config,
    GLM52SharedExpert,
    GLM52W4A8Experts,
    ShardedSafetensorReader,
    W8A8DynamicLinear,
    apply_interleaved_rope,
    npu_rms_norm,
    require_npu,
    torch_npu,
)


class GLM52DenseMLP(nn.Module):
    def __init__(self, gate_up: W8A8DynamicLinear, down: W8A8DynamicLinear):
        super().__init__()
        self.gate_up = gate_up
        self.down = down

    @classmethod
    def from_checkpoint(
        cls,
        reader: ShardedSafetensorReader,
        layer_index: int,
        *,
        device: torch.device,
    ) -> "GLM52DenseMLP":
        prefix = f"model.layers.{layer_index}.mlp"
        return cls(
            W8A8DynamicLinear.from_checkpoint(
                reader,
                [prefix + ".gate_proj", prefix + ".up_proj"],
                device=device,
            ),
            W8A8DynamicLinear.from_checkpoint(
                reader, [prefix + ".down_proj"], device=device
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch_npu.npu_swiglu(self.gate_up(x)))


class GLM52DSAIndexer(nn.Module):
    """Owned BF16/W8 indexer used by GLM-5.2 full-index layers."""

    def __init__(
        self,
        *,
        wq_b: W8A8DynamicLinear,
        wk_weight: torch.Tensor,
        weights_proj_weight: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        num_heads: int,
        head_dim: int,
        rope_dim: int,
        top_k: int,
        cache_length: int,
    ):
        super().__init__()
        self.wq_b = wq_b
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rope_dim = int(rope_dim)
        self.top_k = min(int(top_k), int(cache_length))
        self.cache_length = int(cache_length)
        self.register_buffer("wk_weight", wk_weight.contiguous())
        self.register_buffer(
            "weights_proj_weight", weights_proj_weight.contiguous()
        )
        self.register_buffer("norm_weight", norm_weight.contiguous())
        self.register_buffer("norm_bias", norm_bias.contiguous())

    @classmethod
    def from_checkpoint(
        cls,
        reader: ShardedSafetensorReader,
        layer_index: int,
        *,
        num_heads: int,
        head_dim: int,
        rope_dim: int,
        top_k: int,
        cache_length: int,
        device: torch.device,
    ) -> "GLM52DSAIndexer":
        prefix = f"model.layers.{layer_index}.self_attn.indexer"
        return cls(
            wq_b=W8A8DynamicLinear.from_checkpoint(
                reader, [prefix + ".wq_b"], device=device
            ),
            wk_weight=reader.tensor(prefix + ".wk.weight").to(
                device=device, dtype=torch.bfloat16
            ),
            weights_proj_weight=reader.tensor(
                prefix + ".weights_proj.weight"
            ).to(device=device, dtype=torch.bfloat16),
            norm_weight=reader.tensor(prefix + ".k_norm.weight").to(
                device=device, dtype=torch.bfloat16
            ),
            norm_bias=reader.tensor(prefix + ".k_norm.bias").to(
                device=device, dtype=torch.bfloat16
            ),
            num_heads=num_heads,
            head_dim=head_dim,
            rope_dim=rope_dim,
            top_k=top_k,
            cache_length=cache_length,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        position: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
    ) -> torch.Tensor:
        q = self.wq_b(q_lora).view(1, self.num_heads, self.head_dim)
        k = F.linear(
            hidden_states.reshape(-1, hidden_states.shape[-1]), self.wk_weight
        )
        k = F.layer_norm(
            k,
            (self.head_dim,),
            self.norm_weight,
            self.norm_bias,
            1e-6,
        )
        weights = F.linear(
            hidden_states.reshape(-1, hidden_states.shape[-1]),
            self.weights_proj_weight,
        )

        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        q_pe = apply_interleaved_rope(q_pe, cos, sin)
        k_pe = apply_interleaved_rope(k_pe.view(1, 1, self.rope_dim), cos, sin)
        q = torch.cat((q_pe, q_nope), dim=-1)
        k = torch.cat((k_pe.view(1, self.rope_dim), k_nope), dim=-1)

        torch_npu.scatter_update_(
            key_cache, position, k.view(1, 1, self.head_dim).contiguous(), 1
        )
        scores = torch.matmul(q.float(), key_cache.float().transpose(-1, -2))
        weights = weights.float().unsqueeze(-1)
        scores = (scores * weights).sum(dim=1)
        scores = scores * (self.head_dim**-0.5 * self.num_heads**-0.5)
        cache_positions = torch.arange(
            self.cache_length, device=position.device, dtype=torch.int64
        )
        scores = scores.masked_fill(
            cache_positions.unsqueeze(0) > position.unsqueeze(1),
            torch.finfo(scores.dtype).min,
        )
        return scores.topk(self.top_k, dim=-1).indices.to(torch.int64)


class GLM52DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        layer_index: int,
        config: GLM52Config,
        cache_length: int,
        fused_qkv_a: W8A8DynamicLinear,
        q_b_proj: W8A8DynamicLinear,
        kv_b_proj: BF16Linear,
        o_proj: W8A8DynamicLinear,
        mlp: nn.Module,
        indexer: GLM52DSAIndexer | None,
        input_norm: torch.Tensor,
        post_attention_norm: torch.Tensor,
        q_a_norm: torch.Tensor,
        kv_a_norm: torch.Tensor,
    ):
        super().__init__()
        self.layer_index = int(layer_index)
        self.config = config
        self.cache_length = int(cache_length)
        self.fused_qkv_a = fused_qkv_a
        self.q_b_proj = q_b_proj
        self.kv_b_proj = kv_b_proj
        self.o_proj = o_proj
        self.mlp = mlp
        self.indexer = indexer
        self.register_buffer("input_norm", input_norm.contiguous())
        self.register_buffer("post_attention_norm", post_attention_norm.contiguous())
        self.register_buffer("q_a_norm", q_a_norm.contiguous())
        self.register_buffer("kv_a_norm", kv_a_norm.contiguous())

        device = input_norm.device
        positions = torch.arange(cache_length, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(
                    0,
                    config.qk_rope_head_dim,
                    2,
                    device=device,
                    dtype=torch.float32,
                )
                / config.qk_rope_head_dim
            )
        )
        freqs = positions[:, None] * inv_freq[None, :]
        self.register_buffer(
            "rope_cos",
            freqs.cos().repeat_interleave(2, dim=-1).to(input_norm.dtype),
        )
        self.register_buffer(
            "rope_sin",
            freqs.sin().repeat_interleave(2, dim=-1).to(input_norm.dtype),
        )

    @classmethod
    def from_checkpoint(
        cls,
        reader: ShardedSafetensorReader,
        model_dir: str | Path,
        layer_index: int,
        *,
        cache_length: int,
        device: torch.device,
        progress=None,
    ) -> "GLM52DecoderLayer":
        config = GLM52Config.from_model_dir(model_dir)
        with (Path(model_dir) / "config.json").open() as handle:
            raw = json.load(handle)
        attn = f"model.layers.{layer_index}.self_attn"
        layer = f"model.layers.{layer_index}"
        if layer_index < config.first_k_dense_replace:
            mlp: nn.Module = GLM52DenseMLP.from_checkpoint(
                reader, layer_index, device=device
            )
        else:
            mlp = nn.ModuleDict(
                {
                    "routed": GLM52W4A8Experts.from_checkpoint(
                        reader,
                        layer_index,
                        config,
                        device=device,
                        progress=progress,
                    ),
                    "shared": GLM52SharedExpert.from_checkpoint(
                        reader, layer_index, device=device
                    ),
                }
            )

        indexer_type = raw["indexer_types"][layer_index]
        indexer = None
        if indexer_type == "full":
            indexer = GLM52DSAIndexer.from_checkpoint(
                reader,
                layer_index,
                num_heads=int(raw["index_n_heads"]),
                head_dim=int(raw["index_head_dim"]),
                rope_dim=config.qk_rope_head_dim,
                top_k=int(raw["index_topk"]),
                cache_length=cache_length,
                device=device,
            )
        elif indexer_type != "shared":
            raise ValueError(f"Unsupported indexer type {indexer_type!r}")

        return cls(
            layer_index=layer_index,
            config=config,
            cache_length=cache_length,
            fused_qkv_a=W8A8DynamicLinear.from_checkpoint(
                reader,
                [attn + ".q_a_proj", attn + ".kv_a_proj_with_mqa"],
                device=device,
            ),
            q_b_proj=W8A8DynamicLinear.from_checkpoint(
                reader, [attn + ".q_b_proj"], device=device
            ),
            kv_b_proj=BF16Linear.from_checkpoint(
                reader, attn + ".kv_b_proj", device=device
            ),
            o_proj=W8A8DynamicLinear.from_checkpoint(
                reader, [attn + ".o_proj"], device=device
            ),
            mlp=mlp,
            indexer=indexer,
            input_norm=reader.tensor(layer + ".input_layernorm.weight").to(
                device=device, dtype=torch.bfloat16
            ),
            post_attention_norm=reader.tensor(
                layer + ".post_attention_layernorm.weight"
            ).to(device=device, dtype=torch.bfloat16),
            q_a_norm=reader.tensor(attn + ".q_a_layernorm.weight").to(
                device=device, dtype=torch.bfloat16
            ),
            kv_a_norm=reader.tensor(attn + ".kv_a_layernorm.weight").to(
                device=device, dtype=torch.bfloat16
            ),
        )

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.mlp, nn.ModuleDict):
            return self.mlp["routed"](x) + self.mlp["shared"](x)
        return self.mlp(x)

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        cache_position: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        index_key_cache: torch.Tensor,
        shared_topk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        residual = hidden_states.clone()
        x = npu_rms_norm(hidden_states, self.input_norm, cfg.rms_norm_eps)
        fused = self.fused_qkv_a(x)
        q_a, compressed_kv, k_rope = torch.split(
            fused,
            [cfg.q_lora_rank, cfg.kv_lora_rank, cfg.qk_rope_head_dim],
            dim=-1,
        )
        q_a = npu_rms_norm(q_a, self.q_a_norm, cfg.rms_norm_eps)
        compressed_kv = npu_rms_norm(
            compressed_kv, self.kv_a_norm, cfg.rms_norm_eps
        )
        query = self.q_b_proj(q_a).view(
            1, 1, cfg.num_attention_heads, cfg.qk_head_dim
        )
        query_nope, query_rope = torch.split(
            query, [cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1
        )
        kv = self.kv_b_proj(compressed_kv).view(
            1,
            1,
            cfg.num_attention_heads,
            cfg.qk_nope_head_dim + cfg.v_head_dim,
        )
        key_nope, value = torch.split(
            kv, [cfg.qk_nope_head_dim, cfg.v_head_dim], dim=-1
        )
        position = cache_position.reshape(-1).to(torch.int64)
        cos = torch.index_select(self.rope_cos, 0, position).view(
            1, 1, 1, cfg.qk_rope_head_dim
        )
        sin = torch.index_select(self.rope_sin, 0, position).view(
            1, 1, 1, cfg.qk_rope_head_dim
        )
        query_rope = apply_interleaved_rope(query_rope, cos, sin)
        key_rope = apply_interleaved_rope(
            k_rope.view(1, 1, 1, cfg.qk_rope_head_dim), cos, sin
        ).expand(-1, -1, cfg.num_attention_heads, -1)
        query = torch.cat((query_nope, query_rope), dim=-1).transpose(1, 2)
        key = torch.cat((key_nope, key_rope), dim=-1).transpose(1, 2)
        value = value.transpose(1, 2)
        torch_npu.scatter_update_(key_cache, position, key.contiguous(), 2)
        torch_npu.scatter_update_(value_cache, position, value.contiguous(), 2)

        if self.indexer is not None:
            shared_topk = self.indexer(
                x,
                q_a,
                position,
                cos.view(1, 1, cfg.qk_rope_head_dim),
                sin.view(1, 1, cfg.qk_rope_head_dim),
                index_key_cache,
            )
        selected = shared_topk.reshape(-1)
        selected_key = torch.index_select(key_cache, 2, selected)
        selected_value = torch.index_select(value_cache, 2, selected)
        scores = torch.matmul(
            query.float(), selected_key.float().transpose(-1, -2)
        ) * (cfg.qk_head_dim**-0.5)
        valid = selected.unsqueeze(0) <= position.unsqueeze(1)
        scores = scores.masked_fill(
            ~valid.unsqueeze(1), torch.finfo(scores.dtype).min
        )
        probabilities = torch.softmax(scores, dim=-1).to(value.dtype)
        output = torch.matmul(probabilities, selected_value)
        output = output.transpose(1, 2).reshape(
            1, 1, cfg.attention_output_size
        )
        hidden_states = residual + self.o_proj(output)
        mlp_input = npu_rms_norm(
            hidden_states, self.post_attention_norm, cfg.rms_norm_eps
        )
        return hidden_states + self._mlp(mlp_input), shared_topk


class GLM52LayerStack(nn.Module):
    def __init__(self, layers: list[GLM52DecoderLayer], cache_length: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.cache_length = int(cache_length)
        self.config = layers[0].config
        self.top_k = min(2048, cache_length)

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
    ) -> "GLM52LayerStack":
        if first_layer != 0 or last_layer != 6:
            raise ValueError("The first owned stack rung is fixed to layers 0-6")
        reader = ShardedSafetensorReader(model_dir)
        layers = []
        for layer_index in range(first_layer, last_layer + 1):
            if progress is not None:
                progress(f"loading layer {layer_index}")
            layers.append(
                GLM52DecoderLayer.from_checkpoint(
                    reader,
                    model_dir,
                    layer_index,
                    cache_length=cache_length,
                    device=device,
                    progress=(
                        (lambda message, index=layer_index: progress(
                            f"layer {index}: {message}"
                        ))
                        if progress is not None
                        else None
                    ),
                )
            )
            if progress is not None:
                progress(f"loaded layer {layer_index}")
        return cls(layers, cache_length)

    def make_cache(
        self, *, device: torch.device
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        kv_shape = (
            1,
            self.config.num_attention_heads,
            self.cache_length,
            self.config.v_head_dim,
        )
        key_caches = []
        value_caches = []
        index_caches = []
        for layer in self.layers:
            key = torch.zeros(kv_shape, dtype=torch.bfloat16, device=device)
            key_caches.append(key)
            value_caches.append(torch.zeros_like(key))
            index_caches.append(
                torch.zeros(
                    (1, self.cache_length, 128),
                    dtype=torch.bfloat16,
                    device=device,
                )
                if layer.indexer is not None
                else torch.zeros((1, 1, 1), dtype=torch.bfloat16, device=device)
            )
        return tuple(key_caches), tuple(value_caches), tuple(index_caches)

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        cache_position: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
        index_caches: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        shared_topk = torch.arange(
            self.top_k, device=hidden_states.device, dtype=torch.int64
        ).view(1, -1)
        for index, layer in enumerate(self.layers):
            hidden_states, shared_topk = layer.forward_decode(
                hidden_states,
                cache_position,
                key_caches[index],
                value_caches[index],
                index_caches[index],
                shared_topk,
            )
        return hidden_states
