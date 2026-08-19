#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from absorbed_mla import (
    absorb_kv_b_weight,
    manual_absorbed_attention,
    materialize_absorbed_kv,
    sparse_flash_absorbed_attention,
)
from modeling_glm52_layer import (
    BF16Linear,
    GLM52Config,
    ShardedSafetensorReader,
    W8A8DynamicLinear,
    apply_interleaved_rope,
    npu_rms_norm,
    torch_npu,
)
from modeling_glm52_stack import GLM52DSAIndexer


ATTENTION_PATHS = (
    "decompressed_manual",
    "absorbed_manual",
    "absorbed_sparse_flash",
)


def shard_bounds(size: int, rank: int, world_size: int) -> tuple[int, int]:
    if size % world_size:
        raise ValueError(f"size={size} is not divisible by world_size={world_size}")
    width = size // world_size
    return rank * width, (rank + 1) * width


def tp_all_reduce_sum(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _assert_symmetric(reader: ShardedSafetensorReader, prefix: str) -> None:
    offset = reader.tensor(prefix + ".weight_offset")
    if bool(torch.count_nonzero(offset).item()):
        raise ValueError(f"TP loader requires zero weight offsets: {prefix}")


def load_w8_column_parallel(
    reader: ShardedSafetensorReader,
    prefixes: list[str],
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> W8A8DynamicLinear:
    weights = []
    scales = []
    for prefix in prefixes:
        _assert_symmetric(reader, prefix)
        checkpoint_weight = reader.tensor(prefix + ".weight")
        start, end = shard_bounds(checkpoint_weight.shape[0], rank, world_size)
        weights.append(checkpoint_weight[start:end])
        scales.append(reader.tensor(prefix + ".weight_scale")[start:end])
    weight = torch.cat(weights, dim=0).transpose(0, 1).contiguous().to(device)
    scale = torch.cat(scales, dim=0).flatten().contiguous().to(device)
    return W8A8DynamicLinear(weight, scale)


def load_w8_row_parallel(
    reader: ShardedSafetensorReader,
    prefix: str,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> W8A8DynamicLinear:
    _assert_symmetric(reader, prefix)
    checkpoint_weight = reader.tensor(prefix + ".weight")
    start, end = shard_bounds(checkpoint_weight.shape[1], rank, world_size)
    weight = checkpoint_weight[:, start:end].transpose(0, 1).contiguous().to(device)
    scale = reader.tensor(prefix + ".weight_scale").flatten().contiguous().to(device)
    return W8A8DynamicLinear(weight, scale)


def load_bf16_column_parallel(
    reader: ShardedSafetensorReader,
    prefix: str,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> BF16Linear:
    checkpoint_weight = reader.tensor(prefix + ".weight")
    start, end = shard_bounds(checkpoint_weight.shape[0], rank, world_size)
    return BF16Linear(
        checkpoint_weight[start:end].to(device=device, dtype=torch.bfloat16)
    )


class GLM52DenseTPMLP(nn.Module):
    def __init__(
        self,
        gate_up: W8A8DynamicLinear,
        down: W8A8DynamicLinear,
        *,
        world_size: int,
    ):
        super().__init__()
        self.gate_up = gate_up
        self.down = down
        self.world_size = int(world_size)

    @classmethod
    def from_checkpoint(
        cls,
        reader: ShardedSafetensorReader,
        layer_index: int,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
    ) -> "GLM52DenseTPMLP":
        prefix = f"model.layers.{layer_index}.mlp"
        return cls(
            load_w8_column_parallel(
                reader,
                [prefix + ".gate_proj", prefix + ".up_proj"],
                rank=rank,
                world_size=world_size,
                device=device,
            ),
            load_w8_row_parallel(
                reader,
                prefix + ".down_proj",
                rank=rank,
                world_size=world_size,
                device=device,
            ),
            world_size=world_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.down(torch_npu.npu_swiglu(self.gate_up(x)))
        return tp_all_reduce_sum(local, self.world_size)


class GLM52DenseTPDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        layer_index: int,
        config: GLM52Config,
        cache_length: int,
        rank: int,
        world_size: int,
        fused_qkv_a: W8A8DynamicLinear,
        q_b_proj: W8A8DynamicLinear,
        kv_b_proj: BF16Linear,
        o_proj: W8A8DynamicLinear,
        mlp: GLM52DenseTPMLP,
        indexer: GLM52DSAIndexer,
        input_norm: torch.Tensor,
        post_attention_norm: torch.Tensor,
        q_a_norm: torch.Tensor,
        kv_a_norm: torch.Tensor,
        attention_path: str,
    ):
        super().__init__()
        if attention_path not in ATTENTION_PATHS:
            raise ValueError(f"Unsupported attention path: {attention_path}")
        self.layer_index = int(layer_index)
        self.config = config
        self.cache_length = int(cache_length)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_heads = config.num_attention_heads // world_size
        self.attention_path = attention_path
        self.fused_qkv_a = fused_qkv_a
        self.q_b_proj = q_b_proj
        if attention_path == "decompressed_manual":
            self.kv_b_proj = kv_b_proj
            self.register_buffer("w_uk_t", None)
            self.register_buffer("w_uv", None)
        else:
            w_uk_t, w_uv = absorb_kv_b_weight(
                kv_b_proj.weight,
                local_heads=self.local_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                v_head_dim=config.v_head_dim,
                kv_lora_rank=config.kv_lora_rank,
            )
            self.kv_b_proj = None
            self.register_buffer("w_uk_t", w_uk_t)
            self.register_buffer("w_uv", w_uv)
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
        rank: int,
        world_size: int,
        device: torch.device,
        attention_path: str,
    ) -> "GLM52DenseTPDecoderLayer":
        config = GLM52Config.from_model_dir(model_dir)
        with (Path(model_dir) / "config.json").open() as handle:
            raw = json.load(handle)
        if layer_index >= config.first_k_dense_replace:
            raise ValueError(f"layer {layer_index} is not a dense layer")
        if raw["indexer_types"][layer_index] != "full":
            raise ValueError(f"layer {layer_index} must own a full DSA indexer")
        attn = f"model.layers.{layer_index}.self_attn"
        layer = f"model.layers.{layer_index}"
        return cls(
            layer_index=layer_index,
            config=config,
            cache_length=cache_length,
            rank=rank,
            world_size=world_size,
            fused_qkv_a=W8A8DynamicLinear.from_checkpoint(
                reader,
                [attn + ".q_a_proj", attn + ".kv_a_proj_with_mqa"],
                device=device,
            ),
            q_b_proj=load_w8_column_parallel(
                reader,
                [attn + ".q_b_proj"],
                rank=rank,
                world_size=world_size,
                device=device,
            ),
            kv_b_proj=load_bf16_column_parallel(
                reader,
                attn + ".kv_b_proj",
                rank=rank,
                world_size=world_size,
                device=device,
            ),
            o_proj=load_w8_row_parallel(
                reader,
                attn + ".o_proj",
                rank=rank,
                world_size=world_size,
                device=device,
            ),
            mlp=GLM52DenseTPMLP.from_checkpoint(
                reader,
                layer_index,
                rank=rank,
                world_size=world_size,
                device=device,
            ),
            indexer=GLM52DSAIndexer.from_checkpoint(
                reader,
                layer_index,
                num_heads=int(raw["index_n_heads"]),
                head_dim=int(raw["index_head_dim"]),
                rope_dim=config.qk_rope_head_dim,
                top_k=int(raw["index_topk"]),
                cache_length=cache_length,
                device=device,
            ),
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
            attention_path=attention_path,
        )

    def materialize_reference_kv(
        self,
        primary_cache: torch.Tensor,
        secondary_cache: torch.Tensor,
        *,
        used_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.attention_path == "decompressed_manual":
            return (
                primary_cache[:, :, :used_length],
                secondary_cache[:, :, :used_length],
            )
        assert self.w_uk_t is not None
        assert self.w_uv is not None
        return materialize_absorbed_kv(
            primary_cache,
            secondary_cache,
            self.w_uk_t,
            self.w_uv,
            used_length=used_length,
        )

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        cache_position: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        index_key_cache: torch.Tensor,
    ) -> torch.Tensor:
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
            1, 1, self.local_heads, cfg.qk_head_dim
        )
        query_nope, query_rope = torch.split(
            query, [cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1
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
        )

        if self.attention_path == "decompressed_manual":
            assert self.kv_b_proj is not None
            kv = self.kv_b_proj(compressed_kv).view(
                1,
                1,
                self.local_heads,
                cfg.qk_nope_head_dim + cfg.v_head_dim,
            )
            key_nope, value = torch.split(
                kv, [cfg.qk_nope_head_dim, cfg.v_head_dim], dim=-1
            )
            expanded_key_rope = key_rope.expand(
                -1, -1, self.local_heads, -1
            )
            query = torch.cat((query_nope, query_rope), dim=-1).transpose(1, 2)
            key = torch.cat((key_nope, expanded_key_rope), dim=-1).transpose(1, 2)
            value = value.transpose(1, 2)
            torch_npu.scatter_update_(key_cache, position, key.contiguous(), 2)
            torch_npu.scatter_update_(value_cache, position, value.contiguous(), 2)
        else:
            torch_npu.scatter_update_(
                key_cache, position, compressed_kv.contiguous(), 1
            )
            torch_npu.scatter_update_(
                value_cache,
                position,
                key_rope.view(1, 1, cfg.qk_rope_head_dim).contiguous(),
                1,
            )

        selected = self.indexer(
            x,
            q_a,
            position,
            cos.view(1, 1, cfg.qk_rope_head_dim),
            sin.view(1, 1, cfg.qk_rope_head_dim),
            index_key_cache,
        ).reshape(-1)
        if self.attention_path == "decompressed_manual":
            selected_key = torch.index_select(key_cache, 2, selected)
            selected_value = torch.index_select(value_cache, 2, selected)
            sparse_k = selected.shape[0]
            scores = torch.bmm(
                query.reshape(self.local_heads, 1, cfg.qk_head_dim).float(),
                selected_key.reshape(
                    self.local_heads, sparse_k, cfg.qk_head_dim
                ).float().transpose(1, 2),
            ).view(1, self.local_heads, 1, sparse_k)
            scores = scores * (cfg.qk_head_dim**-0.5)
            valid = selected.unsqueeze(0) <= position.unsqueeze(1)
            scores = scores.masked_fill(
                ~valid.unsqueeze(1), torch.finfo(scores.dtype).min
            )
            probabilities = torch.softmax(scores, dim=-1).to(value.dtype)
            local_attention = torch.bmm(
                probabilities.reshape(self.local_heads, 1, sparse_k),
                selected_value.reshape(
                    self.local_heads, sparse_k, cfg.v_head_dim
                ),
            ).view(1, self.local_heads, 1, cfg.v_head_dim)
            local_attention = local_attention.transpose(1, 2).reshape(
                1, 1, self.local_heads * cfg.v_head_dim
            )
        elif self.attention_path == "absorbed_manual":
            assert self.w_uk_t is not None
            assert self.w_uv is not None
            local_attention = manual_absorbed_attention(
                query_nope,
                query_rope,
                key_cache,
                value_cache,
                self.w_uk_t,
                self.w_uv,
                selected,
                position,
                scale=cfg.qk_head_dim**-0.5,
            )
        else:
            assert self.attention_path == "absorbed_sparse_flash"
            assert self.w_uk_t is not None
            assert self.w_uv is not None
            local_attention = sparse_flash_absorbed_attention(
                query_nope,
                query_rope,
                key_cache,
                value_cache,
                self.w_uk_t,
                self.w_uv,
                selected,
                position,
                scale=cfg.qk_head_dim**-0.5,
            )
        attention_output = tp_all_reduce_sum(
            self.o_proj(local_attention), self.world_size
        )
        hidden_states = residual + attention_output
        mlp_input = npu_rms_norm(
            hidden_states, self.post_attention_norm, cfg.rms_norm_eps
        )
        return hidden_states + self.mlp(mlp_input)


class GLM52DenseTPStack(nn.Module):
    def __init__(
        self,
        layers: list[GLM52DenseTPDecoderLayer],
        *,
        rank: int,
        world_size: int,
        cache_length: int,
        attention_path: str,
    ):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.cache_length = int(cache_length)
        self.attention_path = attention_path
        self.config = layers[0].config
        self.local_heads = self.config.num_attention_heads // world_size

    @classmethod
    def from_checkpoint(
        cls,
        model_dir: str | Path,
        *,
        rank: int,
        world_size: int,
        cache_length: int,
        device: torch.device,
        attention_path: str = "decompressed_manual",
        progress=None,
    ) -> "GLM52DenseTPStack":
        if world_size not in (1, 2):
            raise ValueError("The dense TP rung supports TP1 and TP2")
        reader = ShardedSafetensorReader(model_dir)
        layers = []
        for layer_index in range(3):
            if progress is not None:
                progress(f"loading dense layer {layer_index}")
            layers.append(
                GLM52DenseTPDecoderLayer.from_checkpoint(
                    reader,
                    model_dir,
                    layer_index,
                    cache_length=cache_length,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    attention_path=attention_path,
                )
            )
            if progress is not None:
                progress(f"loaded dense layer {layer_index}")
        return cls(
            layers,
            rank=rank,
            world_size=world_size,
            cache_length=cache_length,
            attention_path=attention_path,
        )

    def make_cache(self, *, device: torch.device):
        if self.attention_path == "decompressed_manual":
            primary_shape = (
                1,
                self.local_heads,
                self.cache_length,
                self.config.v_head_dim,
            )
            secondary_shape = primary_shape
        else:
            primary_shape = (
                1,
                self.cache_length,
                self.config.kv_lora_rank,
            )
            secondary_shape = (
                1,
                self.cache_length,
                self.config.qk_rope_head_dim,
            )
        keys = tuple(
            torch.zeros(primary_shape, dtype=torch.bfloat16, device=device)
            for _ in self.layers
        )
        values = tuple(
            torch.zeros(secondary_shape, dtype=torch.bfloat16, device=device)
            for _ in self.layers
        )
        indices = tuple(
            torch.zeros(
                (1, self.cache_length, 128),
                dtype=torch.bfloat16,
                device=device,
            )
            for _ in self.layers
        )
        return keys, values, indices

    def materialize_reference_caches(self, caches, *, used_length: int):
        primary, secondary, indices = caches
        keys = []
        values = []
        for layer_index, layer in enumerate(self.layers):
            key, value = layer.materialize_reference_kv(
                primary[layer_index],
                secondary[layer_index],
                used_length=used_length,
            )
            keys.append(key)
            values.append(value)
        return tuple(keys), tuple(values), tuple(
            cache[:, :used_length] for cache in indices
        )

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        cache_position: torch.Tensor,
        key_caches: tuple[torch.Tensor, ...],
        value_caches: tuple[torch.Tensor, ...],
        index_caches: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            hidden_states = layer.forward_decode(
                hidden_states,
                cache_position,
                key_caches[index],
                value_caches[index],
                index_caches[index],
            )
        return hidden_states
