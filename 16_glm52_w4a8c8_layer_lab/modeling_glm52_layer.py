#!/usr/bin/env python3

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

try:
    import torch_npu
except ModuleNotFoundError:
    torch_npu = None


@dataclass(frozen=True)
class GLM52Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    moe_intermediate_size: int
    num_experts: int
    top_k: int
    rms_norm_eps: float
    rope_theta: float
    routed_scaling_factor: float
    norm_topk_prob: bool
    scoring_func: str
    first_k_dense_replace: int
    index_topk_freq: int
    index_skip_topk_offset: int

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def attention_output_size(self) -> int:
        return self.num_attention_heads * self.v_head_dim

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "GLM52Config":
        with (Path(model_dir) / "config.json").open() as handle:
            raw = json.load(handle)
        rope = raw["rope_parameters"]
        result = cls(
            hidden_size=int(raw["hidden_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            q_lora_rank=int(raw["q_lora_rank"]),
            kv_lora_rank=int(raw["kv_lora_rank"]),
            qk_nope_head_dim=int(raw["qk_nope_head_dim"]),
            qk_rope_head_dim=int(raw["qk_rope_head_dim"]),
            v_head_dim=int(raw["v_head_dim"]),
            moe_intermediate_size=int(raw["moe_intermediate_size"]),
            num_experts=int(raw["n_routed_experts"]),
            top_k=int(raw["num_experts_per_tok"]),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            rope_theta=float(rope["rope_theta"]),
            routed_scaling_factor=float(raw["routed_scaling_factor"]),
            norm_topk_prob=bool(raw["norm_topk_prob"]),
            scoring_func=str(raw["scoring_func"]),
            first_k_dense_replace=int(raw["first_k_dense_replace"]),
            index_topk_freq=int(raw["index_topk_freq"]),
            index_skip_topk_offset=int(raw["index_skip_topk_offset"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        actual = (
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.moe_intermediate_size,
            self.num_experts,
            self.top_k,
            self.scoring_func,
        )
        expected = (6144, 78, 64, 2048, 512, 192, 64, 256, 2048, 256, 8, "sigmoid")
        if actual != expected:
            raise ValueError(f"Unexpected GLM-5.2 configuration: {actual} != {expected}")

    def layer_uses_dsa_topk(self, layer_index: int) -> bool:
        offset = max(layer_index - self.index_skip_topk_offset + 1, 0)
        return offset % self.index_topk_freq == 0


class ShardedSafetensorReader:
    """Read selected checkpoint tensors, including completed ModelScope temp shards."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        with (self.model_dir / "quant_model_weights.safetensors.index.json").open() as handle:
            self.weight_map = json.load(handle)["weight_map"]
        self._handles: dict[Path, object] = {}

    def _resolve(self, shard_name: str) -> Path:
        normal = self.model_dir / shard_name
        temporary = self.model_dir / "._____temp" / shard_name
        for candidate in (normal, temporary):
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        raise FileNotFoundError(f"Checkpoint shard is not complete: {shard_name}")

    def tensor(self, name: str) -> torch.Tensor:
        shard_name = self.weight_map[name]
        path = self._resolve(shard_name)
        handle = self._handles.get(path)
        if handle is None:
            handle = safe_open(path, framework="pt", device="cpu")
            self._handles[path] = handle
        return handle.get_tensor(name)

    def shards_for_prefixes(self, prefixes: Iterable[str]) -> list[str]:
        prefixes = tuple(prefixes)
        return sorted({shard for name, shard in self.weight_map.items() if name.startswith(prefixes)})


def require_npu() -> None:
    if torch_npu is None:
        raise RuntimeError("torch_npu is required for GLM-5.2 layer inference")


def require_fused_w4_gmm1_op() -> None:
    """Register vLLM-Ascend's logical-width-aware fused W4 GMM binding."""
    importlib.import_module("vllm_ascend.vllm_ascend_C")
    if not hasattr(torch.ops._C_ascend, "grouped_matmul_swiglu_quant_v2"):
        raise RuntimeError("fused W4 grouped-matmul SwiGLU op is unavailable")


def npu_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    require_npu()
    return torch_npu.npu_rms_norm(x, weight, float(eps))[0]


def float_scale_to_int64_bits(scale: torch.Tensor) -> torch.Tensor:
    """Encode FP32 quant scales in the int64 representation used by ACLNN GMM."""
    scale_np = scale.detach().cpu().to(torch.float32).contiguous().numpy()
    bits = np.frombuffer(scale_np.tobytes(), dtype=np.int32).astype(np.int64)
    return torch.from_numpy(bits).reshape(scale_np.shape)


class W8A8DynamicLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor):
        super().__init__()
        if weight.dtype != torch.int8 or weight.dim() != 2:
            raise ValueError(f"Expected INT8 [K,N] weight, got {weight.dtype} {weight.shape}")
        if weight_scale.numel() != weight.shape[1]:
            raise ValueError("W8 scale count does not match output width")
        self.register_buffer("weight", weight.contiguous())
        self.register_buffer("weight_scale", weight_scale.flatten().to(torch.float32).contiguous())

    @classmethod
    def from_checkpoint(
        cls,
        reader: ShardedSafetensorReader,
        prefixes: list[str],
        *,
        device: torch.device,
    ) -> "W8A8DynamicLinear":
        checkpoint_weights = [reader.tensor(prefix + ".weight") for prefix in prefixes]
        checkpoint_scales = [reader.tensor(prefix + ".weight_scale") for prefix in prefixes]
        weight = torch.cat(checkpoint_weights, dim=0).transpose(0, 1).contiguous().to(device)
        scale = torch.cat(checkpoint_scales, dim=0).flatten().contiguous().to(device)
        return cls(weight, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        require_npu()
        leading = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        quantized, per_token_scale = torch_npu.npu_dynamic_quant(flat, dst_type=torch.int8)
        output = torch_npu.npu_quant_matmul(
            quantized,
            self.weight,
            self.weight_scale,
            pertoken_scale=per_token_scale,
            output_dtype=x.dtype,
        )
        return output.reshape(*leading, output.shape[-1])


class BF16Linear(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight.contiguous())

    @classmethod
    def from_checkpoint(
        cls,
        reader: ShardedSafetensorReader,
        prefix: str,
        *,
        device: torch.device,
    ) -> "BF16Linear":
        return cls(reader.tensor(prefix + ".weight").to(device=device, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GE can interpret the singleton sequence axis of [B,1,K] as the
        # contraction dimension. Present an ordinary 2-D matmul and restore the
        # leading dimensions after projection.
        leading = x.shape[:-1]
        output = F.linear(x.reshape(-1, x.shape[-1]), self.weight)
        return output.reshape(*leading, output.shape[-1])


class GLM52W4A8Experts(nn.Module):
    """Owned TP1 W4A8 routed-expert implementation for one GLM-5.2 layer."""

    def __init__(
        self,
        *,
        router_weight: torch.Tensor,
        correction_bias: torch.Tensor,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_scale: torch.Tensor,
        w13_scale_f32: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_bias: torch.Tensor,
        w2_bias: torch.Tensor,
        config: GLM52Config,
        fuse_gmm1_swiglu_quant: bool = False,
    ):
        super().__init__()
        self.config = config
        self.register_buffer("router_weight", router_weight.contiguous())
        self.register_buffer("correction_bias", correction_bias.float().contiguous())
        self.register_buffer("w13_weight", w13_weight.contiguous())
        self.register_buffer("w2_weight", w2_weight.contiguous())
        self.register_buffer("w13_scale", w13_scale.contiguous())
        self.register_buffer("w13_scale_f32", w13_scale_f32.contiguous())
        self.register_buffer("w2_scale", w2_scale.contiguous())
        self.register_buffer("w13_bias", w13_bias.float().contiguous())
        self.register_buffer("w2_bias", w2_bias.float().contiguous())
        self.fuse_gmm1_swiglu_quant = bool(fuse_gmm1_swiglu_quant)
        if self.fuse_gmm1_swiglu_quant:
            require_fused_w4_gmm1_op()

    @classmethod
    def from_checkpoint(
        cls,
        reader: ShardedSafetensorReader,
        layer_index: int,
        config: GLM52Config,
        *,
        device: torch.device,
        progress=None,
        w4_weight_format: str = "native",
        fuse_gmm1_swiglu_quant: bool = False,
    ) -> "GLM52W4A8Experts":
        prefix = f"model.layers.{layer_index}.mlp"
        router_weight = reader.tensor(prefix + ".gate.weight").to(device=device, dtype=torch.bfloat16)
        correction_bias = reader.tensor(prefix + ".gate.e_score_correction_bias").to(device=device)

        # Store weights directly in operator orientation. Four packed INT8 bytes
        # are reinterpreted as one INT32 element after loading.
        w13_i8 = torch.empty(
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            dtype=torch.int8,
            device=device,
        )
        w2_i8 = torch.empty(
            config.num_experts,
            config.moe_intermediate_size,
            config.hidden_size // 2,
            dtype=torch.int8,
            device=device,
        )
        w13_scale_f32 = torch.empty(
            config.num_experts,
            2 * config.moe_intermediate_size,
            dtype=torch.float32,
        )
        w2_scale_f32 = torch.empty(
            config.num_experts,
            config.hidden_size,
            dtype=torch.float32,
        )
        w13_bias_cpu = torch.empty_like(w13_scale_f32)
        w2_bias_cpu = torch.empty_like(w2_scale_f32)

        packed_half = config.moe_intermediate_size // 2
        for expert in range(config.num_experts):
            expert_prefix = f"{prefix}.experts.{expert}"
            gate = reader.tensor(expert_prefix + ".gate_proj.weight")
            up = reader.tensor(expert_prefix + ".up_proj.weight")
            down = reader.tensor(expert_prefix + ".down_proj.weight")
            w13_i8[expert, :, :packed_half].copy_(gate.transpose(0, 1).contiguous().to(device))
            w13_i8[expert, :, packed_half:].copy_(up.transpose(0, 1).contiguous().to(device))
            w2_i8[expert].copy_(down.transpose(0, 1).contiguous().to(device))

            w13_scale_f32[expert].copy_(
                torch.cat(
                    (
                        reader.tensor(expert_prefix + ".gate_proj.weight_scale").flatten(),
                        reader.tensor(expert_prefix + ".up_proj.weight_scale").flatten(),
                    )
                )
            )
            w2_scale_f32[expert].copy_(
                reader.tensor(expert_prefix + ".down_proj.weight_scale").flatten()
            )
            w13_bias_cpu[expert].copy_(
                torch.cat(
                    (
                        reader.tensor(expert_prefix + ".gate_proj.scale_bias").flatten(),
                        reader.tensor(expert_prefix + ".up_proj.scale_bias").flatten(),
                    )
                )
            )
            w2_bias_cpu[expert].copy_(
                reader.tensor(expert_prefix + ".down_proj.scale_bias").sum(dim=-1)
            )
            if progress is not None and (expert + 1) % 32 == 0:
                progress(f"loaded routed experts {expert + 1}/{config.num_experts}")

        if w4_weight_format == "fractal_nz":
            require_npu()
            w13_i8 = torch_npu.npu_format_cast(w13_i8, 29)
            w2_i8 = torch_npu.npu_format_cast(w2_i8, 29)
        elif w4_weight_format != "native":
            raise ValueError(f"Unsupported W4 weight format: {w4_weight_format}")
        w13_weight = w13_i8.view(torch.int32).contiguous()
        w2_weight = w2_i8.view(torch.int32).contiguous()
        return cls(
            router_weight=router_weight,
            correction_bias=correction_bias,
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            # CANN 9.0 GroupedMatmulV5 needs the explicit one-group axis for
            # per-channel W4 scales. Squeezing this to [E,N] makes the tiler
            # misread N as quantGroupNum and reject K=6144.
            w13_scale=float_scale_to_int64_bits(w13_scale_f32)
            .unsqueeze(1)
            .to(device),
            # The fused per-channel GMM1 contract expects [E,N]. The generic
            # W4 GMM contract above still needs the explicit [E,1,N] axis.
            w13_scale_f32=w13_scale_f32.to(device),
            w2_scale=float_scale_to_int64_bits(w2_scale_f32).unsqueeze(1).to(device),
            w13_bias=w13_bias_cpu.to(device),
            w2_bias=w2_bias_cpu.to(device),
            config=config,
            fuse_gmm1_swiglu_quant=fuse_gmm1_swiglu_quant,
        )

    def route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(hidden_states, self.router_weight)
        scores = logits.sigmoid()
        selection_scores = scores.float() + self.correction_bias
        _selection_values, selected = selection_scores.topk(self.config.top_k, dim=-1)
        weights = scores.gather(-1, selected)
        if self.config.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.config.routed_scaling_factor
        return weights.to(hidden_states.dtype), selected

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        require_npu()
        flat = hidden_states.reshape(-1, self.config.hidden_size)
        routing_weights, selected = self.route(flat)
        selected_i32 = selected.to(torch.int32)
        expanded, expanded_row_idx, group_counts, _expanded_scale = (
            torch_npu.npu_moe_init_routing_v2(
                flat,
                selected_i32,
                scale=None,
                offset=None,
                active_num=flat.shape[0] * self.config.top_k,
                expert_capacity=-1,
                expert_num=self.config.num_experts,
                drop_pad_mode=0,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                quant_mode=-1,
                active_expert_range=[0, self.config.num_experts],
                row_idx_type=0,
            )
        )
        expanded_i8, expanded_scale = torch_npu.npu_dynamic_quant(
            expanded, dst_type=torch.int8
        )
        if self.fuse_gmm1_swiglu_quant:
            activated_i8, activated_scale = (
                torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
                    expanded_i8,
                    [self.w13_weight.view(torch.int8)],
                    [self.w13_scale.squeeze(1)],
                    expanded_scale,
                    group_counts,
                    weight_assist_matrix=[self.w13_bias],
                    dequant_mode=0,
                    dequant_dtype=0,
                    quant_mode=0,
                    quant_dtype=0,
                    transpose_weight=False,
                    group_list_type=1,
                    tuning_config=[],
                    swiglu_limit=0.0,
                )
            )
        else:
            gate_up = torch_npu.npu_grouped_matmul(
                x=[expanded_i8],
                weight=[self.w13_weight],
                scale=[self.w13_scale],
                bias=[self.w13_bias],
                per_token_scale=[expanded_scale],
                group_list=group_counts,
                split_item=2,
                group_type=0,
                group_list_type=1,
                output_dtype=hidden_states.dtype,
            )[0]
            activated = torch_npu.npu_swiglu(gate_up)
            activated_i8, activated_scale = torch_npu.npu_dynamic_quant(
                activated, dst_type=torch.int8
            )
        expert_output = torch_npu.npu_grouped_matmul(
            x=[activated_i8],
            weight=[self.w2_weight],
            scale=[self.w2_scale],
            bias=[self.w2_bias],
            per_token_scale=[activated_scale],
            group_list=group_counts,
            split_item=2,
            group_type=0,
            group_list_type=1,
            output_dtype=hidden_states.dtype,
        )[0]
        output = torch_npu.npu_moe_finalize_routing(
            expert_output,
            None,
            None,
            None,
            routing_weights,
            expanded_row_idx,
            selected_i32,
            0,
        )
        return output.view_as(hidden_states)


class GLM52SharedExpert(nn.Module):
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
    ) -> "GLM52SharedExpert":
        prefix = f"model.layers.{layer_index}.mlp.shared_experts"
        return cls(
            W8A8DynamicLinear.from_checkpoint(
                reader,
                [prefix + ".gate_proj", prefix + ".up_proj"],
                device=device,
            ),
            W8A8DynamicLinear.from_checkpoint(
                reader,
                [prefix + ".down_proj"],
                device=device,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch_npu.npu_swiglu(self.gate_up(x)))


def apply_interleaved_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    paired = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    rotated = torch.stack((-paired[..., 1], paired[..., 0]), dim=-1).flatten(-2)
    return x * cos + rotated * sin


class GLM52Layer3(nn.Module):
    """Owned layer-3 decode path with dense, unabsorbed MLA attention."""

    def __init__(
        self,
        *,
        config: GLM52Config,
        cache_length: int,
        fused_qkv_a: W8A8DynamicLinear,
        q_b_proj: W8A8DynamicLinear,
        kv_b_proj: BF16Linear,
        o_proj: W8A8DynamicLinear,
        shared_expert: GLM52SharedExpert,
        routed_experts: GLM52W4A8Experts,
        input_norm: torch.Tensor,
        post_attention_norm: torch.Tensor,
        q_a_norm: torch.Tensor,
        kv_a_norm: torch.Tensor,
    ):
        super().__init__()
        self.config = config
        self.cache_length = int(cache_length)
        self.fused_qkv_a = fused_qkv_a
        self.q_b_proj = q_b_proj
        self.kv_b_proj = kv_b_proj
        self.o_proj = o_proj
        self.shared_expert = shared_expert
        self.routed_experts = routed_experts
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
        cos = freqs.cos().repeat_interleave(2, dim=-1).to(input_norm.dtype)
        sin = freqs.sin().repeat_interleave(2, dim=-1).to(input_norm.dtype)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    @classmethod
    def from_checkpoint(
        cls,
        model_dir: str | Path,
        *,
        layer_index: int,
        cache_length: int,
        device: torch.device,
        progress=None,
    ) -> "GLM52Layer3":
        if layer_index != 3:
            raise ValueError("The first owned rung is intentionally fixed to layer 3")
        config = GLM52Config.from_model_dir(model_dir)
        if config.layer_uses_dsa_topk(layer_index):
            raise ValueError("Layer 3 was expected to skip the DSA top-k indexer")
        reader = ShardedSafetensorReader(model_dir)
        attn = f"model.layers.{layer_index}.self_attn"
        layer = f"model.layers.{layer_index}"
        return cls(
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
            kv_b_proj=BF16Linear.from_checkpoint(reader, attn + ".kv_b_proj", device=device),
            o_proj=W8A8DynamicLinear.from_checkpoint(
                reader, [attn + ".o_proj"], device=device
            ),
            shared_expert=GLM52SharedExpert.from_checkpoint(
                reader, layer_index, device=device
            ),
            routed_experts=GLM52W4A8Experts.from_checkpoint(
                reader,
                layer_index,
                config,
                device=device,
                progress=progress,
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
        )

    def make_cache(self, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (
            1,
            self.config.num_attention_heads,
            self.cache_length,
            self.config.v_head_dim,
        )
        key = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        value = torch.zeros_like(key)
        return key, value

    def attention(
        self,
        x: torch.Tensor,
        cache_position: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
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
        kv_positions = torch.arange(
            self.cache_length, device=x.device, dtype=torch.int64
        )
        attention_mask = kv_positions.unsqueeze(0) > position.unsqueeze(1)
        output = torch_npu.npu_incre_flash_attention(
            query,
            key_cache,
            value_cache,
            atten_mask=attention_mask.contiguous(),
            num_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_attention_heads,
            input_layout="BNSD",
            scale_value=cfg.qk_head_dim**-0.5,
        )
        output = output.transpose(1, 2).reshape(1, 1, cfg.attention_output_size)
        return self.o_proj(output)

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        cache_position: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        # Match the standalone decoder-layer contract used when there is no
        # incoming fused residual. Keeping an explicit residual copy also
        # prevents compiled buffer reuse from consuming the caller's input.
        residual = hidden_states.clone()
        attention_input = npu_rms_norm(
            hidden_states, self.input_norm, self.config.rms_norm_eps
        )
        hidden_states = residual + self.attention(
            attention_input, cache_position, key_cache, value_cache
        )
        mlp_input = npu_rms_norm(
            hidden_states, self.post_attention_norm, self.config.rms_norm_eps
        )
        mlp_output = self.routed_experts(mlp_input) + self.shared_expert(mlp_input)
        return hidden_states + mlp_output
