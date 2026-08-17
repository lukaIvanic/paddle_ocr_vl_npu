#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open

from modeling_qwen3_moe_pipeline import Qwen3MoePipelineStage


class SafeTensorIndex:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        with (self.model_dir / "model.safetensors.index.json").open() as handle:
            payload = json.load(handle)
        self.weight_map = payload["weight_map"]
        self.total_size = int(payload.get("metadata", {}).get("total_size", 0))
        self._stack = contextlib.ExitStack()
        self._handles = {}

    def __enter__(self) -> "SafeTensorIndex":
        for filename in sorted(set(self.weight_map.values())):
            self._handles[filename] = self._stack.enter_context(
                safe_open(
                    str(self.model_dir / filename), framework="pt", device="cpu"
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stack.close()

    def tensor(self, name: str) -> torch.Tensor:
        filename = self.weight_map.get(name)
        if filename is None:
            raise KeyError(f"Checkpoint tensor is missing: {name}")
        return self._handles[filename].get_tensor(name)


def copy_parameter(
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    device: torch.device,
) -> None:
    if tuple(target.shape) != tuple(source.shape):
        raise RuntimeError(
            f"Weight shape mismatch: target={tuple(target.shape)} "
            f"source={tuple(source.shape)}"
        )
    staged = source.contiguous().to(device=device, dtype=target.dtype)
    target.copy_(staged)


def load_pipeline_stage(
    stage: Qwen3MoePipelineStage,
    model_dir: str | Path,
    *,
    device: torch.device,
    progress: Callable[[str], None] | None = None,
) -> None:
    log = progress or (lambda _message: None)
    config = stage.config
    with SafeTensorIndex(model_dir) as checkpoint, torch.no_grad():
        if stage.embed_tokens is not None:
            log("loading token embedding")
            copy_parameter(
                stage.embed_tokens.weight,
                checkpoint.tensor("model.embed_tokens.weight"),
                device=device,
            )

        for local_index, layer in enumerate(stage.layers):
            global_index = stage.layer_start + local_index
            prefix = f"model.layers.{global_index}"
            q_weight = checkpoint.tensor(f"{prefix}.self_attn.q_proj.weight")
            k_weight = checkpoint.tensor(f"{prefix}.self_attn.k_proj.weight")
            v_weight = checkpoint.tensor(f"{prefix}.self_attn.v_proj.weight")
            copy_parameter(
                layer.self_attn.qkv_proj.weight,
                torch.cat((q_weight, k_weight, v_weight), dim=0),
                device=device,
            )
            copy_parameter(
                layer.self_attn.o_proj.weight,
                checkpoint.tensor(f"{prefix}.self_attn.o_proj.weight"),
                device=device,
            )
            copy_parameter(
                layer.self_attn.q_norm.weight,
                checkpoint.tensor(f"{prefix}.self_attn.q_norm.weight"),
                device=device,
            )
            copy_parameter(
                layer.self_attn.k_norm.weight,
                checkpoint.tensor(f"{prefix}.self_attn.k_norm.weight"),
                device=device,
            )
            copy_parameter(
                layer.input_layernorm.weight,
                checkpoint.tensor(f"{prefix}.input_layernorm.weight"),
                device=device,
            )
            copy_parameter(
                layer.post_attention_layernorm.weight,
                checkpoint.tensor(f"{prefix}.post_attention_layernorm.weight"),
                device=device,
            )
            copy_parameter(
                layer.mlp.gate.weight,
                checkpoint.tensor(f"{prefix}.mlp.gate.weight"),
                device=device,
            )

            grouped_matmul = layer.mlp.expert_impl == "grouped_matmul"
            gate_up_shape = (
                (
                    config.num_experts,
                    config.hidden_size,
                    2 * config.moe_intermediate_size,
                )
                if grouped_matmul
                else (
                    config.num_experts,
                    2 * config.moe_intermediate_size,
                    config.hidden_size,
                )
            )
            down_shape = (
                (
                    config.num_experts,
                    config.moe_intermediate_size,
                    config.hidden_size,
                )
                if grouped_matmul
                else (
                    config.num_experts,
                    config.hidden_size,
                    config.moe_intermediate_size,
                )
            )
            gate_up_cpu = torch.empty(
                gate_up_shape,
                dtype=checkpoint.tensor(
                    f"{prefix}.mlp.experts.0.gate_proj.weight"
                ).dtype,
            )
            down_cpu = torch.empty(
                down_shape,
                dtype=checkpoint.tensor(
                    f"{prefix}.mlp.experts.0.down_proj.weight"
                ).dtype,
            )
            for expert_index in range(config.num_experts):
                expert_prefix = f"{prefix}.mlp.experts.{expert_index}"
                gate_weight = checkpoint.tensor(
                    f"{expert_prefix}.gate_proj.weight"
                )
                up_weight = checkpoint.tensor(f"{expert_prefix}.up_proj.weight")
                down_weight = checkpoint.tensor(
                    f"{expert_prefix}.down_proj.weight"
                )
                if grouped_matmul:
                    gate_up_cpu[
                        expert_index, :, : config.moe_intermediate_size
                    ].copy_(gate_weight.transpose(0, 1))
                    gate_up_cpu[
                        expert_index, :, config.moe_intermediate_size :
                    ].copy_(up_weight.transpose(0, 1))
                    down_cpu[expert_index].copy_(down_weight.transpose(0, 1))
                else:
                    gate_up_cpu[
                        expert_index, : config.moe_intermediate_size
                    ].copy_(gate_weight)
                    gate_up_cpu[
                        expert_index, config.moe_intermediate_size :
                    ].copy_(up_weight)
                    down_cpu[expert_index].copy_(down_weight)
            copy_parameter(layer.mlp.gate_up_proj, gate_up_cpu, device=device)
            copy_parameter(layer.mlp.down_proj, down_cpu, device=device)
            del gate_up_cpu, down_cpu
            log(
                f"loaded global layer {global_index} "
                f"({local_index + 1}/{stage.num_layers})"
            )

        if stage.norm is not None:
            copy_parameter(
                stage.norm.weight,
                checkpoint.tensor("model.norm.weight"),
                device=device,
            )
        if stage.lm_head is not None:
            if config.tie_word_embeddings:
                raise RuntimeError(
                    "Qwen3-30B-A3B is expected to use an untied LM head"
                )
            copy_parameter(
                stage.lm_head.weight,
                checkpoint.tensor("lm_head.weight"),
                device=device,
            )
            log("loaded final norm and LM head")
