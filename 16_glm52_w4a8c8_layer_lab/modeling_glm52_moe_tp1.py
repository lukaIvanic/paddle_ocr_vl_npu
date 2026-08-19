#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from modeling_glm52_layer import (
    GLM52Config,
    GLM52SharedExpert,
    GLM52W4A8Experts,
    ShardedSafetensorReader,
    npu_rms_norm,
)


class GLM52MoEMLPBlock(nn.Module):
    def __init__(
        self,
        *,
        config: GLM52Config,
        routed: GLM52W4A8Experts,
        shared: GLM52SharedExpert,
        norm_weight: torch.Tensor,
    ):
        super().__init__()
        self.config = config
        self.routed = routed
        self.shared = shared
        self.register_buffer("norm_weight", norm_weight.contiguous())

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
        fuse_gmm1_swiglu_quant: bool = True,
    ) -> "GLM52MoEMLPBlock":
        if layer_index < config.first_k_dense_replace:
            raise ValueError(f"layer {layer_index} is not a MoE layer")
        layer = f"model.layers.{layer_index}"
        routed = GLM52W4A8Experts.from_checkpoint(
            reader,
            layer_index,
            config,
            device=device,
            progress=progress,
            w4_weight_format=w4_weight_format,
            fuse_gmm1_swiglu_quant=fuse_gmm1_swiglu_quant,
        )
        return cls(
            config=config,
            routed=routed,
            shared=GLM52SharedExpert.from_checkpoint(
                reader, layer_index, device=device
            ),
            norm_weight=reader.tensor(
                layer + ".post_attention_layernorm.weight"
            ).to(device=device, dtype=torch.bfloat16),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states.clone()
        normalized = npu_rms_norm(
            hidden_states, self.norm_weight, self.config.rms_norm_eps
        )
        return residual + self.routed(normalized) + self.shared(normalized)


class GLM52MoEMLPStack(nn.Module):
    def __init__(
        self,
        blocks: list[GLM52MoEMLPBlock],
        *,
        first_layer: int,
        last_layer: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.first_layer = int(first_layer)
        self.last_layer = int(last_layer)
        self.config = blocks[0].config

    @classmethod
    def from_checkpoint(
        cls,
        model_dir: str | Path,
        *,
        first_layer: int,
        last_layer: int,
        device: torch.device,
        progress=None,
        w4_weight_format: str = "fractal_nz",
        fuse_gmm1_swiglu_quant: bool = True,
    ) -> "GLM52MoEMLPStack":
        if last_layer < first_layer:
            raise ValueError("last_layer must not be smaller than first_layer")
        config = GLM52Config.from_model_dir(model_dir)
        reader = ShardedSafetensorReader(model_dir)
        blocks = []
        for layer_index in range(first_layer, last_layer + 1):
            if progress is not None:
                progress(f"loading MoE MLP layer {layer_index}")
            blocks.append(
                GLM52MoEMLPBlock.from_checkpoint(
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
                    fuse_gmm1_swiglu_quant=fuse_gmm1_swiglu_quant,
                )
            )
            if progress is not None:
                progress(f"loaded MoE MLP layer {layer_index}")
        return cls(
            blocks,
            first_layer=first_layer,
            last_layer=last_layer,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states
