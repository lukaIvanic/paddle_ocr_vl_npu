# Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Independent eager PP-DocLayoutV3 inference implementation.

This is a deliberately narrow adaptation of the Apache-2.0 PP-DocLayoutV3
implementation distributed in Transformers 5.5.4. It retains the checkpoint's
module topology and inference math while removing Transformers model/config
framework code, training code, compilation wrappers, and kernel registries.
The image processor remains external for now.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor, nn

from .config import PPDocLayoutV3Config
from .hgnet_v2 import HGNetV2Backbone


def _activation(name: str | None) -> nn.Module:
    if name is None:
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"unsupported activation: {name}")


def _activation_function(name: str):
    if name == "relu":
        return F.relu
    if name == "silu":
        return F.silu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"unsupported activation: {name}")


@dataclass
class PPDocLayoutV3ForObjectDetectionOutput:
    logits: Tensor
    pred_boxes: Tensor
    order_logits: Tensor
    out_masks: Tensor
    last_hidden_state: Tensor | None = None
    intermediate_hidden_states: Tensor | None = None
    intermediate_logits: Tensor | None = None
    intermediate_reference_points: Tensor | None = None
    init_reference_points: Tensor | None = None
    enc_topk_logits: Tensor | None = None
    enc_topk_bboxes: Tensor | None = None
    enc_outputs_class: Tensor | None = None
    enc_outputs_coord_logits: Tensor | None = None


@dataclass
class _HybridEncoderOutput:
    last_hidden_state: list[Tensor]
    mask_feat: Tensor


@dataclass
class _DecoderOutput:
    last_hidden_state: Tensor
    intermediate_hidden_states: Tensor
    intermediate_logits: Tensor
    intermediate_reference_points: Tensor
    decoder_out_order_logits: Tensor
    decoder_out_masks: Tensor


@dataclass
class _CoreOutput:
    last_hidden_state: Tensor
    intermediate_hidden_states: Tensor
    intermediate_logits: Tensor
    intermediate_reference_points: Tensor
    out_order_logits: Tensor
    out_masks: Tensor
    init_reference_points: Tensor
    enc_topk_logits: Tensor
    enc_topk_bboxes: Tensor
    enc_outputs_class: Tensor
    enc_outputs_coord_logits: Tensor


class PPDocLayoutV3GlobalPointer(nn.Module):
    def __init__(self, config: PPDocLayoutV3Config) -> None:
        super().__init__()
        self.head_size = config.global_pointer_head_size
        self.dense = nn.Linear(config.d_model, self.head_size * 2)
        self.dropout = nn.Dropout(config.gp_dropout_value)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, sequence_length, _ = inputs.shape
        projected = self.dense(inputs).reshape(
            batch_size,
            sequence_length,
            2,
            self.head_size,
        )
        queries, keys = torch.unbind(self.dropout(projected), dim=2)
        logits = (queries @ keys.transpose(-2, -1)) / math.sqrt(
            self.head_size
        )
        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=logits.device,
            )
        ).bool()
        return logits.masked_fill(mask.unsqueeze(0), -1e4)


class MultiScaleDeformableAttention(nn.Module):
    def forward(
        self,
        value: Tensor,
        value_spatial_shapes: Tensor,
        value_spatial_shapes_list: list[tuple[int, int]],
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        im2col_step: int,
    ) -> Tensor:
        del value_spatial_shapes, level_start_index, im2col_step
        batch_size, _, num_heads, hidden_dim = value.shape
        _, num_queries, _, num_levels, num_points, _ = (
            sampling_locations.shape
        )
        value_list = value.split(
            [
                height * width
                for height, width in value_spatial_shapes_list
            ],
            dim=1,
        )
        sampling_grids = 2 * sampling_locations - 1
        sampled = []
        for level_id, (height, width) in enumerate(
            value_spatial_shapes_list
        ):
            level_value = (
                value_list[level_id]
                .flatten(2)
                .transpose(1, 2)
                .reshape(
                    batch_size * num_heads,
                    hidden_dim,
                    height,
                    width,
                )
            )
            level_grid = (
                sampling_grids[:, :, :, level_id]
                .transpose(1, 2)
                .flatten(0, 1)
            )
            sampled.append(
                F.grid_sample(
                    level_value,
                    level_grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
            )
        attention_weights = attention_weights.transpose(1, 2).reshape(
            batch_size * num_heads,
            1,
            num_queries,
            num_levels * num_points,
        )
        output = (
            (
                torch.stack(sampled, dim=-2).flatten(-2)
                * attention_weights
            )
            .sum(-1)
            .view(
                batch_size,
                num_heads * hidden_dim,
                num_queries,
            )
        )
        return output.transpose(1, 2).contiguous()


class PPDocLayoutV3MultiscaleDeformableAttention(nn.Module):
    def __init__(
        self,
        config: PPDocLayoutV3Config,
        num_heads: int,
        n_points: int,
    ) -> None:
        super().__init__()
        if config.d_model % num_heads:
            raise ValueError("d_model must be divisible by attention heads")
        self.attn = MultiScaleDeformableAttention()
        self.im2col_step = 64
        self.d_model = config.d_model
        self.n_levels = config.num_feature_levels
        self.n_heads = num_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(
            config.d_model,
            num_heads * self.n_levels * n_points * 2,
        )
        self.attention_weights = nn.Linear(
            config.d_model,
            num_heads * self.n_levels * n_points,
        )
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        position_embeddings: Tensor | None = None,
        reference_points: Tensor | None = None,
        spatial_shapes: Tensor | None = None,
        spatial_shapes_list: list[tuple[int, int]] | None = None,
        level_start_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if encoder_hidden_states is None:
            raise ValueError("encoder hidden states are required")
        if reference_points is None or spatial_shapes is None:
            raise ValueError("reference points and spatial shapes are required")
        if spatial_shapes_list is None or level_start_index is None:
            raise ValueError("spatial shape metadata is required")
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        batch_size, num_queries, _ = hidden_states.shape
        _, sequence_length, _ = encoder_hidden_states.shape
        if (
            sum(height * width for height, width in spatial_shapes_list)
            != sequence_length
        ):
            raise ValueError(
                "spatial shapes do not match encoder sequence length"
            )
        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = value.masked_fill(
                ~attention_mask[..., None],
                float(0),
            )
        value = value.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.d_model // self.n_heads,
        )
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size,
            num_queries,
            self.n_heads,
            self.n_levels,
            self.n_points,
            2,
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size,
            num_queries,
            self.n_heads,
            self.n_levels * self.n_points,
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch_size,
            num_queries,
            self.n_heads,
            self.n_levels,
            self.n_points,
        )
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]],
                -1,
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets
                / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError("reference point width must be two or four")
        output = self.attn(
            value,
            spatial_shapes,
            spatial_shapes_list,
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )
        return self.output_proj(output), attention_weights


class PPDocLayoutV3MLPPredictionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(source, target)
            for source, target in zip(
                [input_dim] + hidden,
                hidden + [output_dim],
            )
        )

    def forward(self, hidden_state: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            hidden_state = (
                F.relu(layer(hidden_state))
                if index < self.num_layers - 1
                else layer(hidden_state)
            )
        return hidden_state


class PPDocLayoutV3ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: str | None = "relu",
    ) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = _activation(activation)

    def forward(self, input_tensor: Tensor) -> Tensor:
        hidden_state = self.convolution(input_tensor)
        hidden_state = self.normalization(hidden_state)
        return self.activation(hidden_state)


class PPDocLayoutV3ScaleHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feature_channels: int,
        fpn_stride: int,
        base_stride: int,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        head_length = max(
            1,
            int(np.log2(fpn_stride) - np.log2(base_stride)),
        )
        layers = []
        for index in range(head_length):
            layers.append(
                PPDocLayoutV3ConvLayer(
                    in_channels if index == 0 else feature_channels,
                    feature_channels,
                    3,
                    1,
                    "silu",
                )
            )
            if fpn_stride != base_stride:
                layers.append(
                    nn.Upsample(
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=align_corners,
                    )
                )
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_state: Tensor) -> Tensor:
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class PPDocLayoutV3MaskFeatFPN(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        fpn_strides: list[int],
        feature_channels: int,
        dropout_ratio: float,
        out_channels: int,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        reorder_index = np.argsort(fpn_strides, axis=0).tolist()
        in_channels = [in_channels[index] for index in reorder_index]
        fpn_strides = [fpn_strides[index] for index in reorder_index]
        self.reorder_index = reorder_index
        self.fpn_strides = fpn_strides
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        self.scale_heads = nn.ModuleList(
            [
                PPDocLayoutV3ScaleHead(
                    in_channels=in_channels[index],
                    feature_channels=feature_channels,
                    fpn_stride=fpn_strides[index],
                    base_stride=fpn_strides[0],
                    align_corners=align_corners,
                )
                for index in range(len(fpn_strides))
            ]
        )
        self.output_conv = PPDocLayoutV3ConvLayer(
            feature_channels,
            out_channels,
            3,
            1,
            "silu",
        )

    def forward(self, inputs: list[Tensor]) -> Tensor:
        ordered = [inputs[index] for index in self.reorder_index]
        output = self.scale_heads[0](ordered[0])
        for index in range(1, len(self.fpn_strides)):
            output = output + F.interpolate(
                self.scale_heads[index](ordered[index]),
                size=output.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        if self.dropout_ratio > 0:
            output = self.dropout(output)
        return self.output_conv(output)


class PPDocLayoutV3EncoderMaskOutput(nn.Module):
    def __init__(self, in_channels: int, num_prototypes: int) -> None:
        super().__init__()
        self.base_conv = PPDocLayoutV3ConvLayer(
            in_channels,
            in_channels,
            3,
            1,
            "silu",
        )
        self.conv = nn.Conv2d(in_channels, num_prototypes, kernel_size=1)

    def forward(self, hidden_state: Tensor) -> Tensor:
        return self.conv(self.base_conv(hidden_state))


class PPDocLayoutV3MLP(nn.Module):
    def __init__(
        self,
        config: PPDocLayoutV3Config,
        hidden_size: int,
        intermediate_size: int,
        activation_function: str,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation_fn = _activation_function(activation_function)
        self.activation_dropout = config.activation_dropout
        self.dropout = config.dropout

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(
            hidden_states,
            p=self.activation_dropout,
            training=self.training,
        )
        hidden_states = self.fc2(hidden_states)
        return F.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )


def eager_attention_forward(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
) -> tuple[Tensor, Tensor]:
    if scaling is None:
        scaling = query.size(-1) ** -0.5
    attention_weights = (
        torch.matmul(query, key.transpose(2, 3)) * scaling
    )
    if attention_mask is not None:
        attention_weights = attention_weights + attention_mask
    attention_weights = F.softmax(attention_weights, dim=-1)
    attention_weights = F.dropout(
        attention_weights,
        p=dropout,
        training=module.training,
    )
    output = torch.matmul(attention_weights, value)
    return output.transpose(1, 2).contiguous(), attention_weights


class PPDocLayoutV3SelfAttention(nn.Module):
    def __init__(
        self,
        config: PPDocLayoutV3Config,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = dropout
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        # Keep the checkpoint's original ``out_proj`` spelling. Newer
        # Transformers releases rename this key during loading.
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_embeddings: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_key_input = (
            hidden_states + position_embeddings
            if position_embeddings is not None
            else hidden_states
        )
        query = self.q_proj(query_key_input).view(hidden_shape).transpose(1, 2)
        key = self.k_proj(query_key_input).view(hidden_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        output, weights = eager_attention_forward(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=(
                0.0 if not self.training else self.attention_dropout
            ),
            scaling=self.scaling,
        )
        output = output.reshape(*input_shape, -1).contiguous()
        return self.out_proj(output), weights


class PPDocLayoutV3ConvNormLayer(nn.Module):
    def __init__(
        self,
        config: PPDocLayoutV3Config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int | None = None,
        activation: str | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(
                (kernel_size - 1) // 2
                if padding is None
                else padding
            ),
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels, config.batch_norm_eps)
        self.activation = _activation(activation)

    def forward(self, hidden_state: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(hidden_state)))


class PPDocLayoutV3EncoderLayer(nn.Module):
    def __init__(self, config: PPDocLayoutV3Config) -> None:
        super().__init__()
        self.normalize_before = config.normalize_before
        self.hidden_size = config.encoder_hidden_dim
        self.self_attn = PPDocLayoutV3SelfAttention(
            config,
            self.hidden_size,
            config.num_attention_heads,
            config.dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(
            self.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.dropout = config.dropout
        # These projections intentionally remain direct children. The
        # published checkpoint predates Transformers' later ``mlp.fc*`` key
        # migration.
        self.fc1 = nn.Linear(
            self.hidden_size,
            config.encoder_ffn_dim,
        )
        self.fc2 = nn.Linear(
            config.encoder_ffn_dim,
            self.hidden_size,
        )
        self.activation_fn = _activation_function(
            config.encoder_activation_function
        )
        self.activation_dropout = config.activation_dropout
        self.final_layer_norm = nn.LayerNorm(
            self.hidden_size,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        spatial_position_embeddings: Tensor | None = None,
    ) -> Tensor:
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask,
            spatial_position_embeddings,
        )
        hidden_states = F.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(
            hidden_states,
            p=self.activation_dropout,
            training=self.training,
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class PPDocLayoutV3RepVggBlock(nn.Module):
    def __init__(self, config: PPDocLayoutV3Config) -> None:
        super().__init__()
        hidden_channels = int(
            config.encoder_hidden_dim * config.hidden_expansion
        )
        self.conv1 = PPDocLayoutV3ConvNormLayer(
            config,
            hidden_channels,
            hidden_channels,
            3,
            1,
            padding=1,
        )
        self.conv2 = PPDocLayoutV3ConvNormLayer(
            config,
            hidden_channels,
            hidden_channels,
            1,
            1,
            padding=0,
        )
        self.activation = _activation(config.activation_function)

    def forward(self, hidden_state: Tensor) -> Tensor:
        return self.activation(
            self.conv1(hidden_state) + self.conv2(hidden_state)
        )


class PPDocLayoutV3CSPRepLayer(nn.Module):
    def __init__(self, config: PPDocLayoutV3Config) -> None:
        super().__init__()
        in_channels = config.encoder_hidden_dim * 2
        out_channels = config.encoder_hidden_dim
        hidden_channels = int(
            out_channels * config.hidden_expansion
        )
        self.conv1 = PPDocLayoutV3ConvNormLayer(
            config,
            in_channels,
            hidden_channels,
            1,
            1,
            activation=config.activation_function,
        )
        self.conv2 = PPDocLayoutV3ConvNormLayer(
            config,
            in_channels,
            hidden_channels,
            1,
            1,
            activation=config.activation_function,
        )
        self.bottlenecks = nn.Sequential(
            *[
                PPDocLayoutV3RepVggBlock(config)
                for _ in range(3)
            ]
        )
        self.conv3 = (
            PPDocLayoutV3ConvNormLayer(
                config,
                hidden_channels,
                out_channels,
                1,
                1,
                activation=config.activation_function,
            )
            if hidden_channels != out_channels
            else nn.Identity()
        )

    def forward(self, hidden_state: Tensor) -> Tensor:
        first = self.bottlenecks(self.conv1(hidden_state))
        second = self.conv2(hidden_state)
        return self.conv3(first + second)


class PPDocLayoutV3SinePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        temperature: int = 10_000,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

    def forward(
        self,
        width: int,
        height: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        grid_w = torch.arange(width, device=device).to(dtype)
        grid_h = torch.arange(height, device=device).to(dtype)
        grid_w, grid_h = torch.meshgrid(
            grid_w,
            grid_h,
            indexing="xy",
        )
        if self.embed_dim % 4:
            raise ValueError("position embedding dimension must divide by four")
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, device=device).to(dtype) / pos_dim
        omega = 1.0 / (self.temperature**omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat(
            [
                out_h.sin(),
                out_h.cos(),
                out_w.sin(),
                out_w.cos(),
            ],
            dim=1,
        )[None, :, :]


class PPDocLayoutV3AIFILayer(nn.Module):
    def __init__(self, config: PPDocLayoutV3Config) -> None:
        super().__init__()
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.eval_size = config.eval_size
        self.position_embedding = PPDocLayoutV3SinePositionEmbedding(
            embed_dim=self.encoder_hidden_dim,
            temperature=config.positional_encoding_temperature,
        )
        self.layers = nn.ModuleList(
            [
                PPDocLayoutV3EncoderLayer(config)
                for _ in range(config.encoder_layers)
            ]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size = hidden_states.shape[0]
        height, width = hidden_states.shape[2:]
        hidden_states = hidden_states.flatten(2).permute(0, 2, 1)
        position_embedding = (
            self.position_embedding(
                width,
                height,
                hidden_states.device,
                hidden_states.dtype,
            )
            if self.training or self.eval_size is None
            else None
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                spatial_position_embeddings=position_embedding,
            )
        return (
            hidden_states.permute(0, 2, 1)
            .reshape(
                batch_size,
                self.encoder_hidden_dim,
                height,
                width,
            )
            .contiguous()
        )


class PPDocLayoutV3HybridEncoder(nn.Module):
    def __init__(self, config: PPDocLayoutV3Config) -> None:
        super().__init__()
        self.config = config
        self.in_channels = config.encoder_in_channels
        self.feat_strides = config.feat_strides
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.encode_proj_layers = config.encode_proj_layers
        self.out_channels = [
            self.encoder_hidden_dim for _ in self.in_channels
        ]
        self.out_strides = self.feat_strides
        self.num_fpn_stages = len(self.in_channels) - 1
        self.num_pan_stages = len(self.in_channels) - 1
        # ``encoder`` is the key path in the official safetensors checkpoint.
        self.encoder = nn.ModuleList(
            [
                PPDocLayoutV3AIFILayer(config)
                for _ in self.encode_proj_layers
            ]
        )
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(self.num_fpn_stages):
            self.lateral_convs.append(
                PPDocLayoutV3ConvNormLayer(
                    config,
                    self.encoder_hidden_dim,
                    self.encoder_hidden_dim,
                    1,
                    1,
                    activation=config.activation_function,
                )
            )
            self.fpn_blocks.append(PPDocLayoutV3CSPRepLayer(config))
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(self.num_pan_stages):
            self.downsample_convs.append(
                PPDocLayoutV3ConvNormLayer(
                    config,
                    self.encoder_hidden_dim,
                    self.encoder_hidden_dim,
                    3,
                    2,
                    activation=config.activation_function,
                )
            )
            self.pan_blocks.append(PPDocLayoutV3CSPRepLayer(config))

        mask_channels = config.mask_feature_channels
        self.mask_feature_head = PPDocLayoutV3MaskFeatFPN(
            [self.encoder_hidden_dim] * len(config.feat_strides),
            list(config.feat_strides),
            feature_channels=mask_channels[0],
            dropout_ratio=0.0,
            out_channels=mask_channels[1],
        )
        self.encoder_mask_lateral = PPDocLayoutV3ConvLayer(
            config.x4_feat_dim,
            mask_channels[1],
            3,
            1,
            "silu",
        )
        self.encoder_mask_output = PPDocLayoutV3EncoderMaskOutput(
            mask_channels[1],
            config.num_prototypes,
        )

    def forward(
        self,
        inputs_embeds: list[Tensor],
        x4_feat: tuple[Tensor, Tensor],
    ) -> _HybridEncoderOutput:
        feature_maps = inputs_embeds
        if self.config.encoder_layers > 0:
            for index, projected_index in enumerate(
                self.encode_proj_layers
            ):
                feature_maps[projected_index] = self.encoder[index](
                    feature_maps[projected_index]
                )

        fpn_feature_maps = [feature_maps[-1]]
        for index, (lateral, fpn_block) in enumerate(
            zip(self.lateral_convs, self.fpn_blocks)
        ):
            backbone_feature = feature_maps[
                self.num_fpn_stages - index - 1
            ]
            top_feature = lateral(fpn_feature_maps[-1])
            fpn_feature_maps[-1] = top_feature
            top_feature = F.interpolate(
                top_feature,
                scale_factor=2.0,
                mode="nearest",
            )
            fpn_feature_maps.append(
                fpn_block(
                    torch.concat(
                        [top_feature, backbone_feature],
                        dim=1,
                    )
                )
            )
        fpn_feature_maps.reverse()

        pan_feature_maps = [fpn_feature_maps[0]]
        for index, (downsample, pan_block) in enumerate(
            zip(self.downsample_convs, self.pan_blocks)
        ):
            downsampled = downsample(pan_feature_maps[-1])
            pan_feature_maps.append(
                pan_block(
                    torch.concat(
                        [downsampled, fpn_feature_maps[index + 1]],
                        dim=1,
                    )
                )
            )

        mask_feat = self.mask_feature_head(pan_feature_maps)
        mask_feat = F.interpolate(
            mask_feat,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        mask_feat = mask_feat + self.encoder_mask_lateral(x4_feat[0])
        mask_feat = self.encoder_mask_output(mask_feat)
        return _HybridEncoderOutput(
            last_hidden_state=pan_feature_maps,
            mask_feat=mask_feat,
        )


class PPDocLayoutV3DecoderLayer(nn.Module):
    def __init__(self, config: PPDocLayoutV3Config) -> None:
        super().__init__()
        self.hidden_size = config.d_model
        self.self_attn = PPDocLayoutV3SelfAttention(
            config,
            self.hidden_size,
            config.decoder_attention_heads,
            config.attention_dropout,
        )
        self.dropout = config.dropout
        self.self_attn_layer_norm = nn.LayerNorm(
            self.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.encoder_attn = (
            PPDocLayoutV3MultiscaleDeformableAttention(
                config,
                config.decoder_attention_heads,
                config.decoder_n_points,
            )
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(
            self.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.fc1 = nn.Linear(
            self.hidden_size,
            config.decoder_ffn_dim,
        )
        self.fc2 = nn.Linear(
            config.decoder_ffn_dim,
            self.hidden_size,
        )
        self.activation_fn = _activation_function(
            config.decoder_activation_function
        )
        self.activation_dropout = config.activation_dropout
        self.final_layer_norm = nn.LayerNorm(
            self.hidden_size,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        hidden_states: Tensor,
        object_queries_position_embeddings: Tensor | None,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        spatial_shapes_list: list[tuple[int, int]],
        level_start_index: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor | None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states,
            encoder_attention_mask,
            object_queries_position_embeddings,
        )
        hidden_states = F.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )
        hidden_states = self.self_attn_layer_norm(
            residual + hidden_states
        )
        residual = hidden_states
        hidden_states, _ = self.encoder_attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=object_queries_position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
        )
        hidden_states = F.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )
        hidden_states = self.encoder_attn_layer_norm(
            residual + hidden_states
        )
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(
            hidden_states,
            p=self.activation_dropout,
            training=self.training,
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )
        return self.final_layer_norm(residual + hidden_states)


def inverse_sigmoid(hidden_state: Tensor, eps: float = 1e-5) -> Tensor:
    hidden_state = hidden_state.clamp(min=0, max=1)
    numerator = hidden_state.clamp(min=eps)
    denominator = (1 - hidden_state).clamp(min=eps)
    return torch.log(numerator / denominator)


class PPDocLayoutV3Decoder(nn.Module):
    """Decoder that computes inference heads only after the final layer."""

    def __init__(self, config: PPDocLayoutV3Config) -> None:
        super().__init__()
        self.dropout = config.dropout
        self.layers = nn.ModuleList(
            [
                PPDocLayoutV3DecoderLayer(config)
                for _ in range(config.decoder_layers)
            ]
        )
        self.query_pos_head = PPDocLayoutV3MLPPredictionHead(
            4,
            2 * config.d_model,
            config.d_model,
            num_layers=2,
        )
        self.bbox_embed: nn.Module | None = None
        self.class_embed: nn.Module | None = None
        self.num_queries = config.num_queries

    def forward(
        self,
        inputs_embeds: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor | None,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        spatial_shapes_list: list[tuple[int, int]],
        level_start_index: Tensor,
        order_head: nn.ModuleList,
        global_pointer: nn.Module,
        mask_query_head: nn.Module,
        norm: nn.Module,
        mask_feat: Tensor,
    ) -> _DecoderOutput:
        hidden_states = inputs_embeds
        reference_points = F.sigmoid(reference_points)
        final_index = -1
        for index, decoder_layer in enumerate(self.layers):
            final_index = index
            position_embeddings = self.query_pos_head(reference_points)
            hidden_states = decoder_layer(
                hidden_states,
                object_queries_position_embeddings=position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                reference_points=reference_points.unsqueeze(2),
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                encoder_attention_mask=encoder_attention_mask,
            )
            if self.bbox_embed is not None:
                predicted_corners = self.bbox_embed(hidden_states)
                reference_points = F.sigmoid(
                    predicted_corners
                    + inverse_sigmoid(reference_points)
                ).detach()

        out_query = norm(hidden_states)
        mask_query_embed = mask_query_head(out_query)
        batch_size, mask_dim, _ = mask_query_embed.shape
        _, _, mask_height, mask_width = mask_feat.shape
        out_mask = torch.bmm(
            mask_query_embed,
            mask_feat.flatten(start_dim=2),
        ).reshape(
            batch_size,
            mask_dim,
            mask_height,
            mask_width,
        )
        if self.class_embed is None:
            raise RuntimeError("decoder classification head is missing")
        logits = self.class_embed(out_query)
        valid_query = out_query[:, -self.num_queries :]
        order_logits = global_pointer(
            order_head[final_index](valid_query)
        )
        return _DecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=hidden_states.unsqueeze(1),
            intermediate_logits=logits.unsqueeze(1),
            intermediate_reference_points=reference_points.unsqueeze(1),
            decoder_out_order_logits=order_logits.unsqueeze(1),
            decoder_out_masks=out_mask.unsqueeze(1),
        )


class PPDocLayoutV3FrozenBatchNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(channels))
        self.register_buffer("bias", torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, hidden_state: Tensor) -> Tensor:
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        scale = weight * (running_var + 1e-5).rsqrt()
        return hidden_state * scale + (bias - running_mean * scale)


def _replace_batch_norm(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            replacement = PPDocLayoutV3FrozenBatchNorm2d(
                child.num_features
            )
            module._modules[name] = replacement
            child = replacement
        if any(child.children()):
            _replace_batch_norm(child)


class PPDocLayoutV3ConvEncoder(nn.Module):
    def __init__(self, config: PPDocLayoutV3Config) -> None:
        super().__init__()
        backbone = HGNetV2Backbone(config.backbone_config)
        if config.freeze_backbone_batch_norms:
            _replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = self.model.channels

    def forward(
        self,
        pixel_values: Tensor,
        pixel_mask: Tensor,
    ) -> list[tuple[Tensor, Tensor]]:
        features = self.model(pixel_values).feature_maps
        outputs = []
        for feature_map in features:
            mask = F.interpolate(
                pixel_mask[None].float(),
                size=feature_map.shape[-2:],
            ).to(torch.bool)[0]
            outputs.append((feature_map, mask))
        return outputs


def mask_to_box_coordinate(mask: Tensor, dtype: torch.dtype) -> Tensor:
    mask = mask.bool()
    height, width = mask.shape[-2:]
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=mask.device),
        torch.arange(width, device=mask.device),
        indexing="ij",
    )
    x_coords = x_coords.to(dtype)
    y_coords = y_coords.to(dtype)
    x_masked = x_coords * mask
    x_max = x_masked.flatten(start_dim=-2).max(dim=-1).values + 1
    maximum = torch.tensor(
        torch.finfo(dtype).max,
        device=mask.device,
        dtype=dtype,
    )
    x_min = (
        torch.where(mask, x_masked, maximum)
        .flatten(start_dim=-2)
        .min(dim=-1)
        .values
    )
    y_masked = y_coords * mask
    y_max = y_masked.flatten(start_dim=-2).max(dim=-1).values + 1
    y_min = (
        torch.where(mask, y_masked, maximum)
        .flatten(start_dim=-2)
        .min(dim=-1)
        .values
    )
    boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    boxes = boxes * torch.any(mask, dim=(-2, -1)).unsqueeze(-1)
    normalizer = torch.tensor(
        [width, height, width, height],
        device=mask.device,
        dtype=dtype,
    )
    x_min, y_min, x_max, y_max = (boxes / normalizer).unbind(dim=-1)
    return torch.stack(
        [
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            x_max - x_min,
            y_max - y_min,
        ],
        dim=-1,
    )


class PPDocLayoutV3Model(nn.Module):
    def __init__(self, config: PPDocLayoutV3Config) -> None:
        super().__init__()
        self.config = config
        self.backbone = PPDocLayoutV3ConvEncoder(config)
        channel_sizes = self.backbone.intermediate_channel_sizes
        encoder_projections = [
            nn.Sequential(
                nn.Conv2d(
                    channels,
                    config.encoder_hidden_dim,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(config.encoder_hidden_dim),
            )
            for channels in channel_sizes
        ]
        self.encoder_input_proj = nn.ModuleList(
            encoder_projections[1:]
        )
        self.encoder = PPDocLayoutV3HybridEncoder(config)
        # The wrapper checkpoint stores a 25-row inference embedding.
        self.denoising_class_embed = nn.Embedding(
            config.num_labels,
            config.d_model,
        )
        if config.learn_initial_query:
            self.weight_embedding = nn.Embedding(
                config.num_queries,
                config.d_model,
            )
        self.enc_output = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
        )
        self.enc_score_head = nn.Linear(
            config.d_model,
            config.num_labels,
        )
        self.enc_bbox_head = PPDocLayoutV3MLPPredictionHead(
            config.d_model,
            config.d_model,
            4,
            num_layers=3,
        )

        decoder_projections = []
        in_channels = config.decoder_in_channels[-1]
        for in_channels in config.decoder_in_channels:
            decoder_projections.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        config.d_model,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(
                        config.d_model,
                        config.batch_norm_eps,
                    ),
                )
            )
        for _ in range(
            config.num_feature_levels
            - len(config.decoder_in_channels)
        ):
            decoder_projections.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        config.d_model,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(
                        config.d_model,
                        config.batch_norm_eps,
                    ),
                )
            )
            in_channels = config.d_model
        self.decoder_input_proj = nn.ModuleList(decoder_projections)
        self.decoder = PPDocLayoutV3Decoder(config)
        self.decoder_order_head = nn.ModuleList(
            [
                nn.Linear(config.d_model, config.d_model)
                for _ in range(config.decoder_layers)
            ]
        )
        self.decoder_global_pointer = PPDocLayoutV3GlobalPointer(config)
        self.decoder_norm = nn.LayerNorm(
            config.d_model,
            eps=config.layer_norm_eps,
        )
        # Match Transformers' tied checkpoint aliases. These are the same
        # objects, so loading the encoder path also loads decoder inference.
        self.decoder.class_embed = self.enc_score_head
        self.decoder.bbox_embed = self.enc_bbox_head
        self.mask_enhanced = config.mask_enhanced
        self.mask_query_head = PPDocLayoutV3MLPPredictionHead(
            config.d_model,
            config.d_model,
            config.num_prototypes,
            num_layers=3,
        )

    def generate_anchors(
        self,
        spatial_shapes: tuple[tuple[int, int], ...],
        grid_size: float = 0.05,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=device).to(dtype),
                torch.arange(width, device=device).to(dtype),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], -1).unsqueeze(0) + 0.5
            grid_xy[..., 0] /= width
            grid_xy[..., 1] /= height
            dimensions = (
                torch.ones_like(grid_xy)
                * grid_size
                * (2.0**level)
            )
            anchors.append(
                torch.concat([grid_xy, dimensions], -1).reshape(
                    -1,
                    height * width,
                    4,
                )
            )
        anchors_tensor = torch.concat(anchors, 1)
        valid_mask = (
            (anchors_tensor > 1e-2)
            * (anchors_tensor < 1 - 1e-2)
        ).all(-1, keepdim=True)
        anchors_tensor = torch.log(
            anchors_tensor / (1 - anchors_tensor)
        )
        maximum = torch.tensor(
            torch.finfo(dtype).max,
            dtype=dtype,
            device=device,
        )
        anchors_tensor = torch.where(
            valid_mask,
            anchors_tensor,
            maximum,
        )
        return anchors_tensor, valid_mask

    def forward(
        self,
        pixel_values: Tensor,
        pixel_mask: Tensor | None = None,
    ) -> _CoreOutput:
        batch_size, _, height, width = pixel_values.shape
        device = pixel_values.device
        if pixel_mask is None:
            pixel_mask = torch.ones(
                (batch_size, height, width),
                device=device,
            )

        features = self.backbone(pixel_values, pixel_mask)
        x4_feat = features.pop(0)
        projected_features = [
            self.encoder_input_proj[level](source)
            for level, (source, _) in enumerate(features)
        ]
        encoder_outputs = self.encoder(
            projected_features,
            x4_feat,
        )
        sources = [
            self.decoder_input_proj[level](source)
            for level, source in enumerate(
                encoder_outputs.last_hidden_state
            )
        ]
        if self.config.num_feature_levels > len(sources):
            original_count = len(sources)
            final_source = encoder_outputs.last_hidden_state[-1]
            sources.append(
                self.decoder_input_proj[original_count](final_source)
            )
            for index in range(
                original_count + 1,
                self.config.num_feature_levels,
            ):
                sources.append(
                    self.decoder_input_proj[index](final_source)
                )

        source_flatten = []
        spatial_shapes_list: list[tuple[int, int]] = []
        for source in sources:
            source_height, source_width = source.shape[-2:]
            spatial_shapes_list.append(
                (source_height, source_width)
            )
            source_flatten.append(
                source.flatten(2).transpose(1, 2)
            )
        source_flatten_tensor = torch.cat(source_flatten, 1)
        # Construct metadata directly from Python shapes. This avoids the
        # IndexPut form that fails on some 310P torch_npu stacks.
        spatial_shapes = torch.tensor(
            spatial_shapes_list,
            device=device,
            dtype=torch.long,
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        anchors, valid_mask = self.generate_anchors(
            tuple(spatial_shapes_list),
            device=device,
            dtype=source_flatten_tensor.dtype,
        )
        memory = (
            valid_mask.to(source_flatten_tensor.dtype)
            * source_flatten_tensor
        )
        output_memory = self.enc_output(memory)
        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_logits = (
            self.enc_bbox_head(output_memory) + anchors
        )
        _, topk_indices = torch.topk(
            enc_outputs_class.max(-1).values,
            self.config.num_queries,
            dim=1,
        )
        reference_points_unact = enc_outputs_coord_logits.gather(
            dim=1,
            index=topk_indices.unsqueeze(-1).repeat(
                1,
                1,
                enc_outputs_coord_logits.shape[-1],
            ),
        )
        output_gather_index = topk_indices.unsqueeze(-1).repeat(
            1,
            1,
            output_memory.shape[-1],
        )
        target = output_memory.gather(
            dim=1,
            index=output_gather_index,
        )
        out_query = self.decoder_norm(target)
        mask_query_embed = self.mask_query_head(out_query)
        batch_size, mask_dim, _ = mask_query_embed.shape
        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        enc_topk_logits = enc_outputs_class.gather(
            dim=1,
            index=topk_indices.unsqueeze(-1).repeat(
                1,
                1,
                enc_outputs_class.shape[-1],
            ),
        )
        if self.config.learn_initial_query:
            target = self.weight_embedding.tile([batch_size, 1, 1])
        else:
            target = output_memory.gather(
                dim=1,
                index=output_gather_index,
            ).detach()

        if self.mask_enhanced:
            _, _, mask_height, mask_width = (
                encoder_outputs.mask_feat.shape
            )
            encoder_masks = torch.bmm(
                mask_query_embed,
                encoder_outputs.mask_feat.flatten(start_dim=2),
            ).reshape(
                batch_size,
                mask_dim,
                mask_height,
                mask_width,
            )
            reference_points = mask_to_box_coordinate(
                encoder_masks > 0,
                dtype=reference_points_unact.dtype,
            )
            reference_points_unact = inverse_sigmoid(
                reference_points
            )

        initial_reference_points = reference_points_unact.detach()
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten_tensor,
            encoder_attention_mask=None,
            reference_points=initial_reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            order_head=self.decoder_order_head,
            global_pointer=self.decoder_global_pointer,
            mask_query_head=self.mask_query_head,
            norm=self.decoder_norm,
            mask_feat=encoder_outputs.mask_feat,
        )
        return _CoreOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=(
                decoder_outputs.intermediate_hidden_states
            ),
            intermediate_logits=decoder_outputs.intermediate_logits,
            intermediate_reference_points=(
                decoder_outputs.intermediate_reference_points
            ),
            out_order_logits=(
                decoder_outputs.decoder_out_order_logits
            ),
            out_masks=decoder_outputs.decoder_out_masks,
            init_reference_points=initial_reference_points,
            enc_topk_logits=enc_topk_logits,
            enc_topk_bboxes=enc_topk_bboxes,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
        )


class OwnedPPDocLayoutV3ForObjectDetection(nn.Module):
    """Project-owned eager PP-DocLayoutV3 detector."""

    def __init__(self, config: PPDocLayoutV3Config) -> None:
        super().__init__()
        self.config = config
        self.model = PPDocLayoutV3Model(config)
        self.num_queries = config.num_queries
        self.load_report: dict[str, object] | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
    ) -> "OwnedPPDocLayoutV3ForObjectDetection":
        model_path = Path(model_dir).expanduser().resolve()
        config = PPDocLayoutV3Config.from_model_dir(model_path)
        model = cls(config)
        checkpoint_path = model_path / "model.safetensors"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"PP-DocLayoutV3 weights are missing: {checkpoint_path}"
            )
        checkpoint = load_file(str(checkpoint_path), device="cpu")
        expected = set(model.state_dict())
        provided = set(checkpoint)
        allowed_missing = {
            key
            for key in expected
            if key.endswith(".num_batches_tracked")
            or key.startswith("model.decoder.class_embed.")
            or key.startswith("model.decoder.bbox_embed.")
        }
        missing = expected - provided
        unexpected = provided - expected
        disallowed_missing = missing - allowed_missing
        if disallowed_missing or unexpected:
            raise RuntimeError(
                "owned PP-DocLayoutV3 checkpoint schema mismatch: "
                f"missing={sorted(disallowed_missing)[:20]}, "
                f"unexpected={sorted(unexpected)[:20]}"
            )
        incompatible = model.load_state_dict(checkpoint, strict=False)
        actual_missing = set(incompatible.missing_keys)
        actual_unexpected = set(incompatible.unexpected_keys)
        if actual_missing != missing or actual_unexpected != unexpected:
            raise RuntimeError(
                "owned PP-DocLayoutV3 loader coverage changed during load: "
                f"missing={sorted(actual_missing)}, "
                f"unexpected={sorted(actual_unexpected)}"
            )
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        checkpoint_elements = sum(
            tensor.numel() for tensor in checkpoint.values()
        )
        model.load_report = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_tensors": len(checkpoint),
            "checkpoint_elements": checkpoint_elements,
            "model_parameters": parameter_count,
            "allowed_missing": sorted(missing),
            "unexpected": [],
        }
        return model

    def forward(
        self,
        pixel_values: Tensor,
        pixel_mask: Tensor | None = None,
        **_: object,
    ) -> PPDocLayoutV3ForObjectDetectionOutput:
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
        )
        logits = outputs.intermediate_logits[:, -1]
        pred_boxes = outputs.intermediate_reference_points[:, -1]
        order_logits = outputs.out_order_logits[:, -1]
        out_masks = outputs.out_masks[:, -1]
        return PPDocLayoutV3ForObjectDetectionOutput(
            logits=logits,
            pred_boxes=pred_boxes,
            order_logits=order_logits,
            out_masks=out_masks,
            last_hidden_state=outputs.last_hidden_state,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_logits=outputs.intermediate_logits,
            intermediate_reference_points=(
                outputs.intermediate_reference_points
            ),
            init_reference_points=outputs.init_reference_points,
            enc_topk_logits=outputs.enc_topk_logits,
            enc_topk_bboxes=outputs.enc_topk_bboxes,
            enc_outputs_class=outputs.enc_outputs_class,
            enc_outputs_coord_logits=outputs.enc_outputs_coord_logits,
        )
