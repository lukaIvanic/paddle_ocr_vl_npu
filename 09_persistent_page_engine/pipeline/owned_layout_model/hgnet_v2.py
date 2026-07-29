# Copyright 2025 Baidu Inc and The HuggingFace Inc. team.
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
"""Inference-only HGNetV2-L backbone owned by this project.

The module topology intentionally matches Transformers 5.5.4 so the official
PP-DocLayoutV3 safetensors checkpoint loads without key translation. Framework
configuration, training heads, and Transformers model utilities are omitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import HGNetV2Config


def _activation(name: str | None) -> nn.Module:
    if name is None:
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"unsupported HGNetV2 activation: {name}")


class HGNetV2LearnableAffineBlock(nn.Module):
    def __init__(
        self,
        scale_value: float = 1.0,
        bias_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]))
        self.bias = nn.Parameter(torch.tensor([bias_value]))

    def forward(self, hidden_state: Tensor) -> Tensor:
        return self.scale * hidden_state + self.bias


class HGNetV2ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: str | None = "relu",
        use_learnable_affine_block: bool = False,
    ) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = _activation(activation)
        self.lab = (
            HGNetV2LearnableAffineBlock()
            if activation and use_learnable_affine_block
            else nn.Identity()
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        hidden_state = self.convolution(input_tensor)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return self.lab(hidden_state)


class HGNetV2ConvLayerLight(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_learnable_affine_block: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = HGNetV2ConvLayer(
            in_channels,
            out_channels,
            kernel_size=1,
            activation=None,
            use_learnable_affine_block=use_learnable_affine_block,
        )
        self.conv2 = HGNetV2ConvLayer(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            use_learnable_affine_block=use_learnable_affine_block,
        )

    def forward(self, hidden_state: Tensor) -> Tensor:
        return self.conv2(self.conv1(hidden_state))


class HGNetV2Embeddings(nn.Module):
    def __init__(self, config: HGNetV2Config) -> None:
        super().__init__()
        self.stem1 = HGNetV2ConvLayer(
            config.stem_channels[0],
            config.stem_channels[1],
            kernel_size=3,
            stride=config.stem_strides[0],
            activation=config.hidden_act,
        )
        self.stem2a = HGNetV2ConvLayer(
            config.stem_channels[1],
            config.stem_channels[1] // 2,
            kernel_size=2,
            stride=config.stem_strides[1],
            activation=config.hidden_act,
        )
        self.stem2b = HGNetV2ConvLayer(
            config.stem_channels[1] // 2,
            config.stem_channels[1],
            kernel_size=2,
            stride=config.stem_strides[2],
            activation=config.hidden_act,
        )
        self.stem3 = HGNetV2ConvLayer(
            config.stem_channels[1] * 2,
            config.stem_channels[1],
            kernel_size=3,
            stride=config.stem_strides[3],
            activation=config.hidden_act,
        )
        self.stem4 = HGNetV2ConvLayer(
            config.stem_channels[1],
            config.stem_channels[2],
            kernel_size=1,
            stride=config.stem_strides[4],
            activation=config.hidden_act,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)
        self.num_channels = config.num_channels

    def forward(self, pixel_values: Tensor) -> Tensor:
        if pixel_values.shape[1] != self.num_channels:
            raise ValueError(
                "pixel channel count does not match the HGNetV2 configuration"
            )
        embedding = self.stem1(pixel_values)
        embedding = F.pad(embedding, (0, 1, 0, 1))
        stem_2a = self.stem2a(embedding)
        stem_2a = F.pad(stem_2a, (0, 1, 0, 1))
        stem_2a = self.stem2b(stem_2a)
        pooled = self.pool(embedding)
        embedding = torch.cat([pooled, stem_2a], dim=1)
        return self.stem4(self.stem3(embedding))


class HGNetV2BasicLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        middle_channels: int,
        out_channels: int,
        layer_num: int,
        kernel_size: int = 3,
        residual: bool = False,
        light_block: bool = False,
        drop_path: float = 0.0,
        use_learnable_affine_block: bool = False,
    ) -> None:
        super().__init__()
        self.residual = residual
        layers = []
        for index in range(layer_num):
            layer_in = in_channels if index == 0 else middle_channels
            layer_type = (
                HGNetV2ConvLayerLight if light_block else HGNetV2ConvLayer
            )
            layers.append(
                layer_type(
                    in_channels=layer_in,
                    out_channels=middle_channels,
                    kernel_size=kernel_size,
                    use_learnable_affine_block=use_learnable_affine_block,
                )
            )
        self.layers = nn.ModuleList(layers)
        total_channels = in_channels + layer_num * middle_channels
        self.aggregation = nn.Sequential(
            HGNetV2ConvLayer(
                total_channels,
                out_channels // 2,
                kernel_size=1,
                stride=1,
                use_learnable_affine_block=use_learnable_affine_block,
            ),
            HGNetV2ConvLayer(
                out_channels // 2,
                out_channels,
                kernel_size=1,
                stride=1,
                use_learnable_affine_block=use_learnable_affine_block,
            ),
        )
        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, hidden_state: Tensor) -> Tensor:
        identity = hidden_state
        outputs = [hidden_state]
        for layer in self.layers:
            hidden_state = layer(hidden_state)
            outputs.append(hidden_state)
        hidden_state = self.aggregation(torch.cat(outputs, dim=1))
        if self.residual:
            hidden_state = self.drop_path(hidden_state) + identity
        return hidden_state


class HGNetV2Stage(nn.Module):
    def __init__(
        self,
        config: HGNetV2Config,
        stage_index: int,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        in_channels = config.stage_in_channels[stage_index]
        out_channels = config.stage_out_channels[stage_index]
        if config.stage_downsample[stage_index]:
            self.downsample = HGNetV2ConvLayer(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=config.stage_downsample_strides[stage_index],
                groups=in_channels,
                activation=None,
            )
        else:
            self.downsample = nn.Identity()
        self.blocks = nn.ModuleList(
            [
                HGNetV2BasicLayer(
                    in_channels if index == 0 else out_channels,
                    config.stage_mid_channels[stage_index],
                    out_channels,
                    config.stage_numb_of_layers[stage_index],
                    residual=index != 0,
                    kernel_size=config.stage_kernel_size[stage_index],
                    light_block=config.stage_light_block[stage_index],
                    drop_path=drop_path,
                    use_learnable_affine_block=(
                        config.use_learnable_affine_block
                    ),
                )
                for index in range(config.stage_num_blocks[stage_index])
            ]
        )

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.downsample(hidden_state)
        for block in self.blocks:
            hidden_state = block(hidden_state)
        return hidden_state


class HGNetV2Encoder(nn.Module):
    def __init__(self, config: HGNetV2Config) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                HGNetV2Stage(config, index)
                for index in range(len(config.stage_in_channels))
            ]
        )

    def forward(self, hidden_state: Tensor) -> tuple[Tensor, ...]:
        hidden_states = []
        for stage in self.stages:
            hidden_states.append(hidden_state)
            hidden_state = stage(hidden_state)
        hidden_states.append(hidden_state)
        return tuple(hidden_states)


@dataclass(frozen=True)
class HGNetV2BackboneOutput:
    feature_maps: tuple[Tensor, ...]


class HGNetV2Backbone(nn.Module):
    def __init__(self, config: HGNetV2Config) -> None:
        super().__init__()
        self.config = config
        self.embedder = HGNetV2Embeddings(config)
        self.encoder = HGNetV2Encoder(config)
        self.stage_names = config.stage_names
        self.out_features = config.out_features
        self.channels = list(config.channels)

    def forward(self, pixel_values: Tensor) -> HGNetV2BackboneOutput:
        embedding = self.embedder(pixel_values)
        hidden_states = self.encoder(embedding)
        feature_maps = tuple(
            hidden_states[index]
            for index, stage in enumerate(self.stage_names)
            if stage in self.out_features
        )
        return HGNetV2BackboneOutput(feature_maps=feature_maps)
