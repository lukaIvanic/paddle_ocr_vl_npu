#!/usr/bin/env python3
"""Production PromptFA PaddleOCR-VL vision format/alignment laboratory.

The measured boundary is the exact production ``VisionPrefillStage``: all 27
vision encoder layers, RoPE, PromptFA (including the 72 -> 80 head-dimension
padding), LayerNorms, residuals, Q/K/V/output projections, FC1/GELU/FC2, and
the final post-LayerNorm.  Only the synthetic shape inputs distinguish this
laboratory from a real crop's vision-transformer call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(HERE))

from paddleocr_vl.model.compile_utils import (  # noqa: E402
    TORCHAIR_EXECUTION_MODE,
    cache_key_part,
    import_torchair,
    short_file_hash,
    torch_npu_version_label,
    torchair_version_label,
)
from paddleocr_vl.model.modeling import (  # noqa: E402
    LocalPaddleOCRVLForConditionalGeneration,
)
from paddleocr_vl.model.vision_prefill import (  # noqa: E402
    VisionPrefillStage,
    apply_rotary_pos_emb_vision,
    get_vision_prompt_fa_layout,
    get_vision_prompt_fa_mask_sparse_mode,
    prepare_vision_prefill,
    prompt_flash_attention_call_head_dim,
    vision_prompt_flash_attention_bnsd,
)
from utils.timing import DeviceTimeline, synchronize  # noqa: E402
from vision_lab import DEFAULT_MODEL, _environment  # noqa: E402


FRACTAL_NZ = 29
DEFAULT_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_vision_matmul_lab"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "tmp/09_persistent_page_engine/vision_matmul_lab"
)
DEFAULT_PROFILE_ROOT = (
    REPO_ROOT
    / ".runtime_cache/09_persistent_page_engine_vision_matmul_profiles"
)
SEQUENCE_LENGTHS = (512, 2048)
BATCH_SIZES = (1, 4)
INTERMEDIATE_SIZES = (4304, 4352)
WEIGHT_FORMATS = ("native", "fractal_nz")
EXECUTIONS = ("raw_eager", "torchair")
ATTENTION_HEAD_PADDING_CHOICES = ("runtime", "weights")
ROTARY_IMPLEMENTATIONS = (
    "separate_manual",
    "joint_manual",
    "joint_inplace_partial",
)
PROFILE_METRIC_CHOICES = ("pipe", "memory", "l2", "memory_access")
LEGACY_SEPARATE_MANUAL_SOURCE_HASH = "b4144440d15e"
JOINT_MANUAL_SOURCE_HASH = "25d9a2dd1d39"
StageInputs = tuple[torch.Tensor, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=BATCH_SIZES,
        default=1,
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        choices=SEQUENCE_LENGTHS,
        required=True,
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        choices=INTERMEDIATE_SIZES,
        required=True,
    )
    parser.add_argument(
        "--weight-format",
        choices=WEIGHT_FORMATS,
        required=True,
    )
    parser.add_argument(
        "--execution",
        choices=EXECUTIONS,
        default="torchair",
    )
    parser.add_argument(
        "--attention-head-padding",
        choices=ATTENTION_HEAD_PADDING_CHOICES,
        default="runtime",
        help=(
            "runtime uses per-layer 72->80 pads and 80->72 slicing; "
            "weights zero-extends attention projections once to native "
            "80-wide heads."
        ),
    )
    parser.add_argument(
        "--rotary-implementation",
        choices=ROTARY_IMPLEMENTATIONS,
        default="separate_manual",
        help=(
            "separate_manual is the existing production-like D80 control; "
            "joint_manual applies identical FP32 math to one contiguous QK "
            "tensor; joint_inplace_partial uses the 910B-only vLLM-Ascend "
            "custom operator after a one-time half-to-interleave permutation."
        ),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument(
        "--allow-compile-if-missing",
        action="store_true",
        help="Permit creation of a missing TorchAir graph cache.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument(
        "--calls-per-sample",
        type=int,
        default=5,
        help=(
            "Full 27-layer replays inside one NPU-event sample. Values above "
            "one amortize host synchronization and launch overhead."
        ),
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--profile-metric",
        choices=PROFILE_METRIC_CHOICES,
        default="pipe",
        help=(
            "AI Core PMU family for this capture. Each family requires a "
            "separate profiler pass."
        ),
    )
    parser.add_argument("--profile-warmup-steps", type=int, default=1)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--parser-topn", type=int, default=200)
    args = parser.parse_args(argv)
    if args.batch_size == 4 and args.sequence_length != 512:
        parser.error("B4 is intentionally bounded to S512 in this experiment")
    if (
        args.rotary_implementation != "separate_manual"
        and args.attention_head_padding != "weights"
    ):
        parser.error(
            "joint rotary implementations require "
            "--attention-head-padding weights"
        )
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.samples <= 0 or args.calls_per_sample <= 0:
        parser.error("--samples and --calls-per-sample must be positive")
    if args.profile_warmup_steps <= 0 or args.profile_steps <= 0:
        parser.error("profiler step counts must be positive")
    return args


def _zero_extended_mlp(
    source_fc1: nn.Linear,
    source_fc2: nn.Linear,
    *,
    target_intermediate: int,
) -> tuple[nn.Linear, nn.Linear]:
    source_intermediate = int(source_fc1.out_features)
    hidden_size = int(source_fc1.in_features)
    if target_intermediate <= source_intermediate:
        raise ValueError(
            "the alignment experiment only supports zero-extension: "
            f"{source_intermediate} -> {target_intermediate}"
        )
    if int(source_fc2.in_features) != source_intermediate:
        raise ValueError("source FC1/FC2 intermediate dimensions disagree")
    if int(source_fc2.out_features) != hidden_size:
        raise ValueError("source FC2 hidden dimension disagrees with FC1")
    device = source_fc1.weight.device
    dtype = source_fc1.weight.dtype
    fc1 = nn.Linear(
        hidden_size,
        target_intermediate,
        bias=source_fc1.bias is not None,
        device=device,
        dtype=dtype,
    )
    fc2 = nn.Linear(
        target_intermediate,
        hidden_size,
        bias=source_fc2.bias is not None,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        fc1.weight.zero_()
        fc1.weight[:source_intermediate].copy_(source_fc1.weight)
        if fc1.bias is not None:
            fc1.bias.zero_()
            fc1.bias[:source_intermediate].copy_(source_fc1.bias)
        fc2.weight.zero_()
        fc2.weight[:, :source_intermediate].copy_(source_fc2.weight)
        if fc2.bias is not None:
            fc2.bias.copy_(source_fc2.bias)
    return fc1, fc2


def _zero_extend_model_mlp(
    model: nn.Module,
    *,
    target_intermediate: int,
) -> None:
    for layer in model.visual.vision_model.encoder.layers:
        fc1, fc2 = _zero_extended_mlp(
            layer.mlp.fc1,
            layer.mlp.fc2,
            target_intermediate=target_intermediate,
        )
        layer.mlp.fc1 = fc1
        layer.mlp.fc2 = fc2


def _copy_real_head_rows(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    num_heads: int,
    real_head_dim: int,
    padded_head_dim: int,
) -> None:
    half = real_head_dim // 2
    padded_half = padded_head_dim // 2
    for head in range(num_heads):
        source_start = head * real_head_dim
        destination_start = head * padded_head_dim
        destination[
            destination_start : destination_start + half
        ].copy_(source[source_start : source_start + half])
        destination[
            destination_start
            + padded_half : destination_start
            + padded_half
            + half
        ].copy_(
            source[
                source_start + half : source_start + real_head_dim
            ]
        )


def _copy_real_head_columns(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    num_heads: int,
    real_head_dim: int,
    padded_head_dim: int,
) -> None:
    half = real_head_dim // 2
    padded_half = padded_head_dim // 2
    for head in range(num_heads):
        source_start = head * real_head_dim
        destination_start = head * padded_head_dim
        destination[
            :,
            destination_start : destination_start + half,
        ].copy_(source[:, source_start : source_start + half])
        destination[
            :,
            destination_start
            + padded_half : destination_start
            + padded_half
            + half,
        ].copy_(
            source[
                :,
                source_start + half : source_start + real_head_dim,
            ]
        )


def _weight_padded_attention(
    attention: nn.Module,
) -> dict[str, int]:
    num_heads = int(attention.num_heads)
    real_head_dim = int(attention.head_dim)
    padded_head_dim = prompt_flash_attention_call_head_dim(real_head_dim)
    if padded_head_dim == real_head_dim:
        raise ValueError("attention head is already PromptFA-aligned")
    hidden_size = num_heads * real_head_dim
    padded_size = num_heads * padded_head_dim
    device = attention.q_proj.weight.device
    dtype = attention.q_proj.weight.dtype

    for name in ("q_proj", "k_proj", "v_proj"):
        source = getattr(attention, name)
        replacement = nn.Linear(
            int(source.in_features),
            padded_size,
            bias=source.bias is not None,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            replacement.weight.zero_()
            _copy_real_head_rows(
                replacement.weight,
                source.weight,
                num_heads=num_heads,
                real_head_dim=real_head_dim,
                padded_head_dim=padded_head_dim,
            )
            if replacement.bias is not None:
                replacement.bias.zero_()
                _copy_real_head_rows(
                    replacement.bias,
                    source.bias,
                    num_heads=num_heads,
                    real_head_dim=real_head_dim,
                    padded_head_dim=padded_head_dim,
                )
        setattr(attention, name, replacement)

    source_out = attention.out_proj
    replacement_out = nn.Linear(
        padded_size,
        int(source_out.out_features),
        bias=source_out.bias is not None,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        replacement_out.weight.zero_()
        _copy_real_head_columns(
            replacement_out.weight,
            source_out.weight,
            num_heads=num_heads,
            real_head_dim=real_head_dim,
            padded_head_dim=padded_head_dim,
        )
        if replacement_out.bias is not None:
            replacement_out.bias.copy_(source_out.bias)
    attention.out_proj = replacement_out
    attention._weight_padded_head_dim = padded_head_dim
    return {
        "num_heads": num_heads,
        "real_head_dim": real_head_dim,
        "padded_head_dim": padded_head_dim,
        "hidden_size": hidden_size,
        "padded_attention_size": padded_size,
    }


def _weight_pad_model_attention(model: nn.Module) -> dict[str, int]:
    metadata: dict[str, int] | None = None
    for layer in model.visual.vision_model.encoder.layers:
        current = _weight_padded_attention(layer.self_attn)
        if metadata is not None and current != metadata:
            raise RuntimeError("vision attention dimensions differ by layer")
        metadata = current
    if metadata is None:
        raise RuntimeError("vision model has no encoder layers")
    return metadata


def _weight_pad_rope(
    value: torch.Tensor,
    *,
    padded_head_dim: int,
    fill_value: float,
) -> torch.Tensor:
    real_head_dim = int(value.shape[-1])
    half = real_head_dim // 2
    padded_half = padded_head_dim // 2
    padded = torch.full(
        (*value.shape[:-1], padded_head_dim),
        fill_value,
        dtype=value.dtype,
        device=value.device,
    )
    padded[..., :half].copy_(value[..., :half])
    padded[..., padded_half : padded_half + half].copy_(
        value[..., half:]
    )
    return padded.contiguous()


class WeightPaddedVisionPrefillStage(VisionPrefillStage):
    """Production stage with PromptFA head padding encoded in Linear weights."""

    def _attention(
        self,
        attention: nn.Module,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _hidden = hidden_states.shape
        num_heads = int(attention.num_heads)
        padded_head_dim = int(attention._weight_padded_head_dim)
        qkv = torch.cat(
            [
                attention.q_proj(hidden_states),
                attention.k_proj(hidden_states),
                attention.v_proj(hidden_states),
            ],
            dim=-1,
        )
        query_states, key_states, value_states = qkv.chunk(3, dim=-1)
        query_states = query_states.view(
            batch_size,
            sequence_length,
            num_heads,
            padded_head_dim,
        )
        key_states = key_states.view(
            batch_size,
            sequence_length,
            num_heads,
            padded_head_dim,
        )
        value_states = value_states.view(
            batch_size,
            sequence_length,
            num_heads,
            padded_head_dim,
        )
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states,
            key_states,
            rope_cos,
            rope_sin,
        )
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()
        attention_output = vision_prompt_flash_attention_bnsd(
            query_states,
            key_states,
            value_states,
            num_heads=num_heads,
            scale=float(attention.scaling),
            atten_mask=attention_mask,
        )
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                num_heads * padded_head_dim,
            )
        )
        return attention.out_proj(attention_output)


def _apply_joint_manual_rotary(
    qk: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> torch.Tensor:
    """Apply the existing FP32 half-RoPE formula once to adjacent Q and K."""

    original_dtype = qk.dtype
    qk_fp32 = qk.float()
    cos_fp32 = rope_cos.unsqueeze(-2).float()
    sin_fp32 = rope_sin.unsqueeze(-2).float()
    half = qk_fp32.shape[-1] // 2
    rotated_half = torch.cat(
        (-qk_fp32[..., half:], qk_fp32[..., :half]),
        dim=-1,
    )
    return (
        qk_fp32 * cos_fp32 + rotated_half * sin_fp32
    ).to(original_dtype)


class JointManualWeightPaddedVisionPrefillStage(
    WeightPaddedVisionPrefillStage
):
    """D80 stage applying the exact manual formula to one contiguous QK."""

    def _attention(
        self,
        attention: nn.Module,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _hidden = hidden_states.shape
        num_heads = int(attention.num_heads)
        padded_head_dim = int(attention._weight_padded_head_dim)
        qk_states = torch.cat(
            (
                attention.q_proj(hidden_states),
                attention.k_proj(hidden_states),
            ),
            dim=-1,
        ).view(
            batch_size,
            sequence_length,
            2 * num_heads,
            padded_head_dim,
        )
        value_states = attention.v_proj(hidden_states).view(
            batch_size,
            sequence_length,
            num_heads,
            padded_head_dim,
        )
        qk_states = _apply_joint_manual_rotary(
            qk_states,
            rope_cos,
            rope_sin,
        )
        qk_bnsd = (
            qk_states.view(
                batch_size,
                sequence_length,
                2,
                num_heads,
                padded_head_dim,
            )
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        query_states, key_states = qk_bnsd.unbind(0)
        value_states = value_states.transpose(1, 2).contiguous()
        attention_output = vision_prompt_flash_attention_bnsd(
            query_states,
            key_states,
            value_states,
            num_heads=num_heads,
            scale=float(attention.scaling),
            atten_mask=attention_mask,
        )
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                num_heads * padded_head_dim,
            )
        )
        return attention.out_proj(attention_output)


def _half_to_interleaved(value: torch.Tensor) -> torch.Tensor:
    """Permute [..., A..., B...] into [..., A0, B0, A1, B1, ...]."""

    half = value.shape[-1] // 2
    first = value[..., :half]
    second = value[..., half:]
    return torch.stack((first, second), dim=-1).flatten(-2).contiguous()


def _interleave_weight_padded_qk(model: nn.Module) -> dict[str, int]:
    """Permanently put padded Q/K projection rows in interleaved RoPE order."""

    layer_count = 0
    tensor_count = 0
    for layer in model.visual.vision_model.encoder.layers:
        attention = layer.self_attn
        num_heads = int(attention.num_heads)
        padded_head_dim = int(attention._weight_padded_head_dim)
        if padded_head_dim != 80:
            raise ValueError(
                "the current in-place comparison is pinned to D80, got "
                f"{padded_head_dim}"
            )
        for projection_name in ("q_proj", "k_proj"):
            projection = getattr(attention, projection_name)
            with torch.no_grad():
                weight = projection.weight.view(
                    num_heads,
                    padded_head_dim,
                    *projection.weight.shape[1:],
                )
                interleaved_weight = torch.stack(
                    (
                        weight[:, : padded_head_dim // 2],
                        weight[:, padded_head_dim // 2 :],
                    ),
                    dim=2,
                ).reshape_as(weight)
                projection.weight.copy_(
                    interleaved_weight.reshape_as(projection.weight)
                )
                tensor_count += 1
                if projection.bias is not None:
                    bias = projection.bias.view(
                        num_heads,
                        padded_head_dim,
                    )
                    interleaved_bias = torch.stack(
                        (
                            bias[:, : padded_head_dim // 2],
                            bias[:, padded_head_dim // 2 :],
                        ),
                        dim=2,
                    ).reshape_as(bias)
                    projection.bias.copy_(
                        interleaved_bias.reshape_as(projection.bias)
                    )
                    tensor_count += 1
        layer_count += 1
    return {
        "interleaved_qk_layers": layer_count,
        "interleaved_qk_parameter_tensors": tensor_count,
    }


def _inplace_partial_rope_inputs(
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare full interleaved FP16 factors once, outside the layer loop."""

    return (
        _half_to_interleaved(rope_cos).to(dtype).unsqueeze(-2),
        _half_to_interleaved(rope_sin).to(dtype).unsqueeze(-2),
    )


class InplacePartialWeightPaddedVisionPrefillStage(
    WeightPaddedVisionPrefillStage
):
    """910B D80 stage using one interleaved in-place QK RoPE call."""

    def _attention(
        self,
        attention: nn.Module,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _hidden = hidden_states.shape
        num_heads = int(attention.num_heads)
        padded_head_dim = int(attention._weight_padded_head_dim)
        qk_states = torch.cat(
            (
                attention.q_proj(hidden_states),
                attention.k_proj(hidden_states),
            ),
            dim=-1,
        ).contiguous()
        qk_states = qk_states.view(
            batch_size,
            sequence_length,
            2 * num_heads,
            padded_head_dim,
        )
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            qk_states,
            rope_cos,
            rope_sin,
            "interleave",
            [0, padded_head_dim],
        )
        qk_bnsd = (
            qk_states.view(
                batch_size,
                sequence_length,
                2,
                num_heads,
                padded_head_dim,
            )
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        query_states, key_states = qk_bnsd.unbind(0)
        value_states = (
            attention.v_proj(hidden_states)
            .view(
                batch_size,
                sequence_length,
                num_heads,
                padded_head_dim,
            )
            .transpose(1, 2)
            .contiguous()
        )
        attention_output = vision_prompt_flash_attention_bnsd(
            query_states,
            key_states,
            value_states,
            num_heads=num_heads,
            scale=float(attention.scaling),
            atten_mask=attention_mask,
        )
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                num_heads * padded_head_dim,
            )
        )
        return attention.out_proj(attention_output)


def _register_inplace_partial_torchair_converter() -> None:
    """Override the generated converter's invalid string GE mode mapping."""

    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge import custom_op

    inplace_rope = (
        torch.ops._C_ascend.inplace_partial_rotary_mul.default
    )

    @register_fx_node_ge_converter(inplace_rope)
    def _convert_inplace_partial_rotary_mul(
        x: Any,
        r1: Any,
        r2: Any,
        rotary_mode: str,
        partial_slice: Sequence[int],
    ) -> None:
        modes = {
            "half": 0,
            "interleave": 1,
            "quarter": 2,
            "interleave-half": 3,
        }
        custom_op(
            "InplacePartialRotaryMul",
            x,
            r1,
            r2,
            modes[rotary_mode],
            partial_slice,
        )


def _linear_modules(stage: nn.Module) -> list[tuple[str, nn.Linear]]:
    modules = [
        (name, module)
        for name, module in stage.named_modules()
        if isinstance(module, nn.Linear)
    ]
    expected = len(stage.transformer.encoder.layers) * 6
    if len(modules) != expected:
        raise RuntimeError(
            f"expected {expected} Linear modules, found {len(modules)}"
        )
    return modules


def _format_histogram(
    modules: list[tuple[str, nn.Linear]],
    torch_npu: Any,
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(int(torch_npu.get_npu_format(module.weight)))
                for _name, module in modules
            ).items()
        )
    )


def _prepare_weight_format(
    stage: nn.Module,
    *,
    requested: str,
    torch_npu: Any,
) -> dict[str, Any]:
    modules = _linear_modules(stage)
    before = _format_histogram(modules, torch_npu)
    metadata: dict[str, Any] = {
        "requested": requested,
        "target_format": (
            "FRACTAL_NZ" if requested == "fractal_nz" else "unchanged"
        ),
        "target_format_code": (
            FRACTAL_NZ if requested == "fractal_nz" else None
        ),
        "linear_weight_count": len(modules),
        "before_format_histogram": before,
        "converted_count": 0,
        "status": "ready",
        "failures": [],
    }
    if requested == "native":
        metadata["after_format_histogram"] = before
        metadata["all_after_are_nz"] = all(
            int(code) == FRACTAL_NZ
            for code in before
        )
        return metadata

    converted = 0
    failures: list[dict[str, Any]] = []
    for name, module in modules:
        before_code = int(torch_npu.get_npu_format(module.weight))
        if before_code == FRACTAL_NZ:
            continue
        try:
            module.weight.data = torch_npu.npu_format_cast(
                module.weight.data,
                FRACTAL_NZ,
            )
            after_code = int(torch_npu.get_npu_format(module.weight))
        except Exception as exc:
            failures.append(
                {
                    "module": name,
                    "before_format": before_code,
                    "error": repr(exc),
                }
            )
            break
        if after_code != FRACTAL_NZ:
            failures.append(
                {
                    "module": name,
                    "before_format": before_code,
                    "after_format": after_code,
                    "error": "format_cast_did_not_produce_fractal_nz",
                }
            )
            break
        converted += 1
    after = _format_histogram(modules, torch_npu)
    all_after_are_nz = all(
        int(torch_npu.get_npu_format(module.weight)) == FRACTAL_NZ
        for _name, module in modules
    )
    metadata.update(
        {
            "converted_count": converted,
            "after_format_histogram": after,
            "all_after_are_nz": all_after_are_nz,
            "failures": failures,
            "status": "ready" if all_after_are_nz else "unsupported",
        }
    )
    return metadata


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sample_summary(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    return {
        "samples": samples,
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "p05": _percentile(samples, 0.05),
        "p95": _percentile(samples, 0.95),
    }


def _repeat(
    run: Callable[..., torch.Tensor],
    inputs: StageInputs,
    calls: int,
) -> torch.Tensor:
    output: torch.Tensor | None = None
    for _ in range(calls):
        output = run(*inputs)
    if output is None:
        raise AssertionError("repeat count must be positive")
    return output


def _measure(
    run: Callable[..., torch.Tensor],
    inputs: StageInputs,
    *,
    device: torch.device,
    samples: int,
    calls_per_sample: int,
    physical_tokens_per_call: int,
    flops_per_call: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    device_block_ms: list[float] = []
    wall_block_ms: list[float] = []
    output: torch.Tensor | None = None
    for _ in range(samples):
        timeline = DeviceTimeline(device)
        wall_started = time.perf_counter()
        output = timeline.measure(
            "full_stack_replays",
            lambda: _repeat(run, inputs, calls_per_sample),
        )
        spans = timeline.resolve_spans()
        wall_block_ms.append((time.perf_counter() - wall_started) * 1000.0)
        device_block_ms.append(
            float(spans["full_stack_replays"]["seconds"]) * 1000.0
        )
    if output is None:
        raise RuntimeError("measurement produced no output")

    device_per_call_ms = [
        value / calls_per_sample for value in device_block_ms
    ]
    wall_per_call_ms = [
        value / calls_per_sample for value in wall_block_ms
    ]
    device = _sample_summary(device_per_call_ms)
    wall = _sample_summary(wall_per_call_ms)
    device_median_s = float(device["median"]) / 1000.0
    wall_median_s = float(wall["median"]) / 1000.0
    return (
        {
            "samples": samples,
            "calls_per_sample": calls_per_sample,
            "total_measured_full_stack_calls": samples * calls_per_sample,
            "device_event_per_call_ms": device,
            "synchronized_wall_per_call_ms": wall,
            "physical_tokens_per_s_device_median": (
                physical_tokens_per_call / device_median_s
            ),
            "physical_tokens_per_s_wall_median": (
                physical_tokens_per_call / wall_median_s
            ),
            "linear_tflop_per_s_device_median": (
                flops_per_call / device_median_s / 1e12
            ),
        },
        output,
    )


def _diff(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left32 = left.float()
    right32 = right.float()
    delta = (left32 - right32).abs()
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "left_finite": bool(torch.isfinite(left32).all().item()),
        "right_finite": bool(torch.isfinite(right32).all().item()),
        "same_shape": tuple(left.shape) == tuple(right.shape),
    }


def _cache_dir(
    root: Path,
    *,
    sequence_length: int,
    batch_size: int,
    intermediate_size: int,
    weight_format: str,
    attention_head_padding: str,
    rotary_implementation: str,
    dtype: torch.dtype,
    device: torch.device,
    model_dir: Path,
) -> Path:
    key_parts = [
        "vision_promptfa_stack",
        "layers27",
        "attention_prompt_flash_attention",
        f"pfalayout{cache_key_part(get_vision_prompt_fa_layout())}",
        (
            "pfasparse"
            f"{get_vision_prompt_fa_mask_sparse_mode()}"
        ),
        f"seq{sequence_length}",
        f"batch{batch_size}",
        f"intermediate{intermediate_size}",
        f"weights{weight_format}",
        f"headpadding{attention_head_padding}",
        f"dtype{cache_key_part(dtype)}",
        f"mode{cache_key_part(TORCHAIR_EXECUTION_MODE)}",
        f"model{short_file_hash(model_dir / 'config.json')}",
        f"torch{cache_key_part(torch.__version__)}",
        f"torchnpu{torch_npu_version_label(device)}",
        f"torchair{torchair_version_label(device)}",
    ]
    if rotary_implementation == "separate_manual":
        # This implementation is byte-for-byte the already measured control.
        # Preserve its warm graph identity while making every new lane explicit.
        key_parts.append(
            f"src{LEGACY_SEPARATE_MANUAL_SOURCE_HASH}"
        )
    elif rotary_implementation == "joint_manual":
        # The joint-manual implementation is unchanged from commit 95bd2ca;
        # unrelated native-lane converter work must not discard its warm graph.
        key_parts.extend(
            [
                f"rotary{cache_key_part(rotary_implementation)}",
                f"src{JOINT_MANUAL_SOURCE_HASH}",
            ]
        )
    else:
        key_parts.extend(
            [
                f"rotary{cache_key_part(rotary_implementation)}",
                f"src{short_file_hash(Path(__file__).resolve())}",
            ]
        )
    key = "_".join(key_parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    compact = (
        f"vision_pfa_b{batch_size}_s{sequence_length}_"
        f"i{intermediate_size}_{weight_format}_"
        f"hp{attention_head_padding}_"
        f"rope{cache_key_part(rotary_implementation)}_{digest}"
    )
    if rotary_implementation == "separate_manual":
        # Retain the exact legacy directory name as well as its key.
        compact = (
            f"vision_pfa_b{batch_size}_s{sequence_length}_"
            f"i{intermediate_size}_{weight_format}_"
            f"hp{attention_head_padding}_{digest}"
        )
    return root.expanduser().resolve() / compact


def _cache_populated(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def _compile(
    stage: nn.Module,
    *,
    cache_dir: Path,
    allow_missing: bool,
    device: torch.device,
    example: StageInputs,
) -> tuple[Callable[..., torch.Tensor], dict[str, Any]]:
    existed = _cache_populated(cache_dir)
    if not existed and not allow_missing:
        raise RuntimeError(
            "the exact graph cache is missing; pass "
            f"--allow-compile-if-missing to create it: {cache_dir}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    torchair, CompilerConfig = import_torchair()
    synchronize(device)
    wrapper_started = time.perf_counter()
    compiled = torchair.inference.cache_compile(
        stage.forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
    )
    synchronize(device)
    wrapper_s = time.perf_counter() - wrapper_started
    synchronize(device)
    first_started = time.perf_counter()
    first_output = compiled(*example)
    synchronize(device)
    first_call_s = time.perf_counter() - first_started
    del first_output
    if not _cache_populated(cache_dir):
        raise RuntimeError(f"TorchAir did not populate cache: {cache_dir}")
    return compiled, {
        "api": "torchair.inference.cache_compile",
        "dynamic": False,
        "fullgraph": True,
        "ge_cache": True,
        "cache_dir": str(cache_dir),
        "cache_existed_before": existed,
        "compile_was_permitted": allow_missing,
        "wrapper_s": wrapper_s,
        "first_call_s": first_call_s,
    }


def _profiler_config(metric: str) -> Any:
    import torch_npu.profiler as npu_prof

    metrics = {
        "pipe": npu_prof.AiCMetrics.PipeUtilization,
        "memory": npu_prof.AiCMetrics.Memory,
        "l2": npu_prof.AiCMetrics.L2Cache,
        "memory_access": npu_prof.AiCMetrics.MemoryAccess,
    }
    return npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1,
        aic_metrics=metrics[metric],
        # AiCMetrics.L2Cache is the per-kernel PMU lane. The separate
        # l2_cache=True switch emits the legacy l2_cache.csv path and is not
        # needed for this cross-lane task analysis.
        l2_cache=False,
        export_type=npu_prof.ExportType.Text,
        data_simplification=False,
    )


def _profiler_capabilities() -> dict[str, Any]:
    import torch_npu.profiler as npu_prof

    capabilities: dict[str, Any] = {}
    for name in (
        "supported_activities",
        "supported_profiler_level",
        "supported_ai_core_metrics",
        "supported_export_type",
    ):
        query = getattr(npu_prof, name, None)
        if query is None:
            capabilities[name] = None
            continue
        try:
            value = query()
            if isinstance(value, (set, frozenset, list, tuple)):
                capabilities[name] = sorted(str(item) for item in value)
            else:
                capabilities[name] = str(value)
        except Exception as exc:
            capabilities[name] = {"error": repr(exc)}
    return capabilities


def _profile(
    run: Callable[..., torch.Tensor],
    inputs: StageInputs,
    *,
    profile_dir: Path,
    metric: str,
    warmup_steps: int,
    active_steps: int,
    label: str,
) -> dict[str, Any]:
    import torch_npu.profiler as npu_prof

    if profile_dir.exists() and any(profile_dir.iterdir()):
        raise RuntimeError(
            f"profile directory already exists and is non-empty: {profile_dir}"
        )
    shutil.rmtree(profile_dir, ignore_errors=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    schedule = npu_prof.schedule(
        wait=0,
        warmup=warmup_steps,
        active=active_steps,
        repeat=1,
    )
    context_started = time.perf_counter()
    with npu_prof.profile(
        activities=[
            npu_prof.ProfilerActivity.CPU,
            npu_prof.ProfilerActivity.NPU,
        ],
        schedule=schedule,
        experimental_config=_profiler_config(metric),
        on_trace_ready=npu_prof.tensorboard_trace_handler(
            str(profile_dir),
            analyse_flag=True,
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
        with_modules=True,
        with_flops=False,
    ) as profiler:
        for step in range(warmup_steps + active_steps):
            phase = "warmup" if step < warmup_steps else "active"
            with torch.profiler.record_function(f"{label}.{phase}.step{step}"):
                output = run(*inputs)
                torch.npu.synchronize()
            profiler.step()
    torch.npu.synchronize()
    del output
    return {
        "profile_dir": str(profile_dir),
        "scheduled_warmup_steps": warmup_steps,
        "active_steps": active_steps,
        "context_wall_s": time.perf_counter() - context_started,
        "throughput_measurement": False,
        "metric": metric,
        "profiler_level": "Level1",
        "capabilities": _profiler_capabilities(),
    }


def _parse_profile(
    profile_dir: Path,
    output_dir: Path,
    *,
    metric: str,
    contract_path: Path,
    topn: int,
) -> dict[str, Any]:
    parser = (
        REPO_ROOT
        / "07_vision_prefill_optimization"
        / "parse_static_visual_encoder_profile.py"
    )
    command = [
        sys.executable,
        str(parser),
        "--profile-dir",
        str(profile_dir),
        "--topn",
        str(topn),
        "--skip-trace",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    source_json = profile_dir / "parsed_profile_summary.json"
    source_md = profile_dir / "parsed_profile_summary.md"
    destination_json = output_dir / "parsed_profile_summary.json"
    destination_md = output_dir / "parsed_profile_summary.md"
    shutil.copyfile(source_json, destination_json)
    shutil.copyfile(source_md, destination_md)
    parsed = json.loads(destination_json.read_text(encoding="utf-8"))
    detailed_output_dir = output_dir / "detailed_profile"
    detailed_command = [
        sys.executable,
        str(HERE / "analyze_vision_matmul_profile.py"),
        "--lane",
        f"{metric}={profile_dir}",
        "--output-dir",
        str(detailed_output_dir),
        "--contract",
        str(contract_path),
    ]
    detailed_completed = subprocess.run(
        detailed_command,
        check=True,
        capture_output=True,
        text=True,
    )
    detailed_json = detailed_output_dir / "profile_analysis.json"
    detailed = json.loads(detailed_json.read_text(encoding="utf-8"))
    dispatch: Counter[str] = Counter()
    dispatch_duration_us: Counter[str] = Counter()
    transdata_count = 0
    transdata_duration_us = 0.0
    weighted_cube_numerator = 0.0
    weighted_cube_denominator = 0.0
    shape_signatures: list[dict[str, Any]] = []
    for run in parsed.get("runs", []):
        kernel_details = run.get("kernel_details", {})
        for row in kernel_details.get("top_kernel_types", []):
            name = str(row.get("name", ""))
            lowered = name.lower()
            count = int(row.get("count", 0))
            duration = float(row.get("duration_us", 0.0))
            if "matmulv2" in lowered:
                dispatch["MatMulV2"] += count
                dispatch_duration_us["MatMulV2"] += duration
            elif "matmulv3" in lowered:
                dispatch["MatMulV3"] += count
                dispatch_duration_us["MatMulV3"] += duration
            elif "matmul" in lowered:
                dispatch[name] += count
                dispatch_duration_us[name] += duration
            if "transdata" in lowered:
                transdata_count += count
                transdata_duration_us += duration
        total_duration = float(kernel_details.get("total_duration_us", 0.0))
        cube = float(
            kernel_details.get("weighted_cube_utilization_pct", 0.0)
        )
        weighted_cube_numerator += cube * total_duration
        weighted_cube_denominator += total_duration
        for row in kernel_details.get("top_shape_format_signatures", []):
            if "matmul" in str(row.get("name", "")).lower():
                shape_signatures.append(row)
    return {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "parsed_json": str(destination_json),
        "parsed_markdown": str(destination_md),
        "detailed_analysis": {
            "command": detailed_command,
            "stdout": detailed_completed.stdout,
            "stderr": detailed_completed.stderr,
            "analysis_json": str(detailed_json),
            "report_markdown": str(
                detailed_output_dir / "profile_report.md"
            ),
            "summary": detailed,
        },
        "dispatch": {
            "counts": dict(dispatch),
            "duration_us": dict(dispatch_duration_us),
        },
        "transdata": {
            "count": transdata_count,
            "duration_us": transdata_duration_us,
        },
        "weighted_cube_utilization_pct": (
            weighted_cube_numerator / weighted_cube_denominator
            if weighted_cube_denominator
            else 0.0
        ),
        "matmul_shape_format_signatures": shape_signatures,
    }


def _matmul_only_profile_metrics(
    parsed_profile: dict[str, Any],
    *,
    active_full_stack_calls: int,
    matmul_kernels_per_full_stack_call: int,
    linear_flops_per_full_stack_call: int,
) -> dict[str, Any]:
    dispatch = parsed_profile.get("dispatch", {})
    counts = dispatch.get("counts", {})
    durations = dispatch.get("duration_us", {})
    observed_kernel_count = sum(int(value) for value in counts.values())
    expected_kernel_count = (
        active_full_stack_calls * matmul_kernels_per_full_stack_call
    )
    if observed_kernel_count != expected_kernel_count:
        raise RuntimeError(
            "profiled MatMul kernel count does not match the full-stack "
            f"contract: observed={observed_kernel_count}, "
            f"expected={expected_kernel_count}"
        )
    total_duration_us = sum(float(value) for value in durations.values())
    if total_duration_us <= 0.0:
        raise RuntimeError("profile contains no positive MatMul duration")
    duration_per_call_ms = (
        total_duration_us / active_full_stack_calls / 1000.0
    )
    return {
        "active_profiled_full_stack_calls": active_full_stack_calls,
        "matmul_kernels_per_full_stack_call": (
            matmul_kernels_per_full_stack_call
        ),
        "observed_matmul_kernel_count": observed_kernel_count,
        "total_matmul_kernel_duration_us": total_duration_us,
        "matmul_kernel_duration_per_full_stack_call_ms": (
            duration_per_call_ms
        ),
        "linear_flops_per_full_stack_call": (
            linear_flops_per_full_stack_call
        ),
        "matmul_only_linear_tflop_per_s": (
            linear_flops_per_full_stack_call
            * active_full_stack_calls
            / total_duration_us
            / 1e6
        ),
    }


def _linear_flops_per_call(
    *,
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    attention_projection_size: int,
    intermediate_size: int,
    layers: int,
) -> int:
    # Four attention projections and the two MLP projections per layer.
    per_token_per_layer = (
        4 * 2 * hidden_size * attention_projection_size
        + 4 * hidden_size * intermediate_size
    )
    return batch_size * sequence_length * layers * per_token_per_layer


def _synthetic_grid(sequence_length: int) -> torch.Tensor:
    height = math.isqrt(sequence_length)
    while sequence_length % height:
        height -= 1
    return torch.tensor(
        [[1, height, sequence_length // height]],
        dtype=torch.int64,
    )


def _batch_inputs(
    inputs: StageInputs,
    *,
    batch_size: int,
) -> StageInputs:
    if batch_size == 1:
        return inputs
    return tuple(
        tensor.repeat(batch_size, 1, 1, 1)
        if tensor.ndim == 4
        else tensor.repeat(batch_size, 1, 1)
        for tensor in inputs
    )  # type: ignore[return-value]


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    import torch_npu

    custom_op_enabled: bool | None = None
    if args.rotary_implementation == "joint_inplace_partial":
        from vllm_ascend.utils import enable_custom_op

        custom_op_enabled = bool(enable_custom_op())
        if not custom_op_enabled:
            raise RuntimeError(
                "vLLM-Ascend custom operators could not be registered"
            )
        if not hasattr(
            torch.ops._C_ascend,
            "inplace_partial_rotary_mul",
        ):
            raise RuntimeError(
                "_C_ascend::inplace_partial_rotary_mul is not registered"
            )
        if args.execution == "torchair":
            _register_inplace_partial_torchair_converter()
    internal_format_gate_enabled = False
    if args.weight_format == "fractal_nz":
        # torch-npu 2.10 defaults this runtime gate to disabled. It must be
        # enabled before the first NPU allocation; otherwise npu_format_cast
        # warns and leaves the tensor in ND even though CANN supports format 29.
        torch.npu.config.allow_internal_format = True
        internal_format_gate_enabled = True
    device = torch.device("npu:0")
    if not torch.npu.is_available():
        raise RuntimeError("vision MatMul lab requires an NPU")
    torch.npu.set_compile_mode(jit_compile=False)
    dtype = torch.float16
    model_dir = args.model.expanduser().resolve()
    sequence_length = int(args.sequence_length)
    batch_size = int(args.batch_size)
    intermediate_size = int(args.intermediate_size)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT
        / (
            f"b{batch_size}_s{sequence_length}_i{intermediate_size}_"
            f"{args.weight_format}_{args.attention_head_padding}_"
            f"{args.rotary_implementation}_{args.execution}"
        )
    ).expanduser().resolve()
    profile_dir = (
        args.profile_dir
        if args.profile_dir is not None
        else DEFAULT_PROFILE_ROOT
        / (
            f"b{batch_size}_s{sequence_length}_i{intermediate_size}_"
            f"{args.weight_format}_{args.attention_head_padding}_"
            f"{args.rotary_implementation}_{args.execution}"
            f"_{args.profile_metric}"
        )
    ).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"output directory already exists and is non-empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_started = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=dtype,
        device=device,
    )
    config = model.config.vision_config
    hidden_size = int(config.hidden_size)
    layers = int(config.num_hidden_layers)
    source_intermediate = int(config.intermediate_size)
    if source_intermediate != 4304 or layers != 27 or hidden_size != 1152:
        raise RuntimeError(
            "this lab is pinned to PaddleOCR-VL-1.6 vision dimensions; got "
            f"hidden={hidden_size}, intermediate={source_intermediate}, "
            f"layers={layers}"
        )
    torch.manual_seed(20260729)
    prefix_hidden_states = (
        torch.randn(
            (sequence_length, hidden_size),
            dtype=dtype,
            device="cpu",
        )
        * 0.02
    ).to(device)
    prepared = prepare_vision_prefill(
        model,
        prefix_hidden_states,
        _synthetic_grid(sequence_length),
        physical_seq_len=sequence_length,
        execution=args.execution,
    )
    inputs = _batch_inputs(
        (
            prepared.prefix_hidden_states,
            prepared.rope_cos,
            prepared.rope_sin,
            prepared.attention_mask,
        ),
        batch_size=batch_size,
    )

    reference_stage = VisionPrefillStage(
        model,
        attention_impl="prompt_flash_attention",
    ).eval()
    synchronize(device)
    reference_output = reference_stage(*inputs)
    synchronize(device)
    if intermediate_size != source_intermediate:
        _zero_extend_model_mlp(
            model,
            target_intermediate=intermediate_size,
        )
    attention_padding_metadata: dict[str, Any] = {
        "mode": args.attention_head_padding,
        "real_head_dim": int(
            config.hidden_size // config.num_attention_heads
        ),
        "padded_head_dim": prompt_flash_attention_call_head_dim(
            int(config.hidden_size // config.num_attention_heads)
        ),
        "runtime_pad_operations_per_layer": 3,
        "runtime_slice_operations_per_layer": 1,
    }
    candidate_inputs = inputs
    weight_padded_manual_output: torch.Tensor | None = None
    rotary_metadata: dict[str, Any] = {
        "implementation": args.rotary_implementation,
        "custom_op_enabled": custom_op_enabled,
        "portable_to_310p": (
            args.rotary_implementation != "joint_inplace_partial"
        ),
    }
    if args.attention_head_padding == "weights":
        attention_padding_metadata.update(
            _weight_pad_model_attention(model)
        )
        padded_rope_cos = _weight_pad_rope(
            inputs[1],
            padded_head_dim=int(
                attention_padding_metadata["padded_head_dim"]
            ),
            fill_value=1.0,
        )
        padded_rope_sin = _weight_pad_rope(
            inputs[2],
            padded_head_dim=int(
                attention_padding_metadata["padded_head_dim"]
            ),
            fill_value=0.0,
        )
        manual_inputs: StageInputs = (
            inputs[0],
            padded_rope_cos,
            padded_rope_sin,
            inputs[3],
        )
        weight_padded_manual_stage = WeightPaddedVisionPrefillStage(
            model,
            attention_impl="prompt_flash_attention",
        ).eval()
        synchronize(device)
        weight_padded_manual_output = weight_padded_manual_stage(
            *manual_inputs
        )
        synchronize(device)
        if args.rotary_implementation == "separate_manual":
            candidate_inputs = manual_inputs
            candidate_stage = weight_padded_manual_stage
            rotary_metadata.update(
                {
                    "qk_layout": "separate_bsnd",
                    "math": "separate FP32 half-RoPE",
                    "operator": "ordinary PyTorch operations",
                }
            )
        elif args.rotary_implementation == "joint_manual":
            candidate_inputs = manual_inputs
            candidate_stage = JointManualWeightPaddedVisionPrefillStage(
                model,
                attention_impl="prompt_flash_attention",
            ).eval()
            rotary_metadata.update(
                {
                    "qk_layout": "contiguous_joint_bsnd",
                    "math": "joint FP32 half-RoPE",
                    "operator": "ordinary PyTorch operations",
                }
            )
        else:
            rotary_metadata.update(_interleave_weight_padded_qk(model))
            rope_segments = _inplace_partial_rope_inputs(
                padded_rope_cos,
                padded_rope_sin,
                dtype=dtype,
            )
            candidate_inputs = (
                inputs[0],
                *rope_segments,
                inputs[3],
            )
            candidate_stage = (
                InplacePartialWeightPaddedVisionPrefillStage(
                    model,
                    attention_impl="prompt_flash_attention",
                ).eval()
            )
            rotary_metadata.update(
                {
                    "qk_layout": "contiguous_joint_bsnd_interleaved",
                    "math": (
                        "one in-place interleaved D80 slice after "
                        "one-time Q/K weight and factor permutation"
                    ),
                    "operator": (
                        "_C_ascend::inplace_partial_rotary_mul"
                    ),
                    "partial_slices": [[0, 80]],
                    "factor_dtype": str(dtype),
                    "explicit_torchair_converter": (
                        args.execution == "torchair"
                    ),
                }
            )
        attention_padding_metadata.update(
            {
                "runtime_pad_operations_per_layer": 0,
                "runtime_slice_operations_per_layer": 0,
            }
        )
    else:
        candidate_stage = VisionPrefillStage(
            model,
            attention_impl="prompt_flash_attention",
        ).eval()

    format_metadata = _prepare_weight_format(
        candidate_stage,
        requested=args.weight_format,
        torch_npu=torch_npu,
    )
    format_metadata["runtime_gate"] = {
        "torch_npu_allow_internal_format_enabled_before_npu_allocation": (
            internal_format_gate_enabled
        ),
        "mm_bmm_format_nd": bool(torch.npu.get_mm_bmm_format_nd()),
        "manual_cast_after_model_load": (
            args.weight_format == "fractal_nz"
        ),
    }
    environment = _environment(device)
    environment.update(
        {
            "hostname": platform.node(),
            "ascend_rt_visible_devices": os.environ.get(
                "ASCEND_RT_VISIBLE_DEVICES"
            ),
            "npu_visible_devices": os.environ.get("NPU_VISIBLE_DEVICES"),
            "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
            "ascend_toolkit_home": os.environ.get("ASCEND_TOOLKIT_HOME"),
        }
    )
    summary: dict[str, Any] = {
        "schema_version": 4,
        "status": format_metadata["status"],
        "purpose": (
            "exact production 27-layer vision PromptFA format/alignment "
            "experiment using synthetic shape inputs"
        ),
        "boundary": (
            "VisionPrefillStage: 27 x (LayerNorm1 + Q/K/V + RoPE + "
            "PromptFA + out projection + residual + LayerNorm2 + "
            "FC1/GELU/FC2 + residual) + post-LayerNorm"
        ),
        "production_stage": True,
        "synthetic_shape_inputs": True,
        "attention": {
            "implementation": "prompt_flash_attention",
            "input_layout": get_vision_prompt_fa_layout().upper(),
            "mask_sparse_mode": get_vision_prompt_fa_mask_sparse_mode(),
            "model_head_dim": int(
                config.hidden_size // config.num_attention_heads
            ),
            "promptfa_call_head_dim": prompt_flash_attention_call_head_dim(
                int(config.hidden_size // config.num_attention_heads)
            ),
            "attention_mask_all_false": True,
            "head_padding": attention_padding_metadata,
            "rotary": rotary_metadata,
        },
        "shape": {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "physical_tokens_per_call": batch_size * sequence_length,
            "hidden_size": hidden_size,
            "source_intermediate_size": source_intermediate,
            "candidate_intermediate_size": intermediate_size,
            "layers": layers,
            "linear_calls_per_layer": 6,
            "linear_calls_per_full_stack": layers * 6,
        },
        "requested": {
            "weight_format": args.weight_format,
            "attention_head_padding": args.attention_head_padding,
            "rotary_implementation": args.rotary_implementation,
            "execution": args.execution,
            "profile_metric": args.profile_metric,
        },
        "weight_format": format_metadata,
        "setup_s_through_format_preparation": time.perf_counter()
        - setup_started,
        "environment": environment,
    }
    summary_path = output_dir / "run_summary.json"
    if format_metadata["status"] != "ready":
        summary["reason"] = (
            "requested FRACTAL_NZ could not be materialized by this "
            "torch_npu/runtime; no mislabeled fallback timing was run"
        )
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2), flush=True)
        return

    synchronize(device)
    raw_candidate_output = candidate_stage(*candidate_inputs)
    synchronize(device)
    raw_candidate_vs_native = _diff(
        raw_candidate_output,
        reference_output,
    )
    raw_candidate_vs_weight_padded_manual = (
        _diff(
            raw_candidate_output,
            weight_padded_manual_output,
        )
        if weight_padded_manual_output is not None
        else None
    )
    cache_dir = _cache_dir(
        args.cache_dir,
        sequence_length=sequence_length,
        batch_size=batch_size,
        intermediate_size=intermediate_size,
        weight_format=args.weight_format,
        attention_head_padding=args.attention_head_padding,
        rotary_implementation=args.rotary_implementation,
        dtype=dtype,
        device=device,
        model_dir=model_dir,
    )
    if args.execution == "torchair":
        run, compile_metadata = _compile(
            candidate_stage,
            cache_dir=cache_dir,
            allow_missing=bool(args.allow_compile_if_missing),
            device=device,
            example=candidate_inputs,
        )
    else:
        run = candidate_stage
        compile_metadata = {
            "api": None,
            "cache_dir": None,
            "cache_existed_before": None,
        }
    for _ in range(args.warmup):
        warm_output = run(*candidate_inputs)
    synchronize(device)
    del warm_output

    flops_per_call = _linear_flops_per_call(
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        attention_projection_size=int(
            attention_padding_metadata.get(
                "padded_attention_size",
                hidden_size,
            )
        ),
        intermediate_size=intermediate_size,
        layers=layers,
    )
    measurements, output = _measure(
        run,
        candidate_inputs,
        device=device,
        samples=args.samples,
        calls_per_sample=args.calls_per_sample,
        physical_tokens_per_call=batch_size * sequence_length,
        flops_per_call=flops_per_call,
    )
    summary.update(
        {
            "status": "completed",
            "compile": compile_metadata,
            "linear_flops_per_full_stack_call": flops_per_call,
            "measurements": measurements,
            "numerics": {
                "raw_candidate_vs_native_4304": raw_candidate_vs_native,
                "raw_candidate_vs_weight_padded_manual": (
                    raw_candidate_vs_weight_padded_manual
                ),
                "measured_output_vs_raw_candidate": _diff(
                    output,
                    raw_candidate_output,
                ),
                "measured_output_finite": bool(
                    torch.isfinite(output.float()).all().item()
                ),
            },
        }
    )

    if args.profile:
        profile_contract_path = output_dir / "profile_contract.json"
        profile_contract = {
            "schema_version": 1,
            "command": [sys.executable, *sys.argv],
            "purpose": summary["purpose"],
            "boundary": summary["boundary"],
            "environment": summary["environment"],
            "shape": summary["shape"],
            "requested": summary["requested"],
            "weight_format": summary["weight_format"],
            "attention": summary["attention"],
            "compile": summary["compile"],
            "linear_flops_per_full_stack_call": flops_per_call,
            "profile": {
                "metric": args.profile_metric,
                "warmup_steps": args.profile_warmup_steps,
                "active_steps": args.profile_steps,
                "throughput_measurement": False,
            },
        }
        profile_contract_path.write_text(
            json.dumps(profile_contract, indent=2) + "\n",
            encoding="utf-8",
        )
        label = (
            "paddleocr_vl.vision_matmul_lab."
            f"B{batch_size}.S{sequence_length}.I{intermediate_size}."
            f"{args.weight_format}.{args.attention_head_padding}."
            f"{args.rotary_implementation}."
            f"{args.execution}"
        )
        summary["profiler"] = _profile(
            run,
            candidate_inputs,
            profile_dir=profile_dir,
            metric=args.profile_metric,
            warmup_steps=args.profile_warmup_steps,
            active_steps=args.profile_steps,
            label=label,
        )
        parsed_profile = _parse_profile(
            profile_dir,
            output_dir,
            metric=args.profile_metric,
            contract_path=profile_contract_path,
            topn=args.parser_topn,
        )
        parsed_profile["matmul_only"] = _matmul_only_profile_metrics(
            parsed_profile,
            active_full_stack_calls=args.profile_steps,
            matmul_kernels_per_full_stack_call=layers * 6,
            linear_flops_per_full_stack_call=flops_per_call,
        )
        summary["parsed_profile"] = parsed_profile

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "status": summary["status"],
                "shape": summary["shape"],
                "weight_format": summary["weight_format"],
                "device_median_ms": measurements[
                    "device_event_per_call_ms"
                ]["median"],
                "physical_tokens_per_s": measurements[
                    "physical_tokens_per_s_device_median"
                ],
                "linear_tflop_per_s": measurements[
                    "linear_tflop_per_s_device_median"
                ],
                "dispatch": summary.get("parsed_profile", {}).get(
                    "dispatch"
                ),
                "matmul_only": summary.get("parsed_profile", {}).get(
                    "matmul_only"
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
