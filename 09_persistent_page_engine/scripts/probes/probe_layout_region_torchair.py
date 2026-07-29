#!/usr/bin/env python3
"""Compile one fixed-shape compute region of the owned layout model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.compile_utils import import_torchair
from pipeline.layout_frontend import OwnedLayoutFrontend, _decode_rgb


DEFAULT_IMAGE = Path(
    "/workspace/datasets/OmniDocBench/images/"
    "page-d1561665-5359-42fe-920c-d6e3bff81953.png"
)
DEFAULT_LAYOUT_MODEL = Path("/workspace/models/PP-DocLayoutV3_safetensors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--layout-model",
        type=Path,
        default=DEFAULT_LAYOUT_MODEL,
    )
    parser.add_argument(
        "--region",
        choices=("decoder", "hybrid_encoder"),
        required=True,
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=30)
    return parser.parse_args()


class _HybridEncoderRegion(nn.Module):
    """Tensor-only boundary around hybrid attention, FPN, and mask features."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        feature_0: Tensor,
        feature_1: Tensor,
        feature_2: Tensor,
        x4_feature: Tensor,
        x4_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        output = self.encoder(
            [feature_0, feature_1, feature_2],
            (x4_feature, x4_mask),
        )
        return (
            output.last_hidden_state[0],
            output.last_hidden_state[1],
            output.last_hidden_state[2],
            output.mask_feat,
        )


class _DecoderRegion(nn.Module):
    """Tensor-only boundary around all six decoder layers and final heads."""

    def __init__(
        self,
        core: nn.Module,
        spatial_shapes_list: tuple[tuple[int, int], ...],
    ) -> None:
        super().__init__()
        self.decoder = core.decoder
        self.order_head = core.decoder_order_head
        self.global_pointer = core.decoder_global_pointer
        self.mask_query_head = core.mask_query_head
        self.norm = core.decoder_norm
        self.spatial_shapes_list = spatial_shapes_list

    def forward(
        self,
        inputs_embeds: Tensor,
        encoder_hidden_states: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        mask_feat: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        output = self.decoder(
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=list(self.spatial_shapes_list),
            level_start_index=level_start_index,
            order_head=self.order_head,
            global_pointer=self.global_pointer,
            mask_query_head=self.mask_query_head,
            norm=self.norm,
            mask_feat=mask_feat,
        )
        return (
            output.last_hidden_state,
            output.intermediate_hidden_states,
            output.intermediate_logits,
            output.intermediate_reference_points,
            output.decoder_out_order_logits,
            output.decoder_out_masks,
        )


def _capture_call(
    module: nn.Module,
    model: nn.Module,
    pixel_values: Tensor,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def capture(
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        calls.append((args, kwargs))

    handle = module.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        model(pixel_values=pixel_values)
    finally:
        handle.remove()
    if len(calls) != 1:
        raise RuntimeError(
            f"expected one region call during detector forward, got {len(calls)}"
        )
    return calls[0]


def _build_region(
    core: nn.Module,
    region_name: str,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
) -> tuple[nn.Module, tuple[Tensor, ...], tuple[str, ...]]:
    if region_name == "hybrid_encoder":
        inputs_embeds = captured_args[0]
        x4_feat = captured_args[1]
        region = _HybridEncoderRegion(core.encoder)
        inputs = (
            inputs_embeds[0],
            inputs_embeds[1],
            inputs_embeds[2],
            x4_feat[0],
            x4_feat[1],
        )
        names = ("feature_0", "feature_1", "feature_2", "mask_feat")
        return region, inputs, names

    spatial_shapes_list = tuple(
        tuple(int(value) for value in shape)
        for shape in captured_kwargs["spatial_shapes_list"]
    )
    region = _DecoderRegion(core, spatial_shapes_list)
    inputs = (
        captured_kwargs["inputs_embeds"],
        captured_kwargs["encoder_hidden_states"],
        captured_kwargs["reference_points"],
        captured_kwargs["spatial_shapes"],
        captured_kwargs["level_start_index"],
        captured_kwargs["mask_feat"],
    )
    names = (
        "last_hidden_state",
        "intermediate_hidden_states",
        "intermediate_logits",
        "intermediate_reference_points",
        "order_logits",
        "out_masks",
    )
    return region, inputs, names


def _cpu_clone(
    names: tuple[str, ...],
    output: tuple[Tensor, ...],
) -> dict[str, Tensor]:
    return {
        name: tensor.detach().float().cpu().clone()
        for name, tensor in zip(names, output, strict=True)
    }


def _diff(
    actual: dict[str, Tensor],
    reference: dict[str, Tensor],
) -> dict[str, dict[str, float | bool]]:
    result: dict[str, dict[str, float | bool]] = {}
    for name in actual:
        delta = (actual[name] - reference[name]).abs()
        result[name] = {
            "max_abs": float(delta.max().item()),
            "mean_abs": float(delta.mean().item()),
            "allclose_atol_1e_3_rtol_1e_3": bool(
                torch.allclose(
                    actual[name],
                    reference[name],
                    atol=1e-3,
                    rtol=1e-3,
                )
            ),
        }
    return result


def _device_average_s(fn: Any, repeats: int) -> float:
    import torch_npu

    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / 1000.0 / repeats


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.warmup_iters < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats positive")

    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("layout region TorchAir probe requires an NPU")
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu:0")

    model_dir = args.layout_model.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    frontend = OwnedLayoutFrontend(
        model_dir,
        device,
        graph_capture=False,
        model_backend="owned",
    )
    image_rgb, _decode_timing = _decode_rgb(image_path)
    pixel_values = frontend._prepare_pixel_values(image_rgb)
    core = frontend.model.model
    target = (
        core.decoder
        if args.region == "decoder"
        else core.encoder
    )
    captured_args, captured_kwargs = _capture_call(
        target,
        frontend.model,
        pixel_values,
    )
    torch_npu.npu.synchronize(device)
    region, inputs, output_names = _build_region(
        core,
        args.region,
        captured_args,
        captured_kwargs,
    )
    region.eval()

    eager_output = region(*inputs)
    torch_npu.npu.synchronize(device)
    eager_reference = _cpu_clone(output_names, eager_output)

    torchair, CompilerConfig = import_torchair()
    compiled = torchair.inference.cache_compile(
        region.forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
    )
    first_call_started = time.perf_counter()
    compiled_output = compiled(*inputs)
    torch_npu.npu.synchronize(device)
    first_call_s = time.perf_counter() - first_call_started
    compiled_reference = _cpu_clone(output_names, compiled_output)

    for _ in range(args.warmup_iters):
        compiled(*inputs)
    torch_npu.npu.synchronize(device)
    compiled_device_s = _device_average_s(
        lambda: compiled(*inputs),
        args.repeats,
    )
    eager_device_s = _device_average_s(
        lambda: region(*inputs),
        args.repeats,
    )
    result = {
        "model": str(model_dir),
        "image": str(image_path),
        "region": args.region,
        "cache_dir": str(cache_dir),
        "input_shapes": [list(tensor.shape) for tensor in inputs],
        "input_dtypes": [str(tensor.dtype) for tensor in inputs],
        "output_shapes": {
            name: list(tensor.shape)
            for name, tensor in zip(
                output_names,
                compiled_output,
                strict=True,
            )
        },
        "first_call_s": first_call_s,
        "warmup_iters": args.warmup_iters,
        "repeats": args.repeats,
        "eager_device_s": eager_device_s,
        "compiled_device_s": compiled_device_s,
        "speedup": eager_device_s / compiled_device_s,
        "diff": _diff(compiled_reference, eager_reference),
    }
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
