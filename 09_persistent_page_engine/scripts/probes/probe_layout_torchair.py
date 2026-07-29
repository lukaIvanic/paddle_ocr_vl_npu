#!/usr/bin/env python3
"""Probe a full fixed-shape PP-DocLayoutV3 TorchAir graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.compile_utils import import_torchair
from pipeline.layout_frontend import OwnedLayoutFrontend, _decode_rgb
from pipeline.layout_model_runtime import _mask_to_box_capture_friendly


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
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--model-backend",
        choices=("transformers", "owned"),
        default="transformers",
    )
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "eager"),
        default="sdpa",
    )
    return parser.parse_args()


def _output_tensors(output: Any) -> dict[str, torch.Tensor]:
    names = ("logits", "pred_boxes", "order_logits", "out_masks")
    tensors = {
        name: getattr(output, name)
        for name in names
        if getattr(output, name, None) is not None
    }
    if not tensors:
        raise RuntimeError("layout model returned no expected tensors")
    return tensors


def _cpu_clone(output: Any) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().float().cpu().clone()
        for name, tensor in _output_tensors(output).items()
    }


def _diff(
    lhs: dict[str, torch.Tensor],
    rhs: dict[str, torch.Tensor],
) -> dict[str, dict[str, float | bool]]:
    result: dict[str, dict[str, float | bool]] = {}
    for name in lhs.keys() & rhs.keys():
        delta = (lhs[name] - rhs[name]).abs()
        result[name] = {
            "max_abs": float(delta.max().item()),
            "mean_abs": float(delta.mean().item()),
            "allclose_atol_1e_3_rtol_1e_3": bool(
                torch.allclose(lhs[name], rhs[name], atol=1e-3, rtol=1e-3)
            ),
        }
    return result


def _device_average_s(
    fn,
    *,
    repeats: int,
) -> float:
    import torch_npu

    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / 1000.0 / repeats


def _record_anchor_call(model: Any) -> tuple[Any, dict[str, Any]]:
    """Record the concrete anchor arguments used by one eager forward."""

    detector = model.model
    original = detector.generate_anchors
    recorded: dict[str, Any] = {}

    def recording_generate_anchors(
        _self: Any,
        spatial_shapes: Any = None,
        grid_size: float = 0.05,
        device: Any = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recorded.update(
            {
                "spatial_shapes": tuple(
                    tuple(int(value) for value in shape)
                    for shape in spatial_shapes
                ),
                "grid_size": float(grid_size),
                "device": device,
                "dtype": dtype,
            }
        )
        return original(
            spatial_shapes=spatial_shapes,
            grid_size=grid_size,
            device=device,
            dtype=dtype,
        )

    detector.generate_anchors = MethodType(
        recording_generate_anchors,
        detector,
    )
    return original, recorded


def _install_static_anchor_cache(
    model: Any,
    *,
    original: Any,
    recorded: dict[str, Any],
) -> None:
    """Return the detector's invariant 800x800 anchors as captured tensors."""

    if not recorded:
        raise RuntimeError("eager forward did not generate detector anchors")
    detector = model.model
    anchors, valid_mask = detector.generate_anchors(
        **recorded,
    )

    def static_generate_anchors(
        _self: Any,
        spatial_shapes: Any = None,
        grid_size: float = 0.05,
        device: Any = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del spatial_shapes, grid_size, device, dtype
        return anchors, valid_mask

    detector.generate_anchors = MethodType(
        static_generate_anchors,
        detector,
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.warmup_iters < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats positive")

    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("layout TorchAir probe requires an NPU")
    torch.npu.set_compile_mode(jit_compile=False)
    # Transformers documents this switch for production compilation. The
    # detector input and spatial feature shapes are fixed in this probe.
    os.environ["TRANSFORMERS_DISABLE_TORCH_CHECK"] = "1"
    device = torch.device("npu:0")
    if args.model_backend == "transformers":
        from transformers.models.pp_doclayout_v3 import (
            modeling_pp_doclayout_v3,
        )

        modeling_pp_doclayout_v3.mask_to_box_coordinate = (
            _mask_to_box_capture_friendly
        )
    model_dir = args.layout_model.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    frontend = OwnedLayoutFrontend(
        model_dir,
        device,
        graph_capture=False,
        model_backend=args.model_backend,
    )
    if args.model_backend == "transformers":
        frontend.model.config._attn_implementation = (
            args.attention_implementation
        )
    image_rgb, _decode_timing = _decode_rgb(image_path)
    pixel_values = frontend._prepare_pixel_values(image_rgb)

    original_generate_anchors, recorded_anchor_call = _record_anchor_call(
        frontend.model
    )
    eager_output = frontend.model(pixel_values=pixel_values)
    torch_npu.npu.synchronize(device)
    eager_reference = _cpu_clone(eager_output)
    _install_static_anchor_cache(
        frontend.model,
        original=original_generate_anchors,
        recorded=recorded_anchor_call,
    )
    cached_anchor_output = frontend.model(pixel_values=pixel_values)
    torch_npu.npu.synchronize(device)
    cached_anchor_reference = _cpu_clone(cached_anchor_output)

    torchair, CompilerConfig = import_torchair()
    compile_started = time.perf_counter()
    compiled = torchair.inference.cache_compile(
        frontend.model.forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
    )
    compiled_output = compiled(pixel_values=pixel_values)
    torch_npu.npu.synchronize(device)
    first_call_s = time.perf_counter() - compile_started
    compiled_reference = _cpu_clone(compiled_output)

    for _ in range(args.warmup_iters):
        compiled(pixel_values=pixel_values)
    torch_npu.npu.synchronize(device)
    compiled_device_s = _device_average_s(
        lambda: compiled(pixel_values=pixel_values),
        repeats=args.repeats,
    )
    eager_device_s = _device_average_s(
        lambda: frontend.model(pixel_values=pixel_values),
        repeats=args.repeats,
    )

    source_paths = [EXPERIMENT_ROOT / "pipeline/layout_model_runtime.py"]
    if args.model_backend == "owned":
        source_paths.extend(
            [
                EXPERIMENT_ROOT
                / "pipeline/owned_layout_model/modeling.py",
                EXPERIMENT_ROOT
                / "pipeline/owned_layout_model/hgnet_v2.py",
                EXPERIMENT_ROOT
                / "pipeline/owned_layout_model/config.py",
            ]
        )
    source_hash = hashlib.sha256(
        b"".join(path.read_bytes() for path in source_paths)
    ).hexdigest()
    result = {
        "image": str(image_path),
        "model": str(model_dir),
        "model_backend": args.model_backend,
        "cache_dir": str(cache_dir),
        "attention_implementation": args.attention_implementation,
        "source_hash": source_hash,
        "first_call_s": first_call_s,
        "warmup_iters": args.warmup_iters,
        "repeats": args.repeats,
        "eager_device_s": eager_device_s,
        "compiled_device_s": compiled_device_s,
        "speedup": eager_device_s / compiled_device_s,
        "cached_anchor_diff": _diff(
            cached_anchor_reference,
            eager_reference,
        ),
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
