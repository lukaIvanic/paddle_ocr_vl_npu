#!/usr/bin/env python3
"""Compare Transformers and project-owned PP-DocLayoutV3 eager inference."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from PIL import Image
from torch import Tensor

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from pipeline.layout_model_runtime import (
    _decoder_forward_final_heads_only,
    _install_pp_doclayout_v3_npu_indexput_compat,
)
from pipeline.owned_layout_model import (
    OwnedPPDocLayoutV3ForObjectDetection,
)


DEFAULT_MODEL = Path("/workspace/models/PP-DocLayoutV3_safetensors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--device",
        choices=("cpu", "npu"),
        default="npu",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _prepare_inputs(processor: Any, image_path: Path) -> dict[str, Tensor]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        return processor(images=image, return_tensors="pt")


def _to_cpu_output(output: Any) -> dict[str, Tensor]:
    return {
        name: getattr(output, name).detach().cpu()
        for name in (
            "logits",
            "pred_boxes",
            "order_logits",
            "out_masks",
        )
    }


def _run(
    model: torch.nn.Module,
    pixel_values: Tensor,
    device: torch.device,
) -> tuple[dict[str, Tensor], float]:
    model.eval().to(device)
    pixel_values = pixel_values.to(device)
    if device.type == "npu":
        torch.npu.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(pixel_values=pixel_values)
    if device.type == "npu":
        torch.npu.synchronize()
    elapsed = time.perf_counter() - started
    return _to_cpu_output(output), elapsed


def _comparison(reference: Tensor, candidate: Tensor) -> dict[str, Any]:
    delta = (candidate - reference).abs()
    reference_abs = reference.abs()
    relative = delta / reference_abs.clamp_min(1e-6)
    return {
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "exact": bool(torch.equal(reference, candidate)),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "rmse": float(torch.sqrt(torch.mean(delta.square()))),
        "max_rel": float(relative.max()),
        "mean_rel": float(relative.mean()),
    }


def main() -> None:
    args = parse_args()
    model_dir = args.model.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.device == "npu":
        import torch_npu  # noqa: F401

        device = torch.device("npu:0")
        torch.npu.set_compile_mode(jit_compile=False)
    else:
        device = torch.device("cpu")

    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    processor = AutoImageProcessor.from_pretrained(model_dir)
    inputs = _prepare_inputs(processor, image_path)
    pixel_values = inputs["pixel_values"]

    oracle_started = time.perf_counter()
    oracle = AutoModelForObjectDetection.from_pretrained(model_dir)
    _install_pp_doclayout_v3_npu_indexput_compat(oracle)
    decoder = oracle.model.decoder
    decoder.forward = MethodType(
        _decoder_forward_final_heads_only,
        decoder,
    )
    decoder._layout_emit_masks = True
    oracle_setup_s = time.perf_counter() - oracle_started
    oracle_output, oracle_forward_s = _run(
        oracle,
        pixel_values,
        device,
    )
    del oracle
    if device.type == "npu":
        torch.npu.empty_cache()

    owned_started = time.perf_counter()
    owned = OwnedPPDocLayoutV3ForObjectDetection.from_pretrained(model_dir)
    owned_setup_s = time.perf_counter() - owned_started
    owned_output, owned_forward_s = _run(
        owned,
        pixel_values,
        device,
    )

    comparisons = {
        name: _comparison(oracle_output[name], owned_output[name])
        for name in oracle_output
    }
    report = {
        "image": str(image_path),
        "model": str(model_dir),
        "device": str(device),
        "input_shape": list(pixel_values.shape),
        "oracle_setup_s": oracle_setup_s,
        "oracle_forward_s": oracle_forward_s,
        "owned_setup_s": owned_setup_s,
        "owned_forward_s": owned_forward_s,
        "owned_load_report": owned.load_report,
        "outputs": comparisons,
    }
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
