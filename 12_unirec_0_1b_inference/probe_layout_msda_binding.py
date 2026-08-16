#!/usr/bin/env python3
"""Validate the native ACLNN MSDA binding against the production decomposition."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401


SPATIAL_SHAPES = ((100, 100), (50, 50), (25, 25))
LEVEL_START_INDEX = (0, 10000, 12500)
BATCH = 1
NUM_QUERIES = 300
NUM_HEADS = 8
HEAD_DIM = 32
NUM_POINTS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-so", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args()


def msda_reference(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    del level_start_index
    batch, _, heads, head_dim = value.shape
    _, queries, _, levels, points, _ = sampling_locations.shape
    split_sizes = [int(height * width) for height, width in SPATIAL_SHAPES]
    value_levels = value.split(split_sizes, dim=1)
    sampling_grids = sampling_locations.mul(2).sub(1)
    sampled = []
    for level in range(levels):
        height, width = SPATIAL_SHAPES[level]
        value_level = (
            value_levels[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch * heads, head_dim, height, width)
        )
        sampling_grid = (
            sampling_grids[:, :, :, level]
            .transpose(1, 2)
            .flatten(0, 1)
        )
        sampled.append(
            F.grid_sample(
                value_level,
                sampling_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    weights = attention_weights.transpose(1, 2).reshape(
        batch * heads, 1, queries, levels * points
    )
    output = (
        torch.stack(sampled, dim=-2).flatten(-2).mul(weights).sum(-1)
    )
    return output.view(batch, heads * head_dim, queries).transpose(1, 2).contiguous()


def make_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260816)
    device = torch.device("npu:0")
    spatial_shapes = torch.tensor(
        SPATIAL_SHAPES, dtype=torch.int32, device=device
    )
    level_start_index = torch.tensor(
        LEVEL_START_INDEX, dtype=torch.int32, device=device
    )
    value = torch.randn(
        (BATCH, sum(h * w for h, w in SPATIAL_SHAPES), NUM_HEADS, HEAD_DIM),
        dtype=dtype,
        device=device,
    )
    sampling_locations = torch.rand(
        (BATCH, NUM_QUERIES, NUM_HEADS, len(SPATIAL_SHAPES), NUM_POINTS, 2),
        dtype=dtype,
        device=device,
    )
    weights = torch.randn(
        (BATCH, NUM_QUERIES, NUM_HEADS, len(SPATIAL_SHAPES), NUM_POINTS),
        dtype=dtype,
        device=device,
    )
    attention_weights = torch.softmax(weights.flatten(-2), dim=-1).view_as(weights)
    return (
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
    )


def benchmark(operation, inputs: tuple[torch.Tensor, ...], warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        operation(*inputs)
    torch.npu.synchronize()
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    started = time.perf_counter()
    start_event.record()
    for _ in range(repeats):
        operation(*inputs)
    end_event.record()
    torch.npu.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000.0
    return {
        "wall_ms_per_call": wall_ms / repeats,
        "event_ms_per_call": float(start_event.elapsed_time(end_event)) / repeats,
    }


def run_dtype(dtype: torch.dtype, warmup: int, repeats: int) -> dict[str, object]:
    label = str(dtype).removeprefix("torch.")
    result: dict[str, object] = {"dtype": label}
    try:
        inputs = make_inputs(dtype)
        native = torch.ops.unirec_layout.msda_aclnn(*inputs)
        reference = msda_reference(*inputs)
        torch.npu.synchronize()
        native_fp32 = native.float()
        reference_fp32 = reference.float()
        difference = (native_fp32 - reference_fp32).abs()
        cosine = F.cosine_similarity(
            native_fp32.flatten(), reference_fp32.flatten(), dim=0
        )
        result.update(
            {
                "status": "pass",
                "output_shape": list(native.shape),
                "output_dtype": str(native.dtype),
                "output_all_finite": bool(torch.isfinite(native).all().item()),
                "max_abs": float(difference.max().item()),
                "mean_abs": float(difference.mean().item()),
                "cosine": float(cosine.item()),
                "allclose_5e_2": bool(
                    torch.allclose(native_fp32, reference_fp32, atol=5e-2, rtol=5e-2)
                ),
                "native_timing": benchmark(
                    torch.ops.unirec_layout.msda_aclnn,
                    inputs,
                    warmup,
                    repeats,
                ),
                "reference_timing": benchmark(
                    msda_reference,
                    inputs,
                    warmup,
                    repeats,
                ),
            }
        )
        native_ms = result["native_timing"]["event_ms_per_call"]
        reference_ms = result["reference_timing"]["event_ms_per_call"]
        result["event_speedup"] = reference_ms / native_ms
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        torch.npu.synchronize()
    print(
        "UNIREC_LAYOUT_MSDA_BINDING_DTYPE "
        f"dtype={label} status={result['status']} "
        f"max_abs={result.get('max_abs')} mean_abs={result.get('mean_abs')} "
        f"cosine={result.get('cosine')} "
        f"native_ms={result.get('native_timing', {}).get('event_ms_per_call')} "
        f"reference_ms={result.get('reference_timing', {}).get('event_ms_per_call')} "
        f"speedup={result.get('event_speedup')}",
        flush=True,
    )
    return result


def main() -> None:
    args = parse_args()
    extension_so = args.extension_so.resolve()
    if not extension_so.is_file():
        raise FileNotFoundError(extension_so)
    torch.ops.load_library(str(extension_so))
    results = [
        run_dtype(torch.float16, args.warmup, args.repeats),
        run_dtype(torch.float32, args.warmup, args.repeats),
    ]
    overall = {
        "format": "unirec_layout_msda_binding_probe_v1",
        "extension_so": str(extension_so),
        "physical_device": __import__("os").environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "torch_npu": __import__("torch_npu").__version__,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "production_shape": {
            "value": [BATCH, 13125, NUM_HEADS, HEAD_DIM],
            "sampling_locations": [
                BATCH, NUM_QUERIES, NUM_HEADS, 3, NUM_POINTS, 2
            ],
            "attention_weights": [
                BATCH, NUM_QUERIES, NUM_HEADS, 3, NUM_POINTS
            ],
            "output": [BATCH, NUM_QUERIES, NUM_HEADS * HEAD_DIM],
        },
        "dtypes": results,
    }
    overall["status"] = (
        "pass"
        if results[0]["status"] == "pass"
        and results[0].get("output_all_finite")
        and results[0].get("allclose_5e_2")
        else "failed"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(overall, indent=2) + "\n")
    print(
        "UNIREC_LAYOUT_MSDA_BINDING "
        f"status={overall['status']} fp16={results[0]['status']} "
        f"fp32={results[1]['status']} output={args.output.resolve()}",
        flush=True,
    )
    if overall["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
