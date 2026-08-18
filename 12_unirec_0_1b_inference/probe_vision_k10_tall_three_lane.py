#!/usr/bin/env python3
"""Compare native, padded eager, and padded compiled vision on two K10 crops."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from layout_process_pool import _resize_recognition_compact_hwc_with_timing  # noqa: E402
from modeling_optimized_unirec import (  # noqa: E402
    OptimizedUniRecRunner,
    UniRecImageProcessor,
    synchronize_device,
)
from prefill_artifact import read_jsonl  # noqa: E402
from vision_bucket_presets import VisionBucketSpec  # noqa: E402
from vision_full_batch import (  # noqa: E402
    BucketedFullVisionRuntime,
    PreprocessedVisionInput,
    _compact_uint8_hwc_to_device,
    _make_host_masks,
)
from vision_prefix_crop_lab import _reconstruct_crops  # noqa: E402


CASES = (
    ("page_000033_crop_0001", VisionBucketSpec(1024, 704, 1)),
    ("page_000119_crop_0002", VisionBucketSpec(1024, 1408, 1)),
)
PROCESS_STARTED = time.perf_counter()


def phase(name: str, **fields: Any) -> None:
    print(
        "UNIREC_VISION_THREE_LANE_PHASE "
        + json.dumps(
            {
                "phase": name,
                "process_elapsed_s": time.perf_counter() - PROCESS_STARTED,
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--page-manifest", type=Path, required=True)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-replays", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=1)
    args = parser.parse_args()
    if args.warmup_replays < 0 or args.timing_repeats < 1:
        parser.error("warmup replays cannot be negative; repeats must be positive")
    return args


def physical_devices() -> list[int]:
    value = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not value:
        raise RuntimeError("set ASCEND_RT_VISIBLE_DEVICES before running")
    devices = [int(item) for item in value.split(",") if item.strip()]
    if any(device in {5, 6} for device in devices):
        raise RuntimeError("physical NPU 5 and 6 are excluded")
    return devices


def difference(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "shape_exact": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    left = reference.float()
    right = candidate.float()
    delta = right - left
    count = delta.numel()
    squared_sum = delta.square().sum(dtype=torch.float64)
    denominator = (
        left.square().sum().sqrt() * right.square().sum().sqrt()
    ).clamp_min(1e-12)
    return {
        "shape_exact": True,
        "shape": list(reference.shape),
        "exact": bool(torch.equal(reference, candidate)),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().sum(dtype=torch.float64).item() / count),
        "rmse": float((squared_sum / count).sqrt().item()),
        "cosine": float(((left * right).sum() / denominator).item()),
    }


def timed(fn: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, float]:
    synchronize_device("npu:0")
    started = time.perf_counter()
    output = fn()
    synchronize_device("npu:0")
    return output, (time.perf_counter() - started) * 1000.0


def compact_output(
    output: torch.Tensor,
    *,
    spec: VisionBucketSpec,
    width: int,
    height: int,
) -> torch.Tensor:
    grid = output.reshape(
        spec.batch_size,
        spec.height // 32,
        spec.width // 32,
        output.shape[-1],
    )
    return grid[
        :1,
        : height // 32,
        : width // 32,
    ].reshape(1, -1, output.shape[-1]).contiguous()


def main() -> None:
    args = parse_args()
    devices = physical_devices()
    import torch_npu

    # Match the production UniRec process.  Atlas 310P otherwise defaults to
    # eager JIT compilation, which turns the supposed raw-eager control into
    # additional shape-specific graph compilations.
    torch_npu.npu.set_compile_mode(jit_compile=False)
    sys.path.insert(0, str(args.openocr_root.expanduser().resolve()))
    from tools.utils.opendoc_onnx_utils.utils import tokenize_figure_of_table

    phase(
        "imports_and_arguments",
        physical_devices=devices,
        npu_jit_compile=False,
    )
    wanted = {request_id for request_id, _spec in CASES}
    by_id = {
        str(row["request_id"]): row
        for row in read_jsonl(args.crop_manifest.expanduser().resolve())
        if str(row["request_id"]) in wanted
    }
    missing = sorted(wanted - set(by_id))
    if missing:
        raise RuntimeError(f"crop manifest is missing request IDs: {missing}")
    rows = [by_id[request_id] for request_id, _spec in CASES]
    images = _reconstruct_crops(
        page_manifest=args.page_manifest.expanduser().resolve(),
        selected_rows=rows,
        tokenize_figure_of_table=tokenize_figure_of_table,
    )
    processor = UniRecImageProcessor()
    items: dict[str, PreprocessedVisionInput] = {}
    for row in rows:
        request_id = str(row["request_id"])
        pixels, _timing = _resize_recognition_compact_hwc_with_timing(
            images[request_id], processor=processor
        )
        expected = [
            int(value)
            for value in row["prefill"]["prep"]["processed_image_size"]
        ]
        actual = [int(pixels.shape[1]), int(pixels.shape[0])]
        if actual != expected:
            raise RuntimeError(f"processed shape changed for {request_id}: {actual} != {expected}")
        items[request_id] = PreprocessedVisionInput(
            source_index=len(items),
            pixel_values=pixels,
            original_image_size=tuple(int(value) for value in images[request_id].size),
            image_source=request_id,
        )
    phase(
        "crop_reconstruction_and_preprocess",
        shapes={
            request_id: [item.processed_width, item.processed_height]
            for request_id, item in items.items()
        },
    )

    runner = OptimizedUniRecRunner(
        model_path=args.model_path.expanduser().resolve(),
        device="npu:0",
        dtype="float16",
        compile_cache_dir=args.cache_dir.expanduser().resolve(),
    )
    phase("model_load")
    specs = tuple(spec for _request_id, spec in CASES)
    runtime = BucketedFullVisionRuntime(
        runner,
        specs=specs,
        diagnostic_graph_log=True,
        focal_depthwise_rewrite="constant_grouped_all",
        weight_format="torchair_internal",
        preset_name="k10_tall_three_lane",
    )
    inventory_before = runtime.cache_inventory()
    phase("graph_registration", graph_keys=[spec.key for spec in specs])

    prepared: dict[str, dict[str, Any]] = {}
    for request_id, spec in CASES:
        item = items[request_id]
        if not spec.accepts(item.processed_width, item.processed_height):
            raise RuntimeError(f"{request_id} does not fit {spec.key}")
        native_pixels = _compact_uint8_hwc_to_device(
            item.pixel_values.copy(), device=runner.device, dtype=runner.dtype
        )
        host_pixels = np.zeros((1, spec.height, spec.width, 3), dtype=np.uint8)
        host_pixels[0, : item.processed_height, : item.processed_width] = item.pixel_values
        host_pixel_mask = np.zeros((1, 1, spec.height, spec.width), dtype=np.uint8)
        host_pixel_mask[0, :, : item.processed_height, : item.processed_width] = 1
        with torch.inference_mode(False):
            pixels_uint8 = torch.from_numpy(host_pixels).to(runner.device)
            padded_pixels = pixels_uint8.permute(0, 3, 1, 2).to(torch.float32)
            padded_pixels.mul_(np.float32(2.0 / 255.0))
            padded_pixels.sub_(np.float32(1.0))
            padded_pixels = padded_pixels.to(runner.dtype).contiguous()
            pixel_mask = torch.from_numpy(host_pixel_mask).to(
                runner.device, dtype=runner.dtype
            )
            padded_pixels.mul_(pixel_mask)
            masks = tuple(
                torch.from_numpy(mask).to(runner.device)
                for mask in _make_host_masks(
                    [(item.processed_width, item.processed_height)], spec=spec
                )
            )
        prepared[request_id] = {
            "item": item,
            "spec": spec,
            "native_pixels": native_pixels,
            "padded_pixels": padded_pixels,
            "masks": masks,
        }
    phase("device_inputs_ready")

    def lanes(request_id: str) -> tuple[tuple[str, Callable[[], torch.Tensor]], ...]:
        row = prepared[request_id]
        spec = row["spec"]
        module = runtime.modules[spec.key]
        compiled = runtime.compiled[spec.key]
        return (
            ("native_unpadded_raw_eager", lambda: runner.model.forward_encoder(row["native_pixels"])),
            ("padded_masked_raw_eager", lambda: module(row["padded_pixels"], *row["masks"])),
            ("padded_masked_torchair", lambda: compiled(row["padded_pixels"], *row["masks"])),
        )

    with torch.inference_mode():
        for replay in range(args.warmup_replays):
            for request_id, _spec in CASES:
                for lane, fn in lanes(request_id):
                    phase("warmup_begin", replay=replay, request_id=request_id, lane=lane)
                    _output, elapsed_ms = timed(fn)
                    phase(
                        "warmup_end",
                        replay=replay,
                        request_id=request_id,
                        lane=lane,
                        synchronized_ms=elapsed_ms,
                    )

        result_rows = []
        for request_id, spec in CASES:
            item = items[request_id]
            outputs: dict[str, torch.Tensor] = {}
            timings: dict[str, list[float]] = {name: [] for name, _fn in lanes(request_id)}
            lane_list = lanes(request_id)
            for repeat in range(args.timing_repeats):
                ordered = lane_list if repeat % 2 == 0 else tuple(reversed(lane_list))
                for name, fn in ordered:
                    output, elapsed_ms = timed(fn)
                    if name != "native_unpadded_raw_eager":
                        output = compact_output(
                            output,
                            spec=spec,
                            width=item.processed_width,
                            height=item.processed_height,
                        )
                    outputs[name] = output
                    timings[name].append(elapsed_ms)
            native = outputs["native_unpadded_raw_eager"]
            padded = outputs["padded_masked_raw_eager"]
            compiled = outputs["padded_masked_torchair"]
            result_rows.append(
                {
                    "request_id": request_id,
                    "processed_size": [item.processed_width, item.processed_height],
                    "bucket": spec.key,
                    "comparisons": {
                        "padded_eager_vs_native": difference(native, padded),
                        "compiled_vs_padded_eager": difference(padded, compiled),
                        "compiled_vs_native": difference(native, compiled),
                    },
                    "timing_p50_ms": {
                        name: statistics.median(values) for name, values in timings.items()
                    },
                }
            )

    inventory_after = runtime.cache_inventory()
    report = {
        "schema": "unirec_vision_k10_tall_three_lane_v1",
        "status": "pass",
        "physical_devices": devices,
        "weights": {
            "focal_depthwise_rewrite": "constant_grouped_all",
            "weight_format": "torchair_internal",
        },
        "rows": result_rows,
        "cache_inventory_unchanged": inventory_before == inventory_after,
        "cache_inventory_before": inventory_before,
        "cache_inventory_after": inventory_after,
        "process_wall_s": time.perf_counter() - PROCESS_STARTED,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    phase("report_write", output=str(output))
    print("UNIREC_VISION_K10_TALL_THREE_LANE: PASS", flush=True)
    for row in result_rows:
        print("UNIREC_VISION_K10_TALL_THREE_LANE_ROW " + json.dumps(row), flush=True)
    print(f"OUTPUT_JSON={output}", flush=True)


if __name__ == "__main__":
    main()
