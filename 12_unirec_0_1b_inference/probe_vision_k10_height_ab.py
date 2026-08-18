#!/usr/bin/env python3
"""Compare one K10 height bucket against its aligned-height control.

This is deliberately a small diagnostic.  It runs two exact crops with the
same processed 640x320 shape through:

* the current K10 960x448 B1 graph;
* a 960x512 B1 graph; and
* the unpadded eager vision encoder.

The suspect and control therefore use identical input bytes and model weights.
Only the compiled canvas height changes between the two bucket lanes.
"""

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

from layout_process_pool import (  # noqa: E402
    _resize_recognition_compact_hwc_with_timing,
)
from modeling_optimized_unirec import (  # noqa: E402
    OptimizedUniRecRunner,
    UniRecImageProcessor,
    synchronize_device,
)
from prefill_artifact import read_jsonl  # noqa: E402
from vision_bucket_presets import (  # noqa: E402
    VISION_BUCKETS_310P_K10_L4_ALL,
    VisionBucketSpec,
)
from vision_full_batch import (  # noqa: E402
    BucketedFullVisionRuntime,
    PreprocessedVisionInput,
    _compact_uint8_hwc_to_device,
)
from vision_prefix_crop_lab import _reconstruct_crops  # noqa: E402


CURRENT_KEY = "960x448_b1"
CONTROL_KEY = "960x512_b1"
DEFAULT_REQUEST_IDS = (
    "page_000032_crop_0002",  # 310P K10 mismatch
    "page_000032_crop_0004",  # same page and shape, known-good K10 control
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--page-manifest", type=Path, required=True)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-id", action="append")
    parser.add_argument("--warmup-replays", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=3)
    args = parser.parse_args()
    if args.warmup_replays < 1 or args.timing_repeats < 1:
        parser.error("warmup replays and timing repeats must be positive")
    request_ids = args.request_id or list(DEFAULT_REQUEST_IDS)
    if len(request_ids) != 2 or len(set(request_ids)) != 2:
        parser.error("provide exactly two unique --request-id values")
    args.request_id = request_ids
    return args


def _physical_devices() -> list[int]:
    value = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not value:
        raise RuntimeError("set ASCEND_RT_VISIBLE_DEVICES before launching")
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if any(device in {5, 6} for device in devices):
        raise RuntimeError("physical NPU 5 and 6 are excluded from UniRec runs")
    return devices


def _load_rows(path: Path, request_ids: list[str]) -> list[dict[str, Any]]:
    wanted = set(request_ids)
    by_id = {
        str(row["request_id"]): row
        for row in read_jsonl(path)
        if str(row["request_id"]) in wanted
    }
    missing = [request_id for request_id in request_ids if request_id not in by_id]
    if missing:
        raise RuntimeError(f"crop manifest is missing request IDs: {missing}")
    return [by_id[request_id] for request_id in request_ids]


def _probe_specs() -> tuple[VisionBucketSpec, ...]:
    # Preserve the K10-L4 graph-slot identities.  Slot 6 is the existing
    # 960x448 graph.  Slot 9 is replaced with the 960x512 graph that occupied
    # slot 9 in K10-L1.  Existing caches can therefore be reused on both chips.
    specs = list(VISION_BUCKETS_310P_K10_L4_ALL)
    if specs[6].key != CURRENT_KEY:
        raise RuntimeError(f"K10 slot 6 drifted: {specs[6].key}")
    specs[9] = VisionBucketSpec(
        960,
        512,
        1,
        planning_cost_ms=21.38,
    )
    if specs[9].key != CONTROL_KEY:
        raise RuntimeError(f"control slot 9 drifted: {specs[9].key}")
    return tuple(specs)


def _diff(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "shape_exact": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    reference_fp32 = reference.float()
    candidate_fp32 = candidate.float()
    delta = candidate_fp32 - reference_fp32
    squared = delta.square()
    dot = (reference_fp32 * candidate_fp32).sum()
    denominator = (
        reference_fp32.square().sum().sqrt()
        * candidate_fp32.square().sum().sqrt()
    ).clamp_min(1e-12)
    return {
        "shape_exact": True,
        "shape": list(reference.shape),
        "exact": bool(torch.equal(reference, candidate)),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rmse": float(squared.mean().sqrt().item()),
        "cosine": float((dot / denominator).item()),
    }


def _time_ms(fn: Callable[[], torch.Tensor], *, device: str) -> tuple[torch.Tensor, float]:
    synchronize_device(device)
    started = time.perf_counter()
    output = fn()
    synchronize_device(device)
    return output, (time.perf_counter() - started) * 1000.0


def main() -> None:
    args = parse_args()
    devices = _physical_devices()
    sys.path.insert(0, str(args.openocr_root.expanduser().resolve()))
    from tools.utils.opendoc_onnx_utils.utils import (  # noqa: PLC0415
        tokenize_figure_of_table,
    )

    page_manifest = args.page_manifest.expanduser().resolve()
    crop_manifest = args.crop_manifest.expanduser().resolve()
    rows = _load_rows(crop_manifest, args.request_id)
    images = _reconstruct_crops(
        page_manifest=page_manifest,
        selected_rows=rows,
        tokenize_figure_of_table=tokenize_figure_of_table,
    )

    processor = UniRecImageProcessor()
    items: list[PreprocessedVisionInput] = []
    for source_index, row in enumerate(rows):
        request_id = str(row["request_id"])
        pixels, _timing = _resize_recognition_compact_hwc_with_timing(
            images[request_id],
            processor=processor,
        )
        expected_size = [
            int(value)
            for value in row["prefill"]["prep"]["processed_image_size"]
        ]
        actual_size = [int(pixels.shape[1]), int(pixels.shape[0])]
        if actual_size != expected_size:
            raise RuntimeError(
                f"processed-size mismatch for {request_id}: "
                f"{actual_size} != {expected_size}"
            )
        if actual_size != [640, 320]:
            raise RuntimeError(
                f"height A/B expects 640x320 crops, got {request_id}={actual_size}"
            )
        items.append(
            PreprocessedVisionInput(
                source_index=source_index,
                pixel_values=pixels,
                original_image_size=tuple(
                    int(value) for value in images[request_id].size
                ),
                image_source=request_id,
            )
        )

    runner = OptimizedUniRecRunner(
        model_path=args.model_path.expanduser().resolve(),
        device="npu:0",
        dtype="float16",
        compile_cache_dir=args.cache_dir.expanduser().resolve(),
    )
    runtime = BucketedFullVisionRuntime(
        runner,
        specs=_probe_specs(),
        focal_depthwise_rewrite="constant_grouped_all",
        weight_format="torchair_internal",
        preset_name="k10_height_ab",
    )
    by_key = {spec.key: spec for spec in runtime.specs}

    def run_eager(item: PreprocessedVisionInput) -> torch.Tensor:
        pixels = _compact_uint8_hwc_to_device(
            item.pixel_values.copy(),
            device=runner.device,
            dtype=runner.dtype,
        )
        with torch.inference_mode():
            return runner.model.forward_encoder(pixels)

    def run_bucket(item: PreprocessedVisionInput, key: str) -> torch.Tensor:
        return runtime._run_bucket(by_key[key], [item])[0].hidden_states

    # Use the real crop data for warmup.  This catches cache-key and first-call
    # behavior without introducing synthetic content into the comparison.
    for _ in range(args.warmup_replays):
        for item in items:
            run_eager(item)
            run_bucket(item, CURRENT_KEY)
            run_bucket(item, CONTROL_KEY)
    synchronize_device(runner.device)

    result_rows = []
    all_timing: dict[str, list[float]] = {
        "eager": [],
        CURRENT_KEY: [],
        CONTROL_KEY: [],
    }
    for item in items:
        latest: dict[str, torch.Tensor] = {}
        timing: dict[str, list[float]] = {
            "eager": [],
            CURRENT_KEY: [],
            CONTROL_KEY: [],
        }
        lanes: tuple[tuple[str, Callable[[], torch.Tensor]], ...] = (
            ("eager", lambda item=item: run_eager(item)),
            (CURRENT_KEY, lambda item=item: run_bucket(item, CURRENT_KEY)),
            (CONTROL_KEY, lambda item=item: run_bucket(item, CONTROL_KEY)),
        )
        for repeat_index in range(args.timing_repeats):
            ordered = lanes if repeat_index % 2 == 0 else tuple(reversed(lanes))
            for name, fn in ordered:
                output, elapsed_ms = _time_ms(fn, device=runner.device)
                latest[name] = output
                timing[name].append(elapsed_ms)
                all_timing[name].append(elapsed_ms)
        result_rows.append(
            {
                "request_id": item.image_source,
                "processed_size": [item.processed_width, item.processed_height],
                "comparisons": {
                    "448_vs_eager": _diff(latest["eager"], latest[CURRENT_KEY]),
                    "512_vs_eager": _diff(latest["eager"], latest[CONTROL_KEY]),
                    "448_vs_512": _diff(
                        latest[CONTROL_KEY], latest[CURRENT_KEY]
                    ),
                },
                "timing_p50_ms": {
                    name: statistics.median(values)
                    for name, values in timing.items()
                },
            }
        )

    report = {
        "status": "pass",
        "physical_devices": devices,
        "probe": "same_640x320_crop_k10_448_vs_aligned_512",
        "weights": {
            "focal_depthwise_rewrite": "constant_grouped_all",
            "weight_format": "torchair_internal",
        },
        "graph_slots": {
            CURRENT_KEY: 6,
            CONTROL_KEY: 9,
        },
        "warmup_replays": args.warmup_replays,
        "timing_repeats": args.timing_repeats,
        "rows": result_rows,
        "aggregate_timing_p50_ms": {
            name: statistics.median(values)
            for name, values in all_timing.items()
        },
        "cache_dirs": {
            key: str(runtime.cache_dirs[key])
            for key in (CURRENT_KEY, CONTROL_KEY)
        },
        "cache_inventory": {
            key: runtime.cache_inventory()[key]
            for key in (CURRENT_KEY, CONTROL_KEY)
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("UNIREC_VISION_K10_HEIGHT_AB " + json.dumps(report), flush=True)
    print(f"OUTPUT_JSON={output}", flush=True)


if __name__ == "__main__":
    main()
