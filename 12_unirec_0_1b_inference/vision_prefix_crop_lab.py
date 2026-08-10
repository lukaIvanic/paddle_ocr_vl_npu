#!/usr/bin/env python3
"""Measure UniRec vision-prefix stages on exact singleton page crops."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from layout_process_pool import _decode_rgb, _prepare_frontend_payload  # noqa: E402
from modeling_optimized_unirec import (  # noqa: E402
    OptimizedUniRecRunner,
    synchronize_device,
)
from prefill_artifact import read_jsonl  # noqa: E402
from vision_atlas import UniRecVisionAtlasRuntime  # noqa: E402


STAGES = (
    "vision_prefix_convolution_stem",
    "vision_prefix_stage0_focal_blocks",
    "vision_prefix_stage0_downsample",
    "vision_prefix_stage1_focal_blocks",
    "vision_prefix_stage1_downsample",
    "vision_crop_prefix_stages_0_1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--page-manifest", type=Path, required=True)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be non-negative and --repeats positive")
    if len(set(args.request_id)) != len(args.request_id):
        parser.error("--request-id values must be unique")
    return args


def _physical_devices() -> list[int]:
    value = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not value:
        raise RuntimeError("source npu-setup before launching the crop lab")
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if 5 in devices:
        raise RuntimeError("physical NPU 5 is excluded from UniRec experiments")
    return devices


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary_ms(values_s: list[float]) -> dict[str, float]:
    values_ms = [value * 1000.0 for value in values_s]
    return {
        "min": min(values_ms),
        "p50": statistics.median(values_ms),
        "mean": statistics.fmean(values_ms),
        "p90": _percentile(values_ms, 0.9),
        "max": max(values_ms),
    }


def _load_selected_rows(
    crop_manifest: Path,
    request_ids: list[str],
) -> list[dict[str, Any]]:
    wanted = set(request_ids)
    rows = {
        str(row["request_id"]): row
        for row in read_jsonl(crop_manifest)
        if str(row["request_id"]) in wanted
    }
    missing = [request_id for request_id in request_ids if request_id not in rows]
    if missing:
        raise RuntimeError(f"crop manifest is missing request IDs: {missing}")
    return [rows[request_id] for request_id in request_ids]


def _reconstruct_crops(
    *,
    page_manifest: Path,
    selected_rows: list[dict[str, Any]],
    crop_margin: Any,
    tokenize_figure_of_table: Any,
) -> dict[str, Image.Image]:
    page_rows = {
        int(row["page_index"]): row for row in read_jsonl(page_manifest)
    }
    selected_by_page: dict[int, list[dict[str, Any]]] = {}
    for row in selected_rows:
        selected_by_page.setdefault(int(row["page_index"]), []).append(row)

    images: dict[str, Image.Image] = {}
    for page_index, crop_rows in selected_by_page.items():
        page = page_rows[page_index]
        path = Path(page["image_path"])
        rgb, _timing = _decode_rgb(path)
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        payload, _frontend_timing = _prepare_frontend_payload(
            page_index=page_index,
            path=path,
            bgr=bgr,
            layout_result=page["layout_results"],
            use_chart_recognition=True,
            crop_margin=crop_margin,
            tokenize_figure_of_table=tokenize_figure_of_table,
        )
        for row in crop_rows:
            crop_index = int(row["crop_index"])
            crop = payload["crops"][crop_index]
            if crop["label"] != row["label"]:
                raise RuntimeError(
                    f"label mismatch for {row['request_id']}: "
                    f"{crop['label']} != {row['label']}"
                )
            image = Image.fromarray(crop["image_rgb"])
            expected_size = tuple(row["prefill"]["prep"]["original_image_size"])
            if image.size != expected_size:
                raise RuntimeError(
                    f"crop-size mismatch for {row['request_id']}: "
                    f"{image.size} != {expected_size}"
                )
            images[str(row["request_id"])] = image
    return images


def main() -> None:
    args = parse_args()
    physical_devices = _physical_devices()
    sys.path.insert(0, str(args.openocr_root.expanduser().resolve()))
    from tools.utils.opendoc_onnx_utils.utils import (  # noqa: PLC0415
        crop_margin,
        tokenize_figure_of_table,
    )

    selected_rows = _load_selected_rows(
        args.crop_manifest.expanduser().resolve(),
        args.request_id,
    )
    images = _reconstruct_crops(
        page_manifest=args.page_manifest.expanduser().resolve(),
        selected_rows=selected_rows,
        crop_margin=crop_margin,
        tokenize_figure_of_table=tokenize_figure_of_table,
    )

    runner = OptimizedUniRecRunner(
        model_path=args.model_path.expanduser().resolve(),
        device="npu:0",
        dtype="float16",
        compile_cache_dir=args.cache_dir.expanduser().resolve(),
    )
    runtime = UniRecVisionAtlasRuntime(runner)
    prepared: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for row in selected_rows:
        request_id = str(row["request_id"])
        prepared[request_id] = runner.prepare_pil_image(
            images[request_id],
            image_source=request_id,
        )
        actual_size = prepared[request_id][1]["processed_image_size"]
        expected_size = row["prefill"]["prep"]["processed_image_size"]
        if actual_size != expected_size:
            raise RuntimeError(
                f"processed-size mismatch for {request_id}: "
                f"{actual_size} != {expected_size}"
            )

    # Compile/warm shared stage-2 and text-prefill graphs, then warm every
    # singleton shape before retaining measurements.
    for row in selected_rows:
        request_id = str(row["request_id"])
        for _ in range(args.warmup):
            runtime.prefill_prepared_packed_for_cohort(
                [prepared[request_id]],
                profile_device_stages=True,
                decode_ready=False,
            )

    results = []
    for row in selected_rows:
        request_id = str(row["request_id"])
        samples = {name: [] for name in STAGES}
        for _ in range(args.repeats):
            item = runtime.prefill_prepared_packed_for_cohort(
                [prepared[request_id]],
                profile_device_stages=True,
                decode_ready=False,
            )[0]
            timings = item.prefill_device_stage_s
            if timings is None:
                raise RuntimeError("profiled singleton returned no device timings")
            for name in STAGES:
                samples[name].append(float(timings[name]))
        results.append(
            {
                "request_id": request_id,
                "page_index": int(row["page_index"]),
                "crop_index": int(row["crop_index"]),
                "label": row["label"],
                "original_size": row["prefill"]["prep"]["original_image_size"],
                "processed_size": row["prefill"]["prep"]["processed_image_size"],
                "encoder_tokens": int(row["cross_kv"]["source_length"]),
                "timing_ms": {
                    name: _summary_ms(values) for name, values in samples.items()
                },
            }
        )

    synchronize_device("npu:0")
    report = {
        "status": "ok",
        "physical_devices": physical_devices,
        "worker_count": 1,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "stage_timing": "NPU events; one synchronization per singleton prefill",
        "results": results,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("UNIREC_VISION_PREFIX_CROP_LAB " + json.dumps(report), flush=True)
    print(f"OUTPUT_JSON={output}", flush=True)


if __name__ == "__main__":
    main()
