#!/usr/bin/env python3
"""Replay and profile the exact one-worker UniRec production vision boundary.

The measured call is ``BucketedFullVisionRuntime.encode``. This is the same
runtime invoked by ``layout_process_pool._prefill_worker_pages_bucketed`` in
the optimized full pipeline. Crop construction and compact uint8 HWC resizing
happen before the measured window, as they do in the production stage split.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from modeling_optimized_unirec import (  # noqa: E402
    OptimizedUniRecRunner,
    UniRecImageProcessor,
    synchronize_device,
)
from prefill_artifact import read_jsonl  # noqa: E402
from profile_prefill_graph_suite import (  # noqa: E402
    _parse_profile,
    _profiler_config,
)
from vision_full_batch import (  # noqa: E402
    DEFAULT_VISION_BUCKETS,
    BucketedFullVisionRuntime,
    PreprocessedVisionInput,
    _compact_uint8_hwc_to_device,
)
PRODUCTION_REFERENCE = {
    "chip": "Ascend 910B2",
    "commit": "f6d5ba8",
    "artifact": (
        "tmp/12_unirec_0_1b_inference/"
        "two_phase_slim_images_full1651_b128_f6d5ba8_20260811/"
        "output/run_summary.json"
    ),
    "pages": 1651,
    "workers": 8,
    "layout_batch_size": 2,
    "recognition_preprocess_threads": 8,
    "recognition_input_contract": "compact_uint8_hwc",
    "vision_page_lookahead": 4,
    "vision_runtime": "BucketedFullVisionRuntime",
    "sequential_core_pages_per_s": 9.8247498725005,
    "vision_bucket_calls": {
        "960x64_b16": 1580,
        "512x256_b16": 403,
        "960x256_b4": 925,
        "512x512_b8": 239,
        "960x512_b4": 400,
    },
    "vision_fallback_rows": 495,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--page-manifest", type=Path, required=True)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--page-offset", type=int, default=0)
    parser.add_argument("--page-limit", type=int, default=32)
    parser.add_argument(
        "--page-lookahead",
        type=int,
        default=4,
        help="Production is 4. Other values are deliberate batching experiments.",
    )
    parser.add_argument("--warmup-replays", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--parity-samples-per-route",
        type=int,
        default=1,
        help="Compare this many real crops per bucket/fallback route with eager.",
    )
    parser.add_argument(
        "--profile-scope",
        choices=("none", "group", "workload"),
        default="group",
    )
    parser.add_argument("--profile-group-index", type=int, default=0)
    parser.add_argument(
        "--profile-metric",
        choices=("pipe", "memory", "l2", "memory_access"),
        default="pipe",
    )
    parser.add_argument("--parser-topn", type=int, default=50)
    args = parser.parse_args(argv)
    if args.page_offset < 0 or args.page_limit < 1:
        parser.error("page offset must be non-negative and page limit positive")
    if args.page_lookahead < 1:
        parser.error("page lookahead must be positive")
    if args.warmup_replays < 0 or args.repeats < 1:
        parser.error("warmups cannot be negative and repeats must be positive")
    if args.parity_samples_per_route < 0:
        parser.error("parity samples cannot be negative")
    if args.profile_group_index < 0:
        parser.error("profile group index cannot be negative")
    if args.parser_topn < 1:
        parser.error("parser topn must be positive")
    return args


def _physical_devices() -> list[int]:
    value = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not value:
        raise RuntimeError("source npu-setup before launching the vision lab")
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(devices) != 1:
        raise RuntimeError(f"vision lab requires one visible NPU, got {devices}")
    if 5 in devices:
        raise RuntimeError("physical NPU 5 is excluded from UniRec experiments")
    return devices


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sample_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "p50": statistics.median(values),
        "mean": statistics.fmean(values),
        "p90": _percentile(values, 0.9),
        "max": max(values),
    }


def _select_manifest_rows(
    page_manifest: Path,
    crop_manifest: Path,
    *,
    page_offset: int,
    page_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_pages = sorted(read_jsonl(page_manifest), key=lambda row: int(row["page_index"]))
    pages = all_pages[page_offset : page_offset + page_limit]
    if len(pages) != page_limit:
        raise RuntimeError(
            f"requested {page_limit} pages at manifest offset {page_offset}, "
            f"found {len(pages)}"
        )
    page_indices = {int(row["page_index"]) for row in pages}
    crops = sorted(
        (
            row
            for row in read_jsonl(crop_manifest)
            if int(row["page_index"]) in page_indices
        ),
        key=lambda row: (int(row["page_index"]), int(row["crop_index"])),
    )
    if not crops:
        raise RuntimeError("selected production pages contain no accepted crops")
    return pages, crops


def _prepare_production_inputs(
    crop_rows: Sequence[dict[str, Any]],
    images: dict[str, Any],
    *,
    processor: UniRecImageProcessor,
    resize_compact: Callable[..., tuple[np.ndarray, dict[str, float]]],
) -> tuple[dict[int, list[PreprocessedVisionInput]], dict[str, float]]:
    by_page: dict[int, list[PreprocessedVisionInput]] = {}
    timing_s: dict[str, float] = {}
    source_index = 0
    for row in crop_rows:
        request_id = str(row["request_id"])
        pixels, detail = resize_compact(
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
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise RuntimeError(
                f"production compact contract failed for {request_id}: "
                f"{pixels.dtype} {pixels.shape}"
            )
        for name, value in detail.items():
            timing_s[name] = timing_s.get(name, 0.0) + float(value)
        item = PreprocessedVisionInput(
            source_index=source_index,
            pixel_values=pixels,
            original_image_size=tuple(int(value) for value in images[request_id].size),
            image_source=request_id,
        )
        by_page.setdefault(int(row["page_index"]), []).append(item)
        source_index += 1
    return by_page, timing_s


def _make_page_groups(
    pages: Sequence[dict[str, Any]],
    inputs_by_page: dict[int, list[PreprocessedVisionInput]],
    *,
    page_lookahead: int,
) -> list[list[PreprocessedVisionInput]]:
    groups = []
    for start in range(0, len(pages), page_lookahead):
        page_group = []
        for page in pages[start : start + page_lookahead]:
            page_group.extend(inputs_by_page.get(int(page["page_index"]), ()))
        if page_group:
            groups.append(
                [
                    PreprocessedVisionInput(
                        source_index=source_index,
                        pixel_values=item.pixel_values,
                        original_image_size=item.original_image_size,
                        image_source=item.image_source,
                    )
                    for source_index, item in enumerate(page_group)
                ]
            )
    if not groups:
        raise RuntimeError("production page grouping produced no vision work")
    return groups


def _runtime_stats_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    return {
        "bucket_calls": {
            key: int(after["bucket_calls"][key]) - int(before["bucket_calls"][key])
            for key in after["bucket_calls"]
        },
        "bucket_real_rows": {
            key: int(after["bucket_real_rows"][key])
            - int(before["bucket_real_rows"][key])
            for key in after["bucket_real_rows"]
        },
        "fallback_rows": int(after["fallback_rows"]) - int(before["fallback_rows"]),
        "compact_input_rows": int(after["compact_input_rows"])
        - int(before["compact_input_rows"]),
        "legacy_input_rows": int(after["legacy_input_rows"])
        - int(before["legacy_input_rows"]),
    }


def _run_groups(
    runtime: BucketedFullVisionRuntime,
    groups: Sequence[Sequence[PreprocessedVisionInput]],
) -> int:
    output_count = 0
    for group in groups:
        outputs = runtime.encode(group)
        output_count += len(outputs)
        del outputs
    return output_count


def _measure_replay(
    run: Callable[[], int],
    *,
    device: str,
) -> dict[str, float | int]:
    import torch_npu

    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    synchronize_device(device)
    started = time.perf_counter()
    start.record()
    output_count = run()
    end.record()
    end.synchronize()
    return {
        "synchronized_wall_ms": (time.perf_counter() - started) * 1000.0,
        "device_timeline_ms": float(start.elapsed_time(end)),
        "output_count": output_count,
    }


def _parity_check(
    runtime: BucketedFullVisionRuntime,
    runner: OptimizedUniRecRunner,
    groups: Sequence[Sequence[PreprocessedVisionInput]],
    *,
    samples_per_route: int,
) -> dict[str, Any]:
    if samples_per_route == 0:
        return {"status": "disabled", "rows": []}
    route_counts: dict[str, int] = {}
    selected: list[tuple[PreprocessedVisionInput, torch.Tensor, str]] = []
    for group in groups:
        outputs = runtime.encode(group)
        for item, output in zip(group, outputs):
            route = output.bucket_key or "fallback_eager"
            count = route_counts.get(route, 0)
            if count < samples_per_route:
                selected.append((item, output.hidden_states, route))
                route_counts[route] = count + 1
        if len(route_counts) == len(DEFAULT_VISION_BUCKETS) + 1 and all(
            count >= samples_per_route for count in route_counts.values()
        ):
            break

    rows = []
    with torch.inference_mode():
        for item, actual, route in selected:
            pixels = _compact_uint8_hwc_to_device(
                item.pixel_values,
                device=runner.device,
                dtype=runner.dtype,
            )
            expected = runner.model.forward_encoder(pixels)
            difference = (actual - expected).abs()
            rows.append(
                {
                    "request_id": item.image_source,
                    "route": route,
                    "processed_size": [item.processed_width, item.processed_height],
                    "allclose_atol_5e_2_rtol_5e_2": bool(
                        torch.allclose(actual, expected, atol=5e-2, rtol=5e-2)
                    ),
                    "max_abs": float(difference.max().item()),
                    "mean_abs": float(difference.mean().item()),
                }
            )
    passed = bool(rows) and all(
        row["allclose_atol_5e_2_rtol_5e_2"] for row in rows
    )
    return {
        "status": "passed" if passed else "failed",
        "sample_count": len(rows),
        "routes": route_counts,
        "rows": rows,
    }


def _profile(
    run: Callable[[], int],
    *,
    output_dir: Path,
    metric: str,
    parser_topn: int,
    device: str,
) -> dict[str, Any]:
    import torch_npu.profiler as npu_prof

    profile_dir = output_dir / f"profile_{metric}"
    profile_dir.mkdir(parents=True, exist_ok=False)
    schedule = npu_prof.schedule(wait=0, warmup=0, active=1, repeat=1)
    synchronize_device(device)
    with npu_prof.profile(
        activities=[
            npu_prof.ProfilerActivity.CPU,
            npu_prof.ProfilerActivity.NPU,
        ],
        schedule=schedule,
        experimental_config=_profiler_config(metric),
        on_trace_ready=npu_prof.tensorboard_trace_handler(
            str(profile_dir), analyse_flag=True
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        with torch.profiler.record_function("unirec.production_vision_encode"):
            output_count = run()
        synchronize_device(device)
        profiler.step()
    return {
        "output_count": output_count,
        "profile_dir": str(profile_dir),
        "parsed": _parse_profile(profile_dir, topn=parser_topn),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    physical_devices = _physical_devices()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    pages, crop_rows = _select_manifest_rows(
        args.page_manifest.expanduser().resolve(),
        args.crop_manifest.expanduser().resolve(),
        page_offset=args.page_offset,
        page_limit=args.page_limit,
    )

    sys.path.insert(0, str(args.openocr_root.expanduser().resolve()))
    from tools.utils.opendoc_onnx_utils.utils import (  # noqa: PLC0415
        crop_margin,
        tokenize_figure_of_table,
    )
    from layout_process_pool import (  # noqa: PLC0415
        _resize_recognition_compact_hwc_with_timing,
    )
    from vision_prefix_crop_lab import _reconstruct_crops  # noqa: PLC0415

    images = _reconstruct_crops(
        page_manifest=args.page_manifest.expanduser().resolve(),
        selected_rows=crop_rows,
        crop_margin=crop_margin,
        tokenize_figure_of_table=tokenize_figure_of_table,
    )
    processor = UniRecImageProcessor()
    inputs_by_page, crop_prepare_s = _prepare_production_inputs(
        crop_rows,
        images,
        processor=processor,
        resize_compact=_resize_recognition_compact_hwc_with_timing,
    )
    del images
    groups = _make_page_groups(
        pages,
        inputs_by_page,
        page_lookahead=args.page_lookahead,
    )

    runner = OptimizedUniRecRunner(
        model_path=args.model_path.expanduser().resolve(),
        device="npu:0",
        dtype="float16",
        compile_cache_dir=args.cache_dir.expanduser().resolve(),
    )
    runtime = BucketedFullVisionRuntime(runner)
    actual_buckets = [spec.key for spec in runtime.specs]
    expected_buckets = [spec.key for spec in DEFAULT_VISION_BUCKETS]
    if actual_buckets != expected_buckets:
        raise RuntimeError(
            f"production bucket drift: {actual_buckets} != {expected_buckets}"
        )

    graph_warmup = runtime.warmup_all(passes=1)
    workload = lambda: _run_groups(runtime, groups)
    for _ in range(args.warmup_replays):
        workload()
    synchronize_device("npu:0")

    before_stats = copy.deepcopy(runtime.summary())
    samples = [
        _measure_replay(workload, device="npu:0") for _ in range(args.repeats)
    ]
    after_stats = copy.deepcopy(runtime.summary())
    stats_delta = _runtime_stats_delta(before_stats, after_stats)
    expected_outputs = len(crop_rows)
    if any(int(sample["output_count"]) != expected_outputs for sample in samples):
        raise RuntimeError("production vision replay lost outputs")

    parity = _parity_check(
        runtime,
        runner,
        groups,
        samples_per_route=args.parity_samples_per_route,
    )
    profile = None
    if args.profile_scope != "none":
        if args.profile_scope == "group":
            if args.profile_group_index >= len(groups):
                raise RuntimeError(
                    f"profile group {args.profile_group_index} is outside "
                    f"the {len(groups)} available groups"
                )
            selected_groups = [groups[args.profile_group_index]]
        else:
            selected_groups = groups
        profile = _profile(
            lambda: _run_groups(runtime, selected_groups),
            output_dir=output_dir,
            metric=args.profile_metric,
            parser_topn=args.parser_topn,
            device="npu:0",
        )

    wall_samples = [float(row["synchronized_wall_ms"]) for row in samples]
    device_samples = [float(row["device_timeline_ms"]) for row in samples]
    physical_rows_per_replay = sum(
        stats_delta["bucket_calls"][spec.key] * spec.batch_size
        for spec in runtime.specs
    ) // args.repeats
    real_rows_per_replay = expected_outputs - (
        stats_delta["fallback_rows"] // args.repeats
    )
    report = {
        "status": "ok" if parity["status"] != "failed" else "correctness_failed",
        "physical_devices": physical_devices,
        "production_reference": PRODUCTION_REFERENCE,
        "production_contract": {
            "worker_count": 1,
            "measured_boundary": "BucketedFullVisionRuntime.encode",
            "input_contract": "compact_uint8_hwc",
            "normalization": "_compact_uint8_hwc_to_device",
            "page_lookahead": args.page_lookahead,
            "page_lookahead_matches_production": (
                args.page_lookahead
                == PRODUCTION_REFERENCE["vision_page_lookahead"]
            ),
            "buckets": actual_buckets,
            "runtime_source": str(Path(runtime.__class__.__module__.replace(".", "/"))),
            "graph_cache_dirs": {
                key: str(path) for key, path in runtime.cache_dirs.items()
            },
            "code_reuse": [
                "layout_process_pool._resize_recognition_compact_hwc_with_timing",
                "vision_full_batch.PreprocessedVisionInput",
                "vision_full_batch.BucketedFullVisionRuntime.encode",
                "vision_full_batch._compact_uint8_hwc_to_device",
            ],
        },
        "workload": {
            "page_offset": args.page_offset,
            "page_count": len(pages),
            "page_group_count": len(groups),
            "page_group_size_histogram": {
                str(size): sum(
                    1
                    for start in range(0, len(pages), args.page_lookahead)
                    if len(pages[start : start + args.page_lookahead]) == size
                )
                for size in sorted(
                    {
                        len(pages[start : start + args.page_lookahead])
                        for start in range(0, len(pages), args.page_lookahead)
                    }
                )
            },
            "crop_count": expected_outputs,
            "crop_prepare_s": crop_prepare_s,
            "repeats": args.repeats,
            "bucket_calls_per_replay": {
                key: value // args.repeats
                for key, value in stats_delta["bucket_calls"].items()
            },
            "bucket_real_rows_per_replay": {
                key: value // args.repeats
                for key, value in stats_delta["bucket_real_rows"].items()
            },
            "fallback_rows_per_replay": (
                stats_delta["fallback_rows"] // args.repeats
            ),
            "compiled_real_rows_per_replay": real_rows_per_replay,
            "compiled_physical_rows_per_replay": physical_rows_per_replay,
            "compiled_slot_efficiency": (
                real_rows_per_replay / physical_rows_per_replay
                if physical_rows_per_replay
                else None
            ),
        },
        "graph_warmup": graph_warmup,
        "timing_ms": {
            "production_boundary_wall": _sample_summary(wall_samples),
            "device_timeline_span": _sample_summary(device_samples),
        },
        "throughput": {
            "crops_per_s_wall_p50": (
                expected_outputs * 1000.0 / statistics.median(wall_samples)
            ),
            "pages_per_s_wall_p50": (
                len(pages) * 1000.0 / statistics.median(wall_samples)
            ),
        },
        "parity": parity,
        "profile_scope": args.profile_scope,
        "profile_group_index": (
            args.profile_group_index if args.profile_scope == "group" else None
        ),
        "profile": profile,
        "measurement_scope": {
            "included": [
                "host fixed-bucket materialization",
                "compact uint8 HWC H2D",
                "NPU float conversion normalization and BCHW transpose",
                "production bucket routing and partial-batch padding",
                "the exact compiled masked full-vision graphs",
                "production output compaction",
                "faithful eager fallback for oversized crops",
            ],
            "excluded": [
                "page decode and layout",
                "crop construction and bicubic resize",
                "packed text prefill and cross-KV export",
            ],
        },
    }
    output_path = output_dir / "vision_production_lab.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "UNIREC_PRODUCTION_VISION_LAB "
        f"status={report['status']} pages={len(pages)} crops={expected_outputs} "
        f"groups={len(groups)} wall_p50_ms={statistics.median(wall_samples):.3f} "
        f"device_p50_ms={statistics.median(device_samples):.3f} "
        f"crops_s={report['throughput']['crops_per_s_wall_p50']:.1f} "
        f"slot_eff={report['workload']['compiled_slot_efficiency']:.3f} "
        f"parity={parity['status']}",
        flush=True,
    )
    print(f"OUTPUT_JSON={output_path}", flush=True)
    if report["status"] != "ok":
        raise RuntimeError("production vision lab failed parity")


if __name__ == "__main__":
    main()
