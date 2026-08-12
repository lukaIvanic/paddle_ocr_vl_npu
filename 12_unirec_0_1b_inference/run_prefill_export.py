#!/usr/bin/env python3
"""Run the full UniRec producer without decoding, optionally exporting cross-KV."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

from layout_process_pool import DynamicLayoutProcessPool, SharedPageLease
from prefill_artifact import (
    CrossKvArtifactWriter,
    CrossKvDiscardSink,
    read_crop_array,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/workspace/models/unirec-0.1b"),
    )
    parser.add_argument(
        "--layout-model",
        type=Path,
        default=Path("/workspace/models/PP-DocLayoutV2_safetensors"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/workspace/datasets/OmniDocBench/images"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-storage",
        choices=("persistent", "discard"),
        default="persistent",
        help=(
            "Persist cross-KV with CRCs for decode replay, or validate and "
            "immediately release each shared payload for producer timing."
        ),
    )
    parser.add_argument("--offset", type=int, default=769)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup-pages", type=int, default=2)
    parser.add_argument(
        "--warmup-repeats",
        type=int,
        default=1,
        help="Repeat the warmup page set in the same persistent worker pool.",
    )
    parser.add_argument("--layout-threshold", type=float, default=0.4)
    parser.add_argument(
        "--layout-execution",
        choices=("eager", "torchair"),
        default="eager",
    )
    parser.add_argument(
        "--layout-dtype",
        choices=("float16", "float32"),
        default="float32",
    )
    parser.add_argument(
        "--layout-weight-format",
        choices=(
            "native",
            "depthwise_fz",
            "torchair_internal",
            "torchair_internal_depthwise_fz",
        ),
        default="native",
    )
    parser.add_argument(
        "--layout-depthwise-rewrite",
        choices=("native", "group16", "group32", "group64", "dense"),
        default="native",
    )
    parser.add_argument(
        "--layout-preformat-frozen-bn-buffers",
        action="store_true",
        help=(
            "Store the original FrozenBN buffers as NC1HWC0 without changing "
            "the FrozenBN expression."
        ),
    )
    parser.add_argument("--layout-batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=("float16",), default="float16")
    parser.add_argument("--cross-cache-length", type=int, default=512)
    parser.add_argument(
        "--layout-cache-dir",
        type=Path,
        default=Path(
            ".runtime_cache/12_unirec_0_1b_inference/layout_process_pool"
        ),
    )
    parser.add_argument(
        "--recognition-cache-dir",
        type=Path,
        default=Path(
            ".runtime_cache/12_unirec_0_1b_inference/opendoc_batched_decode_a372dbf"
        ),
    )
    parser.add_argument(
        "--vision-prefix-shapes-manifest",
        type=Path,
        help=(
            "Compile and dispatch the stages-0/1 vision prefix by every "
            "processed shape in a JSON shape list or prior crop JSONL."
        ),
    )
    parser.add_argument(
        "--vision-full-batches",
        action="store_true",
        help=(
            "Run the complete vision encoder through five masked fixed-canvas "
            "batch graphs before page-local packed text prefill."
        ),
    )
    parser.add_argument(
        "--vision-focal-depthwise-rewrite",
        choices=(
            "native",
            "constant",
            "constant_grouped",
            "constant_grouped_all",
            "group16",
            "aligned_spatial",
        ),
        default="native",
        help=(
            "Select the focal-depthwise implementation inside every fixed "
            "full-vision bucket graph."
        ),
    )
    parser.add_argument(
        "--vision-weight-format",
        choices=("native", "focal_prepack", "torchair_internal"),
        default="native",
        help="Select the full-vision Conv/Linear weight preparation lane.",
    )
    parser.add_argument(
        "--recognition-input-contract",
        choices=("compact_uint8_hwc", "legacy_float32_bchw"),
        default="compact_uint8_hwc",
        help=(
            "Choose whether full-vision workers normalize crops on the NPU "
            "or retain the prior CPU float32 BCHW path."
        ),
    )
    parser.add_argument(
        "--recognition-preprocess-threads",
        type=int,
        default=1,
        help="Persistent crop-resize threads inside each process worker.",
    )
    parser.add_argument(
        "--vision-page-lookahead",
        type=int,
        default=4,
        help="Maximum pages one worker may combine into local vision batches.",
    )
    parser.add_argument(
        "--no-chart-recognition",
        dest="use_chart_recognition",
        action="store_false",
        help="Treat chart regions as images instead of recognition crops.",
    )
    parser.set_defaults(use_chart_recognition=True)
    parser.add_argument("--worker-empty-cache-after-page", action="store_true")
    parser.add_argument(
        "--no-retain-shared-images",
        dest="retain_shared_images",
        action="store_false",
        help=(
            "Exclude page/crop image arrays from worker shared-memory payloads. "
            "Use this for inference-only prefill timing and CPU-RAM decode banks."
        ),
    )
    parser.set_defaults(retain_shared_images=True)
    parser.add_argument(
        "--profile-prefill-device-stages",
        action="store_true",
        help=(
            "Record low-overhead NPU event timings for vision and cross-KV "
            "prefill stages. Events synchronize once per packed prefill group."
        ),
    )
    args = parser.parse_args()
    if args.offset < 0 or args.limit < 1:
        parser.error("--offset must be non-negative and --limit positive")
    if args.workers < 1 or args.warmup_pages < 0 or args.warmup_repeats < 1:
        parser.error(
            "--workers and --warmup-repeats must be positive and "
            "--warmup-pages non-negative"
        )
    if args.cross_cache_length < 1:
        parser.error("--cross-cache-length must be positive")
    if args.recognition_preprocess_threads < 1:
        parser.error("--recognition-preprocess-threads must be positive")
    if args.vision_page_lookahead < 1:
        parser.error("--vision-page-lookahead must be positive")
    if args.layout_batch_size < 1:
        parser.error("--layout-batch-size must be positive")
    if args.layout_batch_size > args.vision_page_lookahead:
        parser.error(
            "--layout-batch-size cannot exceed --vision-page-lookahead"
        )
    if args.layout_batch_size > 1 and not args.vision_full_batches:
        parser.error("--layout-batch-size > 1 requires --vision-full-batches")
    if args.vision_full_batches and args.vision_prefix_shapes_manifest is not None:
        parser.error(
            "--vision-full-batches cannot be combined with "
            "--vision-prefix-shapes-manifest"
        )
    if not args.vision_full_batches and (
        args.vision_focal_depthwise_rewrite != "native"
        or args.vision_weight_format != "native"
    ):
        parser.error(
            "vision rewrite/weight-format controls require "
            "--vision-full-batches"
        )
    return args


def _physical_devices() -> list[int]:
    value = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not value:
        raise RuntimeError(
            "ASCEND_RT_VISIBLE_DEVICES is unset; source npu-setup before launch"
        )
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exception:
        raise RuntimeError(
            f"cannot parse ASCEND_RT_VISIBLE_DEVICES={value!r}"
        ) from exception


def _discard_payload(payload: dict[str, Any]) -> None:
    shared = payload.get("shared_memory")
    if shared is None and not payload.get("crops"):
        return
    if not isinstance(shared, dict):
        raise RuntimeError("warmup payload has no shared-memory arena")
    lease = SharedPageLease(str(shared["name"]))
    lease.close()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def main() -> None:
    args = parse_args()
    physical_devices = _physical_devices()
    if 5 in physical_devices:
        raise RuntimeError("physical NPU 5 is excluded from UniRec experiments")
    os.environ["UNIREC_STATIC_CROSS_CACHE_LEN"] = str(args.cross_cache_length)
    os.environ["UNIREC_RECOGNITION_INPUT_CONTRACT"] = (
        args.recognition_input_contract
    )
    os.environ["UNIREC_RECOGNITION_PREPROCESS_THREADS"] = str(
        args.recognition_preprocess_threads
    )

    openocr_root = args.openocr_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    layout_model = args.layout_model.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    layout_cache_dir = args.layout_cache_dir.expanduser().resolve()
    recognition_cache_dir = args.recognition_cache_dir.expanduser().resolve()
    vision_prefix_shapes_manifest = (
        args.vision_prefix_shapes_manifest.expanduser().resolve()
        if args.vision_prefix_shapes_manifest is not None
        else None
    )

    sys.path.insert(0, str(openocr_root))
    from tools.utils.utility import get_image_file_list

    all_paths = [
        Path(path).resolve()
        for path in sorted(get_image_file_list(str(input_path)))
    ]
    image_paths = all_paths[args.offset : args.offset + args.limit]
    if len(image_paths) != args.limit:
        raise RuntimeError(
            f"requested {args.limit} pages at offset {args.offset}, "
            f"found {len(image_paths)}"
        )

    print(
        "UNIREC_PREFILL_EXPORT_BEGIN "
        f"pages={len(image_paths)} offset={args.offset} workers={args.workers} "
        f"physical_devices={physical_devices} cross_cache={args.cross_cache_length}",
        flush=True,
    )
    total_started = time.perf_counter()
    pool = DynamicLayoutProcessPool(
        worker_count=args.workers,
        model_path=layout_model,
        cache_dir=layout_cache_dir,
        threshold=args.layout_threshold,
        execution=args.layout_execution,
        warmup_paths=image_paths[: args.workers],
        layout_dtype=args.layout_dtype,
        layout_weight_format=args.layout_weight_format,
        layout_depthwise_rewrite=args.layout_depthwise_rewrite,
        layout_preformat_frozen_bn_buffers=(
            args.layout_preformat_frozen_bn_buffers
        ),
        layout_batch_size=args.layout_batch_size,
        openocr_root=openocr_root,
        prepare_pages=True,
        use_chart_recognition=args.use_chart_recognition,
        prefill_recognition=True,
        recognition_model_path=model_path,
        recognition_dtype=args.dtype,
        recognition_cache_dir=recognition_cache_dir,
        recognition_prefix_shapes_manifest=vision_prefix_shapes_manifest,
        recognition_full_vision_buckets=args.vision_full_batches,
        recognition_vision_focal_depthwise_rewrite=(
            args.vision_focal_depthwise_rewrite
        ),
        recognition_vision_weight_format=args.vision_weight_format,
        recognition_page_lookahead=args.vision_page_lookahead,
        empty_cache_after_page=args.worker_empty_cache_after_page,
        profile_prefill_device_stages=args.profile_prefill_device_stages,
        retain_shared_images=args.retain_shared_images,
    )
    setup_s = pool.setup_wall_s
    warmup_summaries = []
    writer = (
        CrossKvArtifactWriter(output_dir)
        if args.artifact_storage == "persistent"
        else CrossKvDiscardSink(output_dir)
    )
    try:
        if args.warmup_pages:
            for repeat_index in range(args.warmup_repeats):
                for payload in pool.iter_map(
                    image_paths[: args.warmup_pages],
                    label=f"prefill_export_warmup_{repeat_index + 1}",
                ):
                    _discard_payload(payload)
                warmup_summary = pool.last_stream_summary
                if warmup_summary is None:
                    raise RuntimeError("producer warmup stream has no timing summary")
                warmup_summaries.append(warmup_summary)
        measured_started = time.perf_counter()
        for payload in pool.iter_map(image_paths, label="prefill_export_measured"):
            writer.add_page(payload)
        stream_wall_s = time.perf_counter() - measured_started
        worker_summary = pool.last_stream_summary
        if worker_summary is None:
            raise RuntimeError("producer worker stream has no timing summary")
        result = writer.finish(
            {
                "status": "ok",
                "model_path": str(model_path),
                "layout_model": str(layout_model),
                "input": str(input_path),
                "artifact_storage": args.artifact_storage,
                "offset": args.offset,
                "limit": args.limit,
                "workers": args.workers,
                "physical_devices": physical_devices,
                "dtype": args.dtype,
                "cross_cache_length": args.cross_cache_length,
                "layout_execution": args.layout_execution,
                "layout_dtype": args.layout_dtype,
                "layout_weight_format": args.layout_weight_format,
                "layout_depthwise_rewrite": args.layout_depthwise_rewrite,
                "layout_preformat_frozen_bn_buffers": (
                    args.layout_preformat_frozen_bn_buffers
                ),
                "layout_batch_size": args.layout_batch_size,
                "vision_prefix_shapes_manifest": (
                    str(vision_prefix_shapes_manifest)
                    if vision_prefix_shapes_manifest is not None
                    else None
                ),
                "vision_full_batches": args.vision_full_batches,
                "vision_focal_depthwise_rewrite": (
                    args.vision_focal_depthwise_rewrite
                ),
                "vision_weight_format": args.vision_weight_format,
                "recognition_input_contract": args.recognition_input_contract,
                "recognition_preprocess_threads": (
                    args.recognition_preprocess_threads
                ),
                "vision_page_lookahead": args.vision_page_lookahead,
                "use_chart_recognition": args.use_chart_recognition,
                "profile_prefill_device_stages": (
                    args.profile_prefill_device_stages
                ),
                "retain_shared_images": args.retain_shared_images,
                "setup_s": setup_s,
                "worker_setup_diagnostics": pool.worker_setup_diagnostics,
                "warmup_repeats": args.warmup_repeats,
                "warmups": warmup_summaries,
                "warmup": warmup_summaries[-1] if warmup_summaries else None,
                "producer_stream_wall_s": stream_wall_s,
                "worker_summary": worker_summary,
            }
        )
        result["producer_wall_s"] = time.perf_counter() - measured_started

        validation_started = time.perf_counter()
        if args.artifact_storage == "persistent":
            crop_rows = read_jsonl(writer.manifest_path)
            if not crop_rows:
                raise RuntimeError("producer emitted no recognition crops")
            sample_indices = sorted({0, len(crop_rows) // 2, len(crop_rows) - 1})
            validated = []
            for index in sample_indices:
                row = crop_rows[index]
                array = read_crop_array(output_dir, row, verify_crc=True)
                validated.append(
                    {
                        "index": index,
                        "request_id": row["request_id"],
                        "shape": list(array.shape),
                        "nbytes": int(array.nbytes),
                    }
                )
                del array
            result["validation"] = {
                "mode": "sample_crc",
                "manifest_crop_count": len(crop_rows),
                "sample_crc_count": len(validated),
                "samples": validated,
                "wall_s": time.perf_counter() - validation_started,
                "passed": len(crop_rows) == writer.crop_count,
            }
        else:
            result["validation"] = {
                "mode": "all_descriptors",
                "descriptor_crop_count": writer.crop_count,
                "wall_s": time.perf_counter() - validation_started,
                "passed": writer.crop_count > 0,
            }
        if not result["validation"]["passed"]:
            raise RuntimeError("prefill output validation failed")
        artifact = result["artifact"]
        result["throughput"] = {
            "pages_per_s": writer.page_count / result["producer_wall_s"],
            "crops_per_s": writer.crop_count / result["producer_wall_s"],
            "real_source_tokens_per_s": (
                writer.real_source_tokens / result["producer_wall_s"]
            ),
            "cross_kv_gib": artifact["cross_kv_payload_bytes"] / 1024**3,
        }
    except BaseException:
        writer.abort()
        raise
    finally:
        shutdown_started = time.perf_counter()
        pool.close()
        shutdown_s = time.perf_counter() - shutdown_started

    result["shutdown_s"] = shutdown_s
    result["total_wall_s"] = time.perf_counter() - total_started
    result["max_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    _atomic_write_json(writer.summary_path, result)
    print("UNIREC_PREFILL_EXPORT_END " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
