#!/usr/bin/env python3
"""Run the full UniRec producer and export CPU cross-KV without decoding."""

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
    parser.add_argument("--offset", type=int, default=769)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup-pages", type=int, default=2)
    parser.add_argument("--layout-threshold", type=float, default=0.4)
    parser.add_argument(
        "--layout-execution",
        choices=("eager", "compiled"),
        default="eager",
    )
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
    parser.add_argument("--use-chart-recognition", action="store_true")
    parser.add_argument("--worker-empty-cache-after-page", action="store_true")
    args = parser.parse_args()
    if args.offset < 0 or args.limit < 1:
        parser.error("--offset must be non-negative and --limit positive")
    if args.workers < 1 or args.warmup_pages < 0:
        parser.error("--workers must be positive and --warmup-pages non-negative")
    if args.cross_cache_length < 1:
        parser.error("--cross-cache-length must be positive")
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

    openocr_root = args.openocr_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    layout_model = args.layout_model.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    layout_cache_dir = args.layout_cache_dir.expanduser().resolve()
    recognition_cache_dir = args.recognition_cache_dir.expanduser().resolve()

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
        openocr_root=openocr_root,
        prepare_pages=True,
        use_chart_recognition=args.use_chart_recognition,
        prefill_recognition=True,
        recognition_model_path=model_path,
        recognition_dtype=args.dtype,
        recognition_cache_dir=recognition_cache_dir,
        empty_cache_after_page=args.worker_empty_cache_after_page,
    )
    setup_s = pool.setup_wall_s
    warmup_summary = None
    writer = CrossKvArtifactWriter(output_dir)
    try:
        if args.warmup_pages:
            warmup_payloads, warmup_summary = pool.map(
                image_paths[: args.warmup_pages],
                label="prefill_export_warmup",
            )
            for payload in warmup_payloads:
                _discard_payload(payload)
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
                "offset": args.offset,
                "limit": args.limit,
                "workers": args.workers,
                "physical_devices": physical_devices,
                "dtype": args.dtype,
                "cross_cache_length": args.cross_cache_length,
                "layout_execution": args.layout_execution,
                "setup_s": setup_s,
                "warmup": warmup_summary,
                "producer_stream_wall_s": stream_wall_s,
                "worker_summary": worker_summary,
            }
        )
        result["producer_wall_s"] = time.perf_counter() - measured_started

        crop_rows = read_jsonl(writer.manifest_path)
        if not crop_rows:
            raise RuntimeError("producer emitted no recognition crops")
        sample_indices = sorted({0, len(crop_rows) // 2, len(crop_rows) - 1})
        validation_started = time.perf_counter()
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
            "manifest_crop_count": len(crop_rows),
            "sample_crc_count": len(validated),
            "samples": validated,
            "wall_s": time.perf_counter() - validation_started,
            "passed": len(crop_rows) == writer.crop_count,
        }
        if not result["validation"]["passed"]:
            raise RuntimeError("artifact manifest count validation failed")
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
