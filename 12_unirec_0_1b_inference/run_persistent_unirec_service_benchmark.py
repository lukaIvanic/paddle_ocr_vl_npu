#!/usr/bin/env python3
"""Warm once, then submit OmniDocBench pages to a persistent UniRec service."""

from __future__ import annotations

import os
from pathlib import Path
import sys


MAIN_PROCESS_MALLOC_CONF = (
    "narenas:2,background_thread:true,"
    "dirty_decay_ms:1000,muzzy_decay_ms:1000"
)
if __name__ == "__main__" and os.environ.get(
    "UNIREC_MAIN_MALLOC_CONF_APPLIED"
) != MAIN_PROCESS_MALLOC_CONF:
    environment = dict(os.environ)
    environment["MALLOC_CONF"] = MAIN_PROCESS_MALLOC_CONF
    environment["UNIREC_MAIN_MALLOC_CONF_APPLIED"] = MAIN_PROCESS_MALLOC_CONF
    os.execvpe(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )

import argparse
from contextlib import redirect_stdout
import json
import time
from typing import Any

os.environ.setdefault("TE_PARALLEL_COMPILER", "1")
os.environ.setdefault("CANN_KNOWLEDGE_BANK_PROCESS_NUM", "0")
os.environ.setdefault("UNIREC_DEINIT_TBE_AFTER_WARMUP", "1")
os.environ.setdefault("UNIREC_PURGE_HOST_AFTER_WARMUP", "1")
os.environ.setdefault("UNIREC_CROSS_KV_D2H_MODE", "packed_cohort")

from low_memory_frontend_pool import (
    CpuCropPreparePool,
    PersistentSharedLayoutProcess,
)
from persistent_unirec_frontend import PersistentUniRecFrontend
from persistent_unirec_service import PersistentUniRecService


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--layout-model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spool-dir", type=Path, required=True)
    parser.add_argument("--layout-cache", type=Path, required=True)
    parser.add_argument("--vision-cache", type=Path, required=True)
    parser.add_argument("--decode-cache-parent", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup-pages", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--recognition-threads", type=int, default=8)
    parser.add_argument("--recognition-resize-chunk-size", type=int, default=0)
    parser.add_argument("--layout-lanes", type=int, default=1)
    parser.add_argument("--layout-batch-size", type=int, default=2)
    parser.add_argument("--layout-threshold", type=float, default=0.5)
    parser.add_argument("--vision-bucket-preset", default="310p_k20_l4")
    parser.add_argument("--vision-lanes", type=int, default=4)
    parser.add_argument("--vision-same-key-shards", type=int, default=1)
    parser.add_argument("--vision-sharded-key-count", type=int, default=4)
    parser.add_argument("--vision-record-budget", type=int, default=128)
    parser.add_argument("--vision-max-calls-per-key", type=int, default=64)
    parser.add_argument("--vision-queue-size", type=int, default=128)
    parser.add_argument("--decode-batch-size", type=int, default=128)
    parser.add_argument("--cross-cache-length", type=int, default=1320)
    parser.add_argument("--self-cache-length", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--ready-queue-size", type=int, default=128)
    parser.add_argument("--vision-tall-fallback", choices=("compiled", "eager"), default="compiled")
    parser.add_argument("--progress-every", type=int, default=32)
    parser.add_argument("--write-outputs", action="store_true")
    return parser.parse_args()


def image_paths(root: Path, *, offset: int, limit: int | None) -> list[Path]:
    paths = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )[offset:]
    return paths if limit is None else paths[:limit]


def main() -> None:
    args = parse_args()
    visible = {
        int(value)
        for value in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    }
    if visible.intersection({5, 6}):
        raise RuntimeError("physical NPU 5 and NPU 6 are excluded")
    paths = image_paths(args.input.resolve(), offset=args.offset, limit=args.limit)
    if not paths:
        raise ValueError("no measured input pages")
    warmup_paths = paths[: min(args.warmup_pages, len(paths))]
    if args.spool_dir.exists() and any(args.spool_dir.iterdir()):
        raise RuntimeError(f"spool directory is not empty: {args.spool_dir}")
    args.spool_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    setup_started = time.perf_counter()
    crop_pool = CpuCropPreparePool(
        workers=args.workers,
        recognition_threads=args.recognition_threads,
        resize_chunk_size=args.recognition_resize_chunk_size,
        openocr_root=args.openocr_root.resolve(),
        spool_root=args.spool_dir.resolve(),
        cross_cache_length=args.cross_cache_length,
        spool_mode="per_page",
    )
    layout = PersistentSharedLayoutProcess(
        model_path=args.layout_model.resolve(),
        cache_dir=args.layout_cache.resolve(),
        device=args.device,
        lanes=args.layout_lanes,
        batch_size=args.layout_batch_size,
        threshold=args.layout_threshold,
    )

    os.environ["UNIREC_STATIC_CACHE_LEN"] = str(args.self_cache_length)
    os.environ["UNIREC_STATIC_CROSS_CACHE_LEN"] = str(args.cross_cache_length)
    os.environ["UNIREC_VISION_BUCKET_PRESET"] = args.vision_bucket_preset
    os.environ["UNIREC_RECOGNITION_INPUT_CONTRACT"] = "compact_uint8_hwc"
    import torch
    import torch_npu

    from bounded_vision_owner import BoundedVisionOwner
    from continuous_unirec import (
        ContinuousUniRecDecoder,
        production_decode_cache_parent,
    )
    from decode_model_optimizations import (
        apply_decode_model_optimizations,
        decode_cache_variant_root,
    )
    from host_memory_diagnostics import process_snapshot
    from modeling_optimized_unirec import OptimizedUniRecRunner
    from persistent_unirec_npu import PersistentUniRecNpuPipeline
    from post_warmup_host_cleanup import cleanup_after_warmup
    from run_opendoc_batched_unirec import assemble_page
    from tbe_compiler_lifecycle import deinitialize_after_warmup
    from vision_bucket_presets import resolve_vision_bucket_specs
    from vision_full_batch import BucketedFullVisionRuntime, VisionBucketSpec

    torch_npu.npu.set_compile_mode(jit_compile=False)
    sys.path.insert(0, str(args.openocr_root.resolve()))
    from tools import infer_doc_onnx

    pipeline = infer_doc_onnx.OpenDocONNX.__new__(infer_doc_onnx.OpenDocONNX)
    pipeline.use_layout_detection = False
    pipeline.use_chart_recognition = True
    pipeline.markdown_ignore_labels = [
        "number",
        "footnote",
        "header",
        "footer",
        "aside_text",
        "footer_image",
        "header_image",
        "chart",
    ]

    runner = OptimizedUniRecRunner(
        model_path=args.model_path.resolve(),
        device=args.device,
        dtype="float16",
        compile_cache_dir=args.vision_cache.resolve(),
    )
    decode_optimizations = apply_decode_model_optimizations(
        runner,
        weight_format="nz",
        lm_head_rows=57344,
    )
    processor_shape = tuple(int(value) for value in runner.processor.max_side)
    runner._static_cross_cache_len_by_processor_max_side[processor_shape] = (
        args.cross_cache_length
    )
    vision_runtime = BucketedFullVisionRuntime(
        runner,
        specs=resolve_vision_bucket_specs(args.vision_bucket_preset),
        focal_depthwise_rewrite="constant_grouped_all",
        weight_format="torchair_internal",
        preset_name=args.vision_bucket_preset,
        synchronize_first_call=False,
    )
    tall_fallback_runtime = None
    if args.vision_tall_fallback == "compiled":
        runner.compile_cache_dir = args.decode_cache_parent.resolve()
        tall_fallback_runtime = BucketedFullVisionRuntime(
            runner,
            specs=(VisionBucketSpec(960, 1408, 1),),
            focal_depthwise_rewrite="constant_grouped_all",
            weight_format="torchair_internal",
            preset_name="compiled_tall_fallback",
            synchronize_first_call=False,
            preapplied_focal_depthwise_rewrite_summary=(
                vision_runtime.focal_depthwise_rewrite_summary
            ),
            preapplied_weight_format_summary=vision_runtime.weight_format_summary,
        )
    runner.compile_cache_dir = args.vision_cache.resolve()
    vision_owner = BoundedVisionOwner(
        vision_runtime,
        lanes=args.vision_lanes,
        same_key_shards=args.vision_same_key_shards,
        sharded_key_count=args.vision_sharded_key_count,
        fallback_runtime=tall_fallback_runtime,
        deinitialize_tbe_after_first_group=False,
    )
    text_runtime = runner._get_compiled_packed_text_prefill_runtime()
    runner.compile_cache_dir = decode_cache_variant_root(
        production_decode_cache_parent(args.decode_cache_parent.resolve()),
        weight_format="nz",
        lm_head_rows=57344,
    )
    decoder = ContinuousUniRecDecoder(
        runner=runner,
        batch_size=args.decode_batch_size,
        max_length=args.max_length,
        decode_mode="compiled_ifa",
        compile_backend="torchair",
        admission_prefetch_depth=0,
        self_cache_length=args.self_cache_length,
        cross_cache_length=args.cross_cache_length,
    )

    holder: dict[str, Any] = {}
    phase = {"name": "warmup", "completed": 0, "total": len(warmup_paths)}

    def response_builder(page: Any) -> dict[str, Any]:
        result = assemble_page(
            page=page,
            pipeline=pipeline,
            infer_doc_onnx=infer_doc_onnx,
        )
        trace = [
            {
                "request_id": crop.request_id,
                "page_index": int(crop.page_index),
                "crop_index": int(crop.crop_index),
                "label": crop.label,
                "text": crop.result["text"],
                "generated_ids": crop.result["generated_ids"],
                "generated_token_count": crop.result["generated_token_count"],
                "prep": crop.result["prep"],
            }
            for crop in page.crops
        ]
        return {"page_index": int(page.page_index), "result": result, "trace": trace}

    def page_complete(request_id: str, response: dict[str, Any]) -> None:
        holder["service"].complete(request_id, response)
        phase["completed"] += 1
        if (
            phase["completed"] % args.progress_every == 0
            or phase["completed"] == phase["total"]
        ):
            print(
                "UNIREC_SERVING_PROGRESS "
                f"phase={phase['name']} pages={phase['completed']}/{phase['total']}",
                flush=True,
            )

    def npu_error(exception: BaseException) -> None:
        service = holder.get("service")
        if service is not None:
            service.fail(exception)

    npu_pipeline = PersistentUniRecNpuPipeline(
        runner=runner,
        vision_owner=vision_owner,
        decoder=decoder,
        text_runtime=text_runtime,
        device=args.device,
        on_page_complete=page_complete,
        response_builder=response_builder,
        on_error=npu_error,
        vision_record_budget=args.vision_record_budget,
        vision_max_calls_per_key=args.vision_max_calls_per_key,
        vision_queue_size=args.vision_queue_size,
        ready_queue_size=args.ready_queue_size,
    )
    frontend = PersistentUniRecFrontend(
        layout=layout,
        crop_pool=crop_pool,
        on_page_ready=npu_pipeline.submit,
    )
    service = PersistentUniRecService(
        frontend=frontend,
        npu_pipeline=npu_pipeline,
    )
    holder["service"] = service
    setup_wall_s = time.perf_counter() - setup_started
    print(
        "UNIREC_SERVING_READY "
        f"setup_wall_s={setup_wall_s:.3f} warmup_pages={len(warmup_paths)}",
        flush=True,
    )

    warmup_started = time.perf_counter()
    warmup_futures = service.submit_many(
        warmup_paths,
        request_prefix="warmup",
    )
    service.wait_futures(warmup_futures)
    service.wait_idle()
    warmup_wall_s = time.perf_counter() - warmup_started
    del warmup_futures
    compiler_cleanup = deinitialize_after_warmup(
        "persistent_service_real_request_warmup_complete"
    )
    host_cleanup = cleanup_after_warmup("persistent_service_hot")
    prior_metrics = service.reset_measurement()
    spool_leftovers = [
        path
        for path in args.spool_dir.iterdir()
        if path.is_file() and path.stat().st_size > 0
    ]
    if spool_leftovers:
        raise RuntimeError(
            "warmup left application pixel spools behind: "
            f"{sorted(spool_leftovers)[:4]}"
        )
    print(
        "UNIREC_SERVING_WARMUP_END "
        f"pages={len(warmup_paths)} wall_s={warmup_wall_s:.3f} "
        f"compiler_cleanup={json.dumps(compiler_cleanup, ensure_ascii=False)} "
        f"host_cleanup={json.dumps(host_cleanup, ensure_ascii=False)}",
        flush=True,
    )

    phase.update(name="measured", completed=0, total=len(paths))
    hot_started = time.perf_counter()
    measured_futures = service.submit_many(paths, request_prefix="measured")
    responses = service.wait_futures(measured_futures)
    service.wait_idle()
    hot_wall_s = time.perf_counter() - hot_started
    measurement = service.measurement()
    pages_per_s = len(paths) / hot_wall_s
    print(
        "UNIREC_SERVING_HOT_END "
        f"pages={len(paths)} wall_s={hot_wall_s:.3f} pages_per_s={pages_per_s:.6f}",
        flush=True,
    )

    shutdown_started = time.perf_counter()
    decode_summary = service.close()
    shutdown_wall_s = time.perf_counter() - shutdown_started

    write_started = time.perf_counter()
    trace_path = args.output_dir / "recognition_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as trace_file:
        for local_page_index, response in enumerate(responses):
            measured_page_index = args.offset + local_page_index
            for row in response["trace"]:
                normalized = dict(row)
                normalized["page_index"] = measured_page_index
                normalized["request_id"] = (
                    f"page_{measured_page_index:06d}_crop_"
                    f"{int(row['crop_index']):04d}"
                )
                trace_file.write(
                    json.dumps(normalized, ensure_ascii=False) + "\n"
                )
    if args.write_outputs:
        import cv2
        import numpy as np

        with Path(os.devnull).open("w", encoding="utf-8") as devnull:
            for response in responses:
                result = response["result"]
                if any(
                    bool(row.get("is_image", False))
                    for row in result["recognition_results"]
                ):
                    page_image = cv2.imread(result["input_path"], cv2.IMREAD_COLOR)
                    if page_image is None:
                        raise RuntimeError(
                            f"failed to reload page {result['input_path']}"
                        )
                    result["_page_image"] = page_image
                else:
                    result["_page_image"] = np.empty((0, 0, 3), dtype=np.uint8)
                with redirect_stdout(devnull):
                    pipeline.save_to_json(result, str(args.output_dir))
                    pipeline.save_to_markdown(result, str(args.output_dir))
                result.pop("_page_image", None)
    write_wall_s = time.perf_counter() - write_started

    summary = {
        "schema": "unirec_persistent_service_benchmark_v2",
        "status": "pass",
        "chip": torch_npu.npu.get_device_name(0),
        "page_count": len(paths),
        "warmup_page_count": len(warmup_paths),
        "setup_wall_s_excluded": setup_wall_s,
        "warmup_wall_s_excluded": warmup_wall_s,
        "hot_pipeline_wall_s": hot_wall_s,
        "hot_pages_per_s": pages_per_s,
        "shutdown_wall_s_excluded": shutdown_wall_s,
        "write_wall_s_excluded": write_wall_s,
        "measurement": measurement,
        "warmup_metrics": prior_metrics,
        "decode": decode_summary,
        "decode_optimizations": decode_optimizations,
        "final_memory": process_snapshot(),
        "settings": {
            "workers": args.workers,
            "recognition_threads": args.recognition_threads,
            "layout_lanes": args.layout_lanes,
            "layout_batch_size": args.layout_batch_size,
            "vision_bucket_preset": args.vision_bucket_preset,
            "vision_lanes": args.vision_lanes,
            "vision_record_budget": args.vision_record_budget,
            "vision_max_calls_per_key": args.vision_max_calls_per_key,
            "vision_queue_size": args.vision_queue_size,
            "decode_batch_size": args.decode_batch_size,
            "cross_cache_length": args.cross_cache_length,
            "self_cache_length": args.self_cache_length,
        },
        "artifacts": {
            "output_dir": str(args.output_dir.resolve()),
            "recognition_trace": str(trace_path.resolve()),
        },
    }
    summary_path = args.output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("UNIREC_SERVING_SUMMARY " + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
