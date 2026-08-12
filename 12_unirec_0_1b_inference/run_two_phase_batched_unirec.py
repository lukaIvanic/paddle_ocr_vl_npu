#!/usr/bin/env python3
"""Run UniRec page prefill to CPU RAM, then decode the retained bank.

This is an isolation benchmark, not the streaming production path.  Persistent
layout/recognition workers finish every selected page and leave their page
arenas resident in CPU shared memory.  The workers are then closed before the
coordinator model is created, warmed, and used for fixed-batch continuous decode.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import SimpleQueue
from threading import Event as ThreadEvent
from threading import Thread
from types import SimpleNamespace
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--layout-model",
        type=Path,
        default=Path("/workspace/models/PP-DocLayoutV2_safetensors"),
    )
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
            "all_conv_fz",
            "linear_nz",
            "torchair_internal",
        ),
        default="native",
    )
    parser.add_argument(
        "--layout-depthwise-rewrite",
        choices=("native", "group16", "group32", "dense"),
        default="native",
    )
    parser.add_argument(
        "--layout-preformat-frozen-bn-buffers",
        action="store_true",
    )
    parser.add_argument("--layout-threshold", type=float, default=0.4)
    parser.add_argument(
        "--layout-cache-dir",
        type=Path,
        default=Path(
            ".runtime_cache/12_unirec_0_1b_inference/layout_process_pool"
        ),
    )
    parser.add_argument("--stock-encoder", type=Path, required=True)
    parser.add_argument("--stock-decoder", type=Path, required=True)
    parser.add_argument("--stock-tokenizer-mapping", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("float16",), default="float16")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup-pages", type=int, default=8)
    parser.add_argument("--layout-batch-size", type=int, default=1)
    parser.add_argument("--vision-page-lookahead", type=int, default=4)
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
    )
    parser.add_argument(
        "--vision-weight-format",
        choices=("native", "focal_prepack", "torchair_internal"),
        default="native",
    )
    parser.add_argument(
        "--no-chart-recognition",
        dest="use_chart_recognition",
        action="store_false",
        help="Exclude chart blocks from UniRec recognition prefill",
    )
    parser.set_defaults(use_chart_recognition=True)
    parser.add_argument("--recognition-preprocess-threads", type=int, default=8)
    parser.add_argument(
        "--recognition-input-contract",
        choices=("legacy_float_chw", "compact_uint8_hwc"),
        default="compact_uint8_hwc",
    )
    parser.add_argument("--cross-cache-length", type=int, default=512)
    parser.add_argument("--self-cache-length", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--decode-batch-size", type=int, default=128)
    parser.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=Path(
            ".runtime_cache/12_unirec_0_1b_inference/"
            "opendoc_batched_decode_a372dbf"
        ),
    )
    parser.add_argument("--decode-warmup-passes", type=int, default=2)
    parser.add_argument(
        "--decode-admission-prefetch-depth",
        type=int,
        default=0,
        help=(
            "Opt-in NPU cross-K/V staging-ring depth. Zero keeps direct "
            "pageable-host admission."
        ),
    )
    parser.add_argument("--prefill-device-timing", action="store_true")
    parser.add_argument("--progress-every-pages", type=int, default=0)
    parser.add_argument("--progress-heartbeat-s", type=float, default=0.0)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.warmup_pages < 0:
        parser.error("--warmup-pages must be non-negative")
    if args.layout_batch_size < 1:
        parser.error("--layout-batch-size must be positive")
    if args.layout_batch_size > args.vision_page_lookahead:
        parser.error(
            "--layout-batch-size cannot exceed --vision-page-lookahead"
        )
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.decode_batch_size < 1:
        parser.error("--decode-batch-size must be positive")
    if args.max_length > args.self_cache_length:
        parser.error("--max-length cannot exceed --self-cache-length")
    if args.decode_admission_prefetch_depth < 0:
        parser.error("--decode-admission-prefetch-depth must be non-negative")
    if args.progress_every_pages < 0 or args.progress_heartbeat_s < 0:
        parser.error("progress intervals must be non-negative")
    return args


def physical_devices() -> list[int]:
    value = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not value:
        raise RuntimeError(
            "ASCEND_RT_VISIBLE_DEVICES is unset; source npu-setup before launch"
        )
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def release_unopened_payload(payload: dict[str, Any]) -> None:
    """Unlink a retained worker arena that has not been materialized."""
    from layout_process_pool import SharedPageLease

    shared = payload.get("shared_memory")
    if not isinstance(shared, dict):
        return
    try:
        lease = SharedPageLease(str(shared["name"]))
    except FileNotFoundError:
        return
    lease.close()


def payload_totals(payloads: list[dict[str, Any]]) -> dict[str, int]:
    crop_count = 0
    cross_kv_bytes = 0
    rejected_crop_count = 0
    real_source_tokens = 0
    physical_source_tokens = 0
    for payload in payloads:
        rejected_crop_count += int(
            payload.get("cross_capacity_rejected_crops", 0)
        )
        for crop in payload["crops"]:
            descriptor = crop.get("worker_cross_kv_descriptor")
            metadata = crop.get("worker_prefill_metadata")
            if not isinstance(descriptor, dict) or not isinstance(metadata, dict):
                raise RuntimeError("retained crop has no worker-prefill payload")
            crop_count += 1
            cross_kv_bytes += int(descriptor["nbytes"])
            real_source_tokens += int(
                metadata["text_prefill_real_source_tokens"]
            )
            physical_source_tokens += int(
                metadata["text_prefill_physical_source_tokens"]
            )
    return {
        "crop_count": crop_count,
        "cross_kv_bytes": cross_kv_bytes,
        "rejected_crop_count": rejected_crop_count,
        "real_source_tokens": real_source_tokens,
        "physical_source_tokens": physical_source_tokens,
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def main() -> None:
    args = parse_args()
    devices = physical_devices()
    if 5 in devices or 6 in devices:
        raise RuntimeError(
            "physical NPU 5 and NPU 6 are excluded from UniRec experiments"
        )
    os.environ["UNIREC_STATIC_CACHE_LEN"] = str(args.self_cache_length)
    os.environ["UNIREC_STATIC_CROSS_CACHE_LEN"] = str(args.cross_cache_length)
    os.environ["UNIREC_RECOGNITION_PREPROCESS_THREADS"] = str(
        args.recognition_preprocess_threads
    )
    os.environ["UNIREC_RECOGNITION_INPUT_CONTRACT"] = (
        args.recognition_input_contract
    )

    openocr_root = args.openocr_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    layout_model = args.layout_model.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(openocr_root))
    from tools.utils.utility import get_image_file_list

    image_paths = [
        Path(path).resolve()
        for path in sorted(get_image_file_list(str(input_path)))
    ][args.offset : args.offset + args.limit]
    if len(image_paths) != args.limit:
        raise RuntimeError(
            f"requested {args.limit} pages at offset {args.offset}, "
            f"found {len(image_paths)}"
        )

    from layout_process_pool import DynamicLayoutProcessPool

    lifecycle_started = time.perf_counter()
    retained_payloads: list[dict[str, Any]] = []
    warmup_summary: dict[str, Any] | None = None
    prefill_summary: dict[str, Any] | None = None
    print(
        "UNIREC_TWO_PHASE_PREFILL_SETUP_BEGIN "
        f"pages={len(image_paths)} workers={args.workers}",
        flush=True,
    )
    pool = DynamicLayoutProcessPool(
        worker_count=args.workers,
        model_path=layout_model,
        cache_dir=args.layout_cache_dir.expanduser().resolve(),
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
        recognition_cache_dir=args.compile_cache_dir.expanduser().resolve(),
        recognition_full_vision_buckets=True,
        recognition_vision_focal_depthwise_rewrite=(
            args.vision_focal_depthwise_rewrite
        ),
        recognition_vision_weight_format=args.vision_weight_format,
        recognition_page_lookahead=args.vision_page_lookahead,
        profile_prefill_device_stages=args.prefill_device_timing,
        retain_shared_images=False,
        progress_every_pages=args.progress_every_pages,
        progress_heartbeat_s=args.progress_heartbeat_s,
    )
    prefill_worker_setup_s = pool.setup_wall_s
    print(
        "UNIREC_TWO_PHASE_PREFILL_SETUP_END "
        f"wall_s={prefill_worker_setup_s:.3f}",
        flush=True,
    )
    warmup_wall_s = 0.0
    prefill_phase_wall_s = 0.0
    prefill_worker_shutdown_s = 0.0
    try:
        if args.warmup_pages:
            warmup_started = time.perf_counter()
            warmup_payloads, warmup_summary = pool.map(
                image_paths[: args.warmup_pages],
                label="two_phase_warmup",
            )
            for payload in warmup_payloads:
                release_unopened_payload(payload)
            warmup_wall_s = time.perf_counter() - warmup_started

        print(
            "UNIREC_TWO_PHASE_PREFILL_BEGIN "
            f"pages={len(image_paths)} workers={args.workers}",
            flush=True,
        )
        prefill_started = time.perf_counter()
        retained_payloads, prefill_summary = pool.map(
            image_paths,
            label="two_phase_measured_prefill",
        )
        prefill_phase_wall_s = time.perf_counter() - prefill_started
        retained = payload_totals(retained_payloads)
        print(
            "UNIREC_TWO_PHASE_PREFILL_END "
            + json.dumps(
                {
                    "wall_s": prefill_phase_wall_s,
                    "pages_per_s": len(image_paths) / prefill_phase_wall_s,
                    "retained_shared_bytes": prefill_summary[
                        "shared_payload_bytes"
                    ],
                    **retained,
                }
            ),
            flush=True,
        )
    except BaseException:
        for payload in retained_payloads:
            release_unopened_payload(payload)
        raise
    finally:
        shutdown_started = time.perf_counter()
        pool.close()
        prefill_worker_shutdown_s = time.perf_counter() - shutdown_started

    # Import the coordinator model only after every prefill worker is gone.
    print(
        "UNIREC_TWO_PHASE_DECODE_SETUP_BEGIN "
        f"batch={args.decode_batch_size} self_kv={args.self_cache_length} "
        f"cross_kv={args.cross_cache_length}",
        flush=True,
    )
    decode_setup_started = time.perf_counter()
    import torch_npu

    torch_npu.npu.set_compile_mode(jit_compile=False)
    from tools import infer_doc_onnx

    import run_opendoc_batched_unirec as base
    from continuous_unirec import (
        ContinuousCompletedItem,
        ContinuousReadyItem,
        ContinuousUniRecDecoder,
    )
    from modeling_optimized_unirec import OptimizedUniRecRunner

    pipeline = infer_doc_onnx.OpenDocONNX(
        layout_model_path=str(args.layout_model.expanduser().resolve()),
        unirec_encoder_path=str(args.stock_encoder.expanduser().resolve()),
        unirec_decoder_path=str(args.stock_decoder.expanduser().resolve()),
        tokenizer_mapping_path=str(
            args.stock_tokenizer_mapping.expanduser().resolve()
        ),
        use_gpu=False,
        layout_threshold=args.layout_threshold,
        use_layout_detection=False,
        auto_download=False,
        max_parallel_blocks=1,
    )
    runner = OptimizedUniRecRunner(
        model_path=model_path,
        device=args.device,
        dtype=args.dtype,
        compile_cache_dir=args.compile_cache_dir.expanduser().resolve(),
    )
    processor_shape = tuple(int(value) for value in runner.processor.max_side)
    runner._static_cross_cache_len_by_processor_max_side[processor_shape] = (
        args.cross_cache_length
    )
    decode_graph_warmup = base.warmup_configured_graphs(
        args=SimpleNamespace(
            text_prefill_mode="eager",
            decode_mode="compiled_ifa",
            compile_backend="torchair",
            decode_batch_size=args.decode_batch_size,
        ),
        runner=runner,
        vision_atlas_runtime=None,
        passes=args.decode_warmup_passes,
    )
    decode_setup_s = time.perf_counter() - decode_setup_started
    print(
        "UNIREC_TWO_PHASE_DECODE_SETUP_END "
        f"wall_s={decode_setup_s:.3f} batch={args.decode_batch_size}",
        flush=True,
    )

    metrics = base.RunMetrics()
    pending_pages: deque[base.PageRequest] = deque()
    pending_writes: deque[
        tuple[base.PageRequest, Future[tuple[float, float, float, bool]]]
    ] = deque()
    write_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="unirec-two-phase-writer",
    )
    written_pages = 0
    output_image_reload_s = 0.0
    output_image_reload_pages = 0
    max_pending_writes = 8
    completion_queue: SimpleQueue[
        tuple[str, base.PageRequest | ContinuousCompletedItem] | None
    ] = SimpleQueue()
    collector_errors: list[BaseException] = []
    collector_completion_processing_s = 0.0
    collector_join_s = 0.0
    collector_stopped = False
    decode_progress_stop = ThreadEvent()
    decode_completed_crops = 0
    decode_last_completion_at = time.perf_counter()

    def write_page(result: dict[str, Any]) -> tuple[float, float, float, bool]:
        image_reload_s = 0.0
        reloaded_image = False
        if any(
            bool(record.get("is_image", False))
            for record in result["recognition_results"]
        ):
            import cv2

            reload_started = time.perf_counter()
            page_image = cv2.imread(result["input_path"], cv2.IMREAD_COLOR)
            image_reload_s = time.perf_counter() - reload_started
            if page_image is None:
                raise RuntimeError(
                    f"failed to reload output page: {result['input_path']}"
                )
            result["_page_image"] = page_image
            reloaded_image = True
        else:
            # OpenDoc's writer otherwise reloads the input unconditionally.
            # A zero-sized sentinel is sufficient when no image region is saved.
            result["_page_image"] = np.empty((0, 0, 3), dtype=np.uint8)
        started = time.perf_counter()
        pipeline.save_to_json(result, str(output_dir))
        pipeline.save_to_markdown(result, str(output_dir))
        completed_at = time.perf_counter()
        return (
            completed_at - started,
            completed_at,
            image_reload_s,
            reloaded_image,
        )

    def record_completed_write(
        page: base.PageRequest,
        future: Future[tuple[float, float, float, bool]],
    ) -> None:
        nonlocal written_pages, output_image_reload_s, output_image_reload_pages
        write_s, completed_at, image_reload_s, reloaded_image = future.result()
        metrics.output_write_s += write_s
        output_image_reload_s += image_reload_s
        output_image_reload_pages += int(reloaded_image)
        written_pages += 1
        print(
            "UNIREC_TWO_PHASE_DECODE_PAGE "
            f"pages={written_pages}/{len(image_paths)} "
            f"page_index={page.page_index} crops={len(page.crops)} "
            f"elapsed_s={completed_at - decode_phase_started:.3f}",
            flush=True,
        )
        metrics.page_records.append(
            {
                "page_index": page.page_index,
                "image": str(page.image_path),
                "crop_count": len(page.crops),
                "layout_s": page.layout_s,
                "decode_phase_completion_s": completed_at - decode_phase_started,
            }
        )
        base.release_page_frontend_storage(page)

    def drain_completed_writes(*, wait: bool, count: int | None = None) -> None:
        drained = 0
        while pending_writes and (count is None or drained < count):
            page, future = pending_writes[0]
            if not wait and not future.done():
                break
            pending_writes.popleft()
            record_completed_write(page, future)
            drained += 1

    def submit_page_write(page: base.PageRequest, result: dict[str, Any]) -> None:
        drain_completed_writes(wait=False)
        if len(pending_writes) >= max_pending_writes:
            wait_started = time.perf_counter()
            drain_completed_writes(wait=True, count=1)
            metrics.output_write_backpressure_s += time.perf_counter() - wait_started
        pending_writes.append((page, write_executor.submit(write_page, result)))
        metrics.output_write_max_pending = max(
            metrics.output_write_max_pending,
            len(pending_writes),
        )

    def flush_ready_pages() -> None:
        drain_completed_writes(wait=False)
        while pending_pages and pending_pages[0].is_ready():
            page = pending_pages.popleft()
            assembly_started = time.perf_counter()
            result = base.assemble_page(
                page=page,
                pipeline=pipeline,
                infer_doc_onnx=infer_doc_onnx,
            )
            metrics.output_assembly_s += time.perf_counter() - assembly_started
            submit_page_write(page, result)

    def ready_source():
        for payload in retained_payloads:
            page = base.page_request_from_process_payload(
                payload,
                measured_layout_s=float(payload["frontend_timing_s"]["layout_s"]),
            )
            metrics.layout_s += page.layout_s
            metrics.page_prepare_total_s += page.prepare_page_total_s
            base.accumulate_stage_seconds(
                metrics.frontend_timing_s,
                page.frontend_timing_s,
            )
            completion_queue.put(("page", page))
            for crop in page.crops:
                item = base.build_worker_prefilled_item(crop)
                base.record_prefill_metrics(metrics, item)
                yield ContinuousReadyItem(
                    request_id=crop.request_id,
                    payload=crop,
                    prefilled=item,
                )

    def process_completed_crop(completed_item: ContinuousCompletedItem) -> None:
        crop = completed_item.payload
        if not isinstance(crop, base.CropRequest):
            raise TypeError(f"unexpected crop payload: {type(crop)!r}")
        result = completed_item.result
        crop.result = result
        metrics.crop_records.append(
            {
                "request_id": crop.request_id,
                "page": crop.page_name,
                "page_index": crop.page_index,
                "crop_index": crop.crop_index,
                "label": crop.label,
                "crop_size": list(crop.image_size),
                "processed_image_size": result["prep"]["processed_image_size"],
                "encoder_seq_len_hint": result["prep"]["encoder_seq_len_hint"],
                "token_ids": result["generated_ids"],
                "text": result["text"],
                "token_count": result["generated_token_count"],
                "decode_token_count": result["decode_generated_token_count"],
                "prefill_s": result["ttft_s"],
                "prefill_device_stage_s": result.get("prefill_device_stage_s"),
                "text_prefill_execution": result.get("text_prefill_execution"),
                "text_prefill_real_source_tokens": result.get(
                    "text_prefill_real_source_tokens"
                ),
                "text_prefill_physical_source_tokens": result.get(
                    "text_prefill_physical_source_tokens"
                ),
                "decode_slot": completed_item.slot,
                "admission_index": completed_item.admission_index,
                "completion_index": completed_item.completion_index,
            }
        )
        flush_ready_pages()

    def collect_completions() -> None:
        nonlocal collector_completion_processing_s
        try:
            while True:
                message = completion_queue.get()
                if message is None:
                    break
                kind, payload = message
                if kind == "page":
                    if not isinstance(payload, base.PageRequest):
                        raise TypeError(
                            f"unexpected page payload: {type(payload)!r}"
                        )
                    pending_pages.append(payload)
                    flush_ready_pages()
                    continue
                if kind != "crop" or not isinstance(
                    payload,
                    ContinuousCompletedItem,
                ):
                    raise TypeError(
                        f"unexpected completion message: {kind!r} "
                        f"{type(payload)!r}"
                    )
                started = time.perf_counter()
                process_completed_crop(payload)
                collector_completion_processing_s += (
                    time.perf_counter() - started
                )
            flush_ready_pages()
            if pending_pages:
                raise RuntimeError(
                    f"unfinished pages after decode: {len(pending_pages)}"
                )
            final_drain_started = time.perf_counter()
            drain_completed_writes(wait=True)
            metrics.output_write_final_drain_s = (
                time.perf_counter() - final_drain_started
            )
        except BaseException as exception:
            collector_errors.append(exception)

    def enqueue_completed_crop(completed_item: ContinuousCompletedItem) -> None:
        nonlocal decode_completed_crops, decode_last_completion_at
        decode_completed_crops += 1
        decode_last_completion_at = time.perf_counter()
        completion_queue.put(("crop", completed_item))

    def report_decode_heartbeat() -> None:
        while not decode_progress_stop.wait(15.0):
            now = time.perf_counter()
            print(
                "UNIREC_TWO_PHASE_DECODE_HEARTBEAT "
                f"completed_crops={decode_completed_crops}/{retained['crop_count']} "
                f"written_pages={written_pages}/{len(image_paths)} "
                f"elapsed_s={now - decode_phase_started:.1f} "
                f"silence_s={now - decode_last_completion_at:.1f}",
                flush=True,
            )

    collector_thread = Thread(
        target=collect_completions,
        name="unirec-two-phase-result-collector",
    )

    def stop_collector() -> None:
        nonlocal collector_join_s, collector_stopped
        if collector_stopped:
            return
        collector_stopped = True
        completion_queue.put(None)
        join_started = time.perf_counter()
        collector_thread.join()
        collector_join_s = time.perf_counter() - join_started

    decode_phase_started = time.perf_counter()
    decode_last_completion_at = decode_phase_started
    decode_progress_thread = Thread(
        target=report_decode_heartbeat,
        name="unirec-two-phase-decode-progress",
        daemon=True,
    )
    collector_thread.start()
    decode_progress_thread.start()
    decode_inference_wall_s = 0.0
    try:
        continuous_decode = ContinuousUniRecDecoder(
            runner=runner,
            batch_size=args.decode_batch_size,
            max_length=args.max_length,
            decode_mode="compiled_ifa",
            compile_backend="torchair",
            admission_prefetch_depth=args.decode_admission_prefetch_depth,
        ).run(ready_source(), on_complete=enqueue_completed_crop)
        decode_inference_wall_s = time.perf_counter() - decode_phase_started
        base.record_direct_arena_admission_metrics(metrics, continuous_decode)
        metrics.decode_s = float(continuous_decode["decode_s"])
        metrics.raw_decode_token_slots = int(
            continuous_decode["raw_decode_token_slots"]
        )
        metrics.effective_decode_tokens = int(
            continuous_decode["effective_decode_tokens"]
        )
        metrics.idle_decode_token_slots = int(
            continuous_decode["idle_decode_token_slots"]
        )
        metrics.padding_decode_token_slots = (
            metrics.raw_decode_token_slots - metrics.effective_decode_tokens
        )
        stop_collector()
        decode_progress_stop.set()
        decode_progress_thread.join(timeout=20.0)
        if collector_errors:
            raise RuntimeError("background result collector failed") from (
                collector_errors[0]
            )
        write_executor.shutdown(wait=True)
        if written_pages != len(image_paths):
            raise RuntimeError(
                f"written page mismatch: {written_pages} != {len(image_paths)}"
            )
    except BaseException:
        decode_progress_stop.set()
        decode_progress_thread.join(timeout=20.0)
        stop_collector()
        write_executor.shutdown(wait=True, cancel_futures=True)
        for payload in retained_payloads:
            release_unopened_payload(payload)
        for page in pending_pages:
            base.release_page_frontend_storage(page)
        for page, _future in pending_writes:
            base.release_page_frontend_storage(page)
        raise
    decode_phase_wall_s = time.perf_counter() - decode_phase_started

    trace_path = output_dir / "recognition_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for record in sorted(
            metrics.crop_records,
            key=lambda item: (item["page_index"], item["crop_index"]),
        ):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    retained = payload_totals(retained_payloads)
    sequential_inference_core_wall_s = (
        prefill_phase_wall_s + decode_inference_wall_s
    )
    sequential_core_wall_s = prefill_phase_wall_s + decode_phase_wall_s
    summary = {
        "status": "ok",
        "execution": "prefill_then_decode_cpu_shared_memory",
        "physical_devices": devices,
        "page_count": len(image_paths),
        "offset": args.offset,
        "workers": args.workers,
        "layout_batch_size": args.layout_batch_size,
        "layout_execution": args.layout_execution,
        "layout_dtype": args.layout_dtype,
        "layout_weight_format": args.layout_weight_format,
        "layout_depthwise_rewrite": args.layout_depthwise_rewrite,
        "layout_preformat_frozen_bn_buffers": (
            args.layout_preformat_frozen_bn_buffers
        ),
        "vision_focal_depthwise_rewrite": (
            args.vision_focal_depthwise_rewrite
        ),
        "vision_weight_format": args.vision_weight_format,
        "recognition_preprocess_threads": args.recognition_preprocess_threads,
        "use_chart_recognition": args.use_chart_recognition,
        "vision_prefill_mode": "compiled_full_buckets",
        "text_prefill_mode": "compiled_packed_s1024",
        "decode_mode": "compiled_ifa",
        "decode_scheduling": "continuous",
        "decode_batch_size": args.decode_batch_size,
        "decode_admission_prefetch_depth": (
            args.decode_admission_prefetch_depth
        ),
        "self_cache_length": args.self_cache_length,
        "cross_cache_length": args.cross_cache_length,
        "retained_bank": {
            **retained,
            "shared_payload_bytes": prefill_summary["shared_payload_bytes"],
            "storage": "page_scoped_posix_shared_memory_cross_kv_only",
            "retained_images": False,
            "disk_bytes": 0,
        },
        "timing_s": {
            "prefill_worker_setup": prefill_worker_setup_s,
            "prefill_warmup": warmup_wall_s,
            "prefill_phase": prefill_phase_wall_s,
            "prefill_worker_shutdown": prefill_worker_shutdown_s,
            "decode_setup_and_graph_warmup": decode_setup_s,
            "decode_inference_including_ingress": decode_inference_wall_s,
            "decode_phase_including_ingress_and_output": decode_phase_wall_s,
            "sequential_inference_core": sequential_inference_core_wall_s,
            "sequential_core_prefill_plus_decode": sequential_core_wall_s,
            "lifecycle": time.perf_counter() - lifecycle_started,
            "decode_graph": metrics.decode_s,
            "output_assembly": metrics.output_assembly_s,
            "output_write": metrics.output_write_s,
            "output_image_reload": output_image_reload_s,
            "output_write_backpressure": metrics.output_write_backpressure_s,
            "output_write_final_drain": metrics.output_write_final_drain_s,
            "completion_collector_processing": (
                collector_completion_processing_s
            ),
            "completion_collector_final_join": collector_join_s,
        },
        "throughput": {
            "prefill_pages_per_s": len(image_paths) / prefill_phase_wall_s,
            "decode_inference_pages_per_s": len(image_paths)
            / decode_inference_wall_s,
            "decode_phase_pages_per_s": len(image_paths) / decode_phase_wall_s,
            "sequential_inference_core_pages_per_s": len(image_paths)
            / sequential_inference_core_wall_s,
            "sequential_core_pages_per_s": len(image_paths)
            / sequential_core_wall_s,
            "decode_effective_tokens_per_s": (
                metrics.effective_decode_tokens / metrics.decode_s
                if metrics.decode_s
                else None
            ),
            "decode_raw_token_slots_per_s": (
                metrics.raw_decode_token_slots / metrics.decode_s
                if metrics.decode_s
                else None
            ),
        },
        "prefill_worker_setup_diagnostics": pool.worker_setup_diagnostics,
        "prefill_warmup_summary": warmup_summary,
        "prefill_phase_summary": prefill_summary,
        "decode_graph_warmup": decode_graph_warmup,
        "decode": continuous_decode,
        "prefill_device_stage_s": metrics.prefill_device_stage_s,
        "text_prefill_real_source_tokens": metrics.text_prefill_real_source_tokens,
        "text_prefill_physical_source_tokens": (
            metrics.text_prefill_physical_source_tokens
        ),
        "crop_count": len(metrics.crop_records),
        "output_image_reload_pages": output_image_reload_pages,
        "output_write_max_pending": metrics.output_write_max_pending,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "trace": str(trace_path),
    }
    atomic_write_json(output_dir / "run_summary.json", summary)
    print(
        "UNIREC_TWO_PHASE_END " + json.dumps(summary, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
