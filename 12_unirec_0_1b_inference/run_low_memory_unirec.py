#!/usr/bin/env python3
"""Accuracy-safe UniRec with W4/T8 CPU work and one bounded NPU owner."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
from queue import Queue
import sys
from threading import Thread
import time
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("TE_PARALLEL_COMPILER", "1")
os.environ.setdefault("CANN_KNOWLEDGE_BANK_PROCESS_NUM", "0")
os.environ.setdefault("UNIREC_DEINIT_TBE_AFTER_WARMUP", "1")
os.environ.setdefault("UNIREC_PURGE_HOST_AFTER_WARMUP", "1")
os.environ.setdefault("UNIREC_CROSS_KV_D2H_MODE", "packed_cohort")

from low_memory_frontend_pool import CpuCropPreparePool, SharedLayoutProcess


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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--recognition-threads", type=int, default=8)
    parser.add_argument("--layout-lanes", type=int, default=1)
    parser.add_argument("--layout-batch-size", type=int, default=2)
    parser.add_argument("--layout-threshold", type=float, default=0.5)
    parser.add_argument("--vision-bucket-preset", default="310p_k20_l4")
    parser.add_argument("--vision-lanes", type=int, default=4)
    parser.add_argument("--vision-same-key-shards", type=int, default=2)
    parser.add_argument("--vision-sharded-key-count", type=int, default=4)
    parser.add_argument(
        "--recognition-schedule",
        choices=("two_phase", "streaming"),
        default="two_phase",
    )
    parser.add_argument("--decode-batch-size", type=int, default=128)
    parser.add_argument("--cross-cache-length", type=int, default=1320)
    parser.add_argument("--self-cache-length", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--progress-every", type=int, default=32)
    parser.add_argument(
        "--defer-output-write",
        action="store_true",
        help=(
            "Write frontend metadata and recognition_trace.jsonl, but defer "
            "Markdown/JSON/image materialization to a fresh CPU process."
        ),
    )
    return parser.parse_args()


def image_paths(root: Path, *, offset: int, limit: int | None) -> list[Path]:
    paths = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )[offset:]
    return paths if limit is None else paths[:limit]


def _payload_to_pages(payloads: list[dict[str, Any]]) -> tuple[list[Any], list[dict[str, Any]]]:
    from run_opendoc_batched_unirec import CropRequest, PageRequest

    pages = []
    records = []
    source_index = 0
    for payload in payloads:
        page_index = int(payload["page_index"])
        path = Path(payload["image_path"])
        crops = []
        for crop_payload in payload["crops"]:
            crop = CropRequest(
                page_index=page_index,
                crop_index=int(crop_payload["crop_index"]),
                page_name=path.name,
                image=None,
                label=str(crop_payload["label"]),
                figure_token_map=dict(crop_payload["figure_token_map"]),
                source_image_size=tuple(
                    int(value) for value in crop_payload["source_image_size"]
                ),
            )
            crops.append(crop)
            records.append(
                {
                    "source_index": source_index,
                    "request_id": crop.request_id,
                    "source_image_size": list(crop.source_image_size),
                    "processed_image_size": list(
                        crop_payload["processed_image_size"]
                    ),
                    "processed_pixel_values_descriptor": dict(
                        crop_payload["processed_pixel_values_descriptor"]
                    ),
                    "crop": crop,
                }
            )
            source_index += 1
        frontend = dict(payload["frontend_timing_s"])
        pages.append(
            PageRequest(
                page_index=page_index,
                image_path=path,
                image=None,
                width=int(payload["width"]),
                height=int(payload["height"]),
                layout_results=payload["layout_results"],
                blocks=payload["blocks"],
                vlm_block_ids=[int(value) for value in payload["vlm_block_ids"]],
                crops=crops,
                drop_figures_set=set(payload["drop_figures_set"]),
                started_at=float(payload["started_at"]),
                layout_s=float(frontend.get("layout_s", 0.0)),
                prepare_page_total_s=sum(
                    float(value)
                    for name, value in frontend.items()
                    if name.endswith("_s")
                ),
                frontend_timing_s=frontend,
            )
        )
    return pages, records


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
        raise ValueError("no input pages")
    if args.spool_dir.exists() and any(args.spool_dir.iterdir()):
        raise RuntimeError(f"spool directory is not empty: {args.spool_dir}")
    args.spool_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    process_started = time.perf_counter()

    print(
        "UNIREC_LOWMEM_FRONTEND_BEGIN "
        f"pages={len(paths)} workers={args.workers} "
        f"threads={args.recognition_threads} layout_lanes={args.layout_lanes}",
        flush=True,
    )
    frontend_started = time.perf_counter()
    crop_pool = CpuCropPreparePool(
        workers=args.workers,
        recognition_threads=args.recognition_threads,
        openocr_root=args.openocr_root.resolve(),
        spool_root=args.spool_dir.resolve(),
        cross_cache_length=args.cross_cache_length,
    )
    layout = SharedLayoutProcess(
        paths=paths,
        model_path=args.layout_model.resolve(),
        cache_dir=args.layout_cache.resolve(),
        device=args.device,
        lanes=args.layout_lanes,
        batch_size=args.layout_batch_size,
        threshold=args.layout_threshold,
        page_spool_root=args.spool_dir.resolve(),
    )
    submitted = 0
    layout_items = []
    for item in layout.iter_pages():
        layout_items.append(item)
        submitted += 1
        if submitted % args.progress_every == 0 or submitted == len(paths):
            print(
                "UNIREC_LOWMEM_LAYOUT_PROGRESS "
                f"pages={submitted}/{len(paths)} "
                f"elapsed_s={time.perf_counter() - frontend_started:.3f}",
                flush=True,
            )
    layout.close()
    print(
        "UNIREC_LOWMEM_LAYOUT_RELEASED "
        f"pages={len(layout_items)} elapsed_s={time.perf_counter() - frontend_started:.3f}",
        flush=True,
    )
    for item in layout_items:
        crop_pool.submit(
            page_index=int(item["page_index"]),
            path=Path(item["path"]),
            rgb=None,
            rgb_descriptor=None,
            layout_result=item["layout_result"],
            started_at=float(item["started_at"]),
        )
    del layout_items
    payloads, frontend_summary = crop_pool.finish()
    crop_pool.close()
    frontend_wall_s = time.perf_counter() - frontend_started
    print(
        "UNIREC_LOWMEM_FRONTEND_END "
        f"pages={len(payloads)} crops={frontend_summary.crop_count} "
        f"wall_s={frontend_wall_s:.3f}",
        flush=True,
    )
    frontend_payloads_path = args.output_dir / "frontend_payloads.json"
    if args.defer_output_write:
        frontend_payloads_path.write_text(
            json.dumps(payloads, ensure_ascii=False),
            encoding="utf-8",
        )

    # Accelerator imports start only after every frontend process has exited.
    os.environ["UNIREC_STATIC_CACHE_LEN"] = str(args.self_cache_length)
    os.environ["UNIREC_STATIC_CROSS_CACHE_LEN"] = str(args.cross_cache_length)
    os.environ["UNIREC_VISION_BUCKET_PRESET"] = args.vision_bucket_preset
    os.environ["UNIREC_RECOGNITION_INPUT_CONTRACT"] = "compact_uint8_hwc"
    import torch
    import torch_npu

    from bounded_vision_owner import BoundedVisionOwner
    from continuous_unirec import (
        ContinuousCompletedItem,
        ContinuousReadyItem,
        ContinuousUniRecDecoder,
        ContinuousWorkerPrefilledItem,
        production_decode_cache_parent,
    )
    from decode_model_optimizations import (
        apply_decode_model_optimizations,
        decode_cache_variant_root,
    )
    from host_memory_diagnostics import process_snapshot
    from modeling_optimized_unirec import OptimizedUniRecRunner
    from post_warmup_host_cleanup import cleanup_after_warmup
    from run_opendoc_batched_unirec import (
        assemble_page,
        iter_greedy_text_packs,
        warmup_configured_graphs,
    )
    from tbe_compiler_lifecycle import deinitialize_after_warmup
    from vision_bucket_presets import resolve_vision_bucket_specs
    from vision_full_batch import BucketedFullVisionRuntime

    torch_npu.npu.set_compile_mode(jit_compile=False)
    pipeline = None
    infer_doc_onnx = None
    if not args.defer_output_write:
        import cv2
        import numpy as np

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
    pages, vision_records = _payload_to_pages(payloads)
    del payloads

    recognition_started = time.perf_counter()
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
    vision_owner = BoundedVisionOwner(
        vision_runtime,
        lanes=args.vision_lanes,
        same_key_shards=args.vision_same_key_shards,
        sharded_key_count=args.vision_sharded_key_count,
    )
    text_runtime = runner._get_compiled_packed_text_prefill_runtime()
    encoded: list[Any] | None = None
    vision_summary: dict[str, Any] | None = None
    post_vision_cleanup: dict[str, Any] | None = None
    vision_tbe_deinit: dict[str, Any] | None = None
    post_vision_memory: dict[str, Any] | None = None
    vision_wall_s = 0.0
    if args.recognition_schedule == "two_phase":
        encoded, vision_summary = vision_owner.encode(vision_records)
        vision_owner.close()
        # All spatial outputs are materialized. The decoder and packed text
        # graph do not use the vision encoder again.
        runner.model.encoder = None
        del vision_owner, vision_runtime
        gc.collect()
        torch.npu.empty_cache()
        post_vision_cleanup = cleanup_after_warmup("low_memory_vision_complete")
        vision_tbe_deinit = deinitialize_after_warmup(
            "low_memory_vision_complete"
        )
        post_vision_memory = process_snapshot()
        vision_wall_s = time.perf_counter() - recognition_started
        print(
            "UNIREC_LOWMEM_VISION_END "
            f"crops={len(encoded)} graphs={vision_summary['graph_count']} "
            f"wall_s={vision_summary['wall_s']:.3f}",
            flush=True,
        )

    text_stream = torch.npu.Stream(device=torch.device(args.device))
    text_input = torch.zeros(
        (1, text_runtime.bucket, runner.config.d_model),
        dtype=runner.dtype,
        device=torch.device(args.device),
    )
    with torch.inference_mode(), torch.npu.stream(text_stream):
        text_runtime.compiled(text_input)
    text_stream.synchronize()
    text_runtime._first_call = False
    del text_input

    runner.compile_cache_dir = decode_cache_variant_root(
        production_decode_cache_parent(args.decode_cache_parent.resolve()),
        weight_format="nz",
        lm_head_rows=57344,
    )
    decode_warmup = warmup_configured_graphs(
        args=SimpleNamespace(
            text_prefill_mode="eager",
            decode_mode="compiled_ifa",
            compile_backend="torchair",
            decode_batch_size=args.decode_batch_size,
        ),
        runner=runner,
        vision_atlas_runtime=None,
        passes=2,
        warmup_decode=True,
    )
    tbe_deinit = deinitialize_after_warmup("low_memory_owner_ready")
    if args.recognition_schedule == "streaming":
        post_vision_cleanup = cleanup_after_warmup("low_memory_stream_ready")
    ready_memory = process_snapshot()

    ready_queue: Queue[Any] = Queue(maxsize=16)
    source_index_by_request_id = {
        str(record["request_id"]): int(record["source_index"])
        for record in vision_records
    }
    producer_stats: dict[str, Any] = {
        "groups": 0,
        "crops": 0,
        "wall_s": 0.0,
        "cross_kv_d2h_s": 0.0,
        "cross_kv_storage": "npu_bounded_queue",
        "cross_kv_npu_bytes": 0,
        "recognition_schedule": args.recognition_schedule,
    }

    def queue_page_prefill(page: Any, encoded_items: dict[int, Any]) -> None:
        groups = iter_greedy_text_packs(iter(page.crops), runner=runner)
        for use_packed, crop_group in groups:
            if not use_packed:
                raise RuntimeError(
                    "accuracy-safe low-memory path encountered a text "
                    f"prefill fallback: {crop_group[0].request_id}"
                )
            encoded_group = []
            for crop in crop_group:
                index = source_index_by_request_id[crop.request_id]
                item = encoded_items.pop(index)
                encoded_group.append((item.hidden_states, item.prep))
            with torch.inference_mode(), torch.npu.stream(text_stream):
                items = runner.prefill_encoder_hidden_states_packed_for_cohort(
                    encoded_group,
                    profile_device_stages=False,
                    decode_ready=False,
                )
                npu_exports = []
                for item in items:
                    actual_length = int(
                        item.kv_cache.actual_cross_attention_length or 0
                    )
                    if actual_length <= 0:
                        raise RuntimeError("text prefill produced an empty cross cache")
                    packed_npu = torch.stack(
                        tuple(
                            tensor[:, :, :actual_length, :]
                            for tensor in (
                                *item.kv_cache.cross_key_cache,
                                *item.kv_cache.cross_value_cache,
                            )
                        ),
                        dim=0,
                    ).contiguous()
                    npu_exports.append((packed_npu, actual_length))
            text_stream.synchronize()
            producer_stats["groups"] += 1
            for crop, item, export in zip(crop_group, items, npu_exports):
                packed_npu, actual_length = export
                producer_stats["cross_kv_npu_bytes"] += int(
                    packed_npu.numel() * packed_npu.element_size()
                )
                prefilled = ContinuousWorkerPrefilledItem(
                    packed_cross_kv=packed_npu,
                    prep=dict(item.prep),
                    prefill_s=float(item.prefill_s),
                    actual_cross_attention_length=int(actual_length),
                    prefill_device_stage_s=item.prefill_device_stage_s,
                    text_prefill_execution=str(item.text_prefill_execution),
                    text_prefill_real_source_tokens=int(
                        item.text_prefill_real_source_tokens or actual_length
                    ),
                    text_prefill_physical_source_tokens=int(
                        item.text_prefill_physical_source_tokens
                        or item.text_prefill_real_source_tokens
                        or actual_length
                    ),
                )

                def release(prefilled: Any = prefilled) -> None:
                    prefilled.packed_cross_kv = None

                ready_queue.put(
                    ContinuousReadyItem(
                        request_id=crop.request_id,
                        payload=crop,
                        prefilled=prefilled,
                        on_admitted=release,
                    )
                )
                producer_stats["crops"] += 1
            del items, npu_exports, encoded_group

    def producer() -> None:
        nonlocal vision_owner, vision_runtime, vision_summary
        nonlocal post_vision_memory, vision_tbe_deinit, vision_wall_s
        started = time.perf_counter()
        try:
            if args.recognition_schedule == "two_phase":
                assert encoded is not None
                encoded_items = {
                    index: item for index, item in enumerate(encoded) if item is not None
                }
                encoded.clear()
                for page in pages:
                    queue_page_prefill(page, encoded_items)
            else:
                encoded_items: dict[int, Any] = {}
                remaining = {
                    int(page.page_index): len(page.crops)
                    for page in pages
                    if page.crops
                }
                pages_by_index = {int(page.page_index): page for page in pages}
                queued_pages: set[int] = set()
                page_queue: Queue[int | None] = Queue(maxsize=16)
                text_errors: list[BaseException] = []

                def stream_text_prefill() -> None:
                    try:
                        while True:
                            page_index = page_queue.get()
                            if page_index is None:
                                return
                            queue_page_prefill(
                                pages_by_index[page_index],
                                encoded_items,
                            )
                    except BaseException as exception:
                        text_errors.append(exception)
                        ready_queue.put(exception)

                text_thread = Thread(
                    target=stream_text_prefill,
                    name="unirec-lowmem-stream-text",
                    daemon=True,
                )
                text_thread.start()

                def on_encoded_batch(batch: list[Any]) -> None:
                    ready_pages = set()
                    for item in batch:
                        index = int(item.source_index)
                        if index in encoded_items:
                            raise RuntimeError(f"duplicate streamed vision item {index}")
                        encoded_items[index] = item
                        page_index = int(vision_records[index]["crop"].page_index)
                        remaining[page_index] -= 1
                        if remaining[page_index] == 0:
                            ready_pages.add(page_index)
                    for page_index in sorted(ready_pages):
                        if page_index in queued_pages:
                            raise RuntimeError(f"page {page_index} queued twice")
                        queued_pages.add(page_index)
                        page_queue.put(page_index)

                _unused, vision_summary = vision_owner.encode(
                    vision_records,
                    on_encoded_batch=on_encoded_batch,
                    retain_outputs=False,
                )
                producer_stats["vision_complete_s"] = (
                    time.perf_counter() - started
                )
                page_queue.put(None)
                text_thread.join(timeout=120.0)
                if text_thread.is_alive():
                    raise RuntimeError("streaming text-prefill thread did not stop")
                if text_errors:
                    raise RuntimeError("streaming text prefill failed") from text_errors[0]
                producer_stats["text_complete_s"] = time.perf_counter() - started
                if encoded_items:
                    raise RuntimeError(
                        f"streaming vision retained {len(encoded_items)} crops"
                    )
                expected_pages = {index for index, count in remaining.items() if count == 0}
                if queued_pages != expected_pages:
                    raise RuntimeError(
                        "streaming page completion mismatch: "
                        f"queued={len(queued_pages)} expected={len(expected_pages)}"
                    )
                vision_owner.close()
                runner.model.encoder = None
                del vision_owner, vision_runtime
                gc.collect()
                torch.npu.empty_cache()
                vision_tbe_deinit = deinitialize_after_warmup(
                    "low_memory_stream_vision_complete"
                )
                post_vision_memory = process_snapshot()
                vision_wall_s = time.perf_counter() - recognition_started
                print(
                    "UNIREC_LOWMEM_VISION_END "
                    f"crops={len(vision_records)} "
                    f"graphs={vision_summary['graph_count']} "
                    f"wall_s={vision_summary['wall_s']:.3f}",
                    flush=True,
                )
            producer_stats["wall_s"] = time.perf_counter() - started
            ready_queue.put(None)
        except BaseException as exception:
            ready_queue.put(exception)

    producer_thread = Thread(
        target=producer,
        name="unirec-lowmem-text-prefill",
        daemon=True,
    )
    producer_thread.start()

    def ready_source() -> Any:
        while True:
            item = ready_queue.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    trace_path = args.output_dir / "recognition_trace.jsonl"
    trace_file = trace_path.open("w", encoding="utf-8")
    completed_crops = 0

    def complete(item: ContinuousCompletedItem) -> None:
        nonlocal completed_crops
        item.payload.result = item.result
        trace_file.write(
            json.dumps(
                {
                    "request_id": item.request_id,
                    "page_index": item.payload.page_index,
                    "crop_index": item.payload.crop_index,
                    "label": item.payload.label,
                    "text": item.result["text"],
                    "generated_ids": item.result["generated_ids"],
                    "generated_token_count": item.result["generated_token_count"],
                    "prep": item.result["prep"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        completed_crops += 1
        if completed_crops % 1000 == 0 or completed_crops == len(vision_records):
            print(
                "UNIREC_LOWMEM_DECODE_PROGRESS "
                f"crops={completed_crops}/{len(vision_records)}",
                flush=True,
            )

    decode_started = time.perf_counter()
    decode_summary = ContinuousUniRecDecoder(
        runner=runner,
        batch_size=args.decode_batch_size,
        max_length=args.max_length,
        decode_mode="compiled_ifa",
        compile_backend="torchair",
        admission_prefetch_depth=0,
        self_cache_length=args.self_cache_length,
        cross_cache_length=args.cross_cache_length,
    ).run(
        ready_source(),
        on_complete=complete,
        graph_warmup_passes=2,
    )
    decode_wall_s = time.perf_counter() - decode_started
    producer_thread.join(timeout=60.0)
    if producer_thread.is_alive():
        raise RuntimeError("text-prefill producer did not stop")
    trace_file.close()

    write_started = time.perf_counter()
    written = 0
    if not args.defer_output_write:
        assert pipeline is not None and infer_doc_onnx is not None
        with Path(os.devnull).open("w", encoding="utf-8") as devnull:
            for page in pages:
                result = assemble_page(
                    page=page,
                    pipeline=pipeline,
                    infer_doc_onnx=infer_doc_onnx,
                )
                if any(
                    bool(row.get("is_image", False))
                    for row in result["recognition_results"]
                ):
                    page_image = cv2.imread(result["input_path"], cv2.IMREAD_COLOR)
                    if page_image is None:
                        raise RuntimeError(f"failed to reload page {result['input_path']}")
                    result["_page_image"] = page_image
                else:
                    result["_page_image"] = np.empty((0, 0, 3), dtype=np.uint8)
                with redirect_stdout(devnull):
                    pipeline.save_to_json(result, str(args.output_dir))
                    pipeline.save_to_markdown(result, str(args.output_dir))
                for crop in page.crops:
                    crop.result = None
                written += 1
                if written % args.progress_every == 0 or written == len(pages):
                    print(
                        f"UNIREC_LOWMEM_WRITE_PROGRESS pages={written}/{len(pages)}",
                        flush=True,
                    )
    write_s = time.perf_counter() - write_started
    process_wall_s = time.perf_counter() - process_started
    summary = {
        "schema": "unirec_low_memory_full_pipeline_v1",
        "status": "pass",
        "commit": os.popen("git rev-parse HEAD").read().strip(),
        "chip": torch_npu.npu.get_device_name(0),
        "page_count": len(pages),
        "crop_count": len(vision_records),
        "pages_per_s": len(pages) / process_wall_s,
        "process_wall_s": process_wall_s,
        "frontend_wall_s": frontend_wall_s,
        "vision_phase_wall_s": vision_wall_s,
        "decode_wall_s": decode_wall_s,
        "write_wall_s": write_s,
        "output_write_deferred": bool(args.defer_output_write),
        "frontend": asdict(frontend_summary),
        "layout": layout.summary,
        "layout_ready_memory": layout.ready["snapshot"],
        "vision": vision_summary,
        "post_vision_cleanup": post_vision_cleanup,
        "vision_tbe_deinit": vision_tbe_deinit,
        "post_vision_memory": post_vision_memory,
        "text_prefill": producer_stats,
        "decode": decode_summary,
        "decode_optimizations": decode_optimizations,
        "decode_warmup": decode_warmup,
        "tbe_deinit": tbe_deinit,
        "recognition_owner_ready_memory": ready_memory,
        "final_memory": process_snapshot(),
        "settings": {
            "workers": args.workers,
            "recognition_threads": args.recognition_threads,
            "layout_lanes": args.layout_lanes,
            "layout_batch_size": args.layout_batch_size,
            "layout_threshold": args.layout_threshold,
            "vision_bucket_preset": args.vision_bucket_preset,
            "vision_lanes": args.vision_lanes,
            "vision_same_key_shards": args.vision_same_key_shards,
            "vision_sharded_key_count": args.vision_sharded_key_count,
            "recognition_schedule": args.recognition_schedule,
            "decode_batch_size": args.decode_batch_size,
            "cross_cache_length": args.cross_cache_length,
            "self_cache_length": args.self_cache_length,
        },
        "artifacts": {
            "output_dir": str(args.output_dir.resolve()),
            "spool_dir": str(args.spool_dir.resolve()),
            "recognition_trace": str(trace_path.resolve()),
            "frontend_payloads": (
                str(frontend_payloads_path.resolve())
                if args.defer_output_write
                else None
            ),
        },
    }
    summary_path = args.output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("UNIREC_LOWMEM_SUMMARY " + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
