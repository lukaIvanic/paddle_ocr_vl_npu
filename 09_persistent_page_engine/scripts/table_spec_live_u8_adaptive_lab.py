#!/usr/bin/env python3
"""Run live U8 table drafts and adaptive verification with safe CPU overlap."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(HERE))

import table_row_ocr_lab as row_lab  # noqa: E402
import table_spec_adaptive_k_lab as adaptive_lab  # noqa: E402
import table_spec_decode_lab as fixed_lab  # noqa: E402
from paddleocr_vl.model.text_spec_verify import (  # noqa: E402
    torchair_cache_dir_for_spec_shape,
)
from paddleocr_vl.model.token_selection import TOKEN_SELECTION_CHOICES  # noqa: E402
from paddleocr_vl.serving.table_speculative import TableDraftMatcher  # noqa: E402
from paddleocr_vl.serving.types import RecognitionRequest  # noqa: E402
from pipeline.layout_output import normalize_recognition_text  # noqa: E402
from table_row_split_lab import load_crop  # noqa: E402


DEFAULT_TAIL_IDS = (
    "page_000263_table_box_id_7",
    "page_000271_table_box_id_1",
    "page_000277_table_box_id_1",
    "page_000279_table_box_id_0",
    "page_000279_table_box-fy04hrwa",
    "page_000288_table_box_id_1",
    "page_000290_table_box_id_1",
    "page_001595_table_box_id_1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/workspace/datasets/OmniDocBench/images"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/workspace/models/PaddleOCR-VL-1.6"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-id", action="append", default=[])
    parser.add_argument(
        "--cold-request-id",
        default="page_000010_table_box_id_1",
        help="Run this page through the complete pipeline before measured pages.",
    )
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument(
        "--measurement-passes",
        type=int,
        default=1,
        help="Run every selected page this many times; only the final pass measures.",
    )

    parser.add_argument(
        "--b1-decode-optimization",
        default="combined_apply_complete_layer_prefetch1_rope_lut",
    )
    parser.add_argument("--b1-decode-vocab-token-ids", type=Path, required=True)
    parser.add_argument("--b1-cache-length", type=int, default=4096)
    parser.add_argument("--b1-max-new-tokens", type=int, default=4096)
    parser.add_argument("--b1-vision-buckets", default="4096")

    parser.add_argument(
        "--draft-decode-optimization",
        default="combined_apply_complete_layer_prefetch1_rope_lut",
    )
    parser.add_argument("--draft-decode-vocab-token-ids", type=Path, required=True)
    parser.add_argument("--draft-cache-length", type=int, default=768)
    parser.add_argument(
        "--draft-row-count",
        type=int,
        default=8,
        help="Number of snapped horizontal row drafts (U).",
    )
    parser.add_argument("--draft-batch-size", type=int, default=8)
    parser.add_argument(
        "--draft-vision-packing",
        choices=("greedy", "cohort"),
        default="greedy",
    )
    parser.add_argument("--draft-vision-pack-target", type=int, default=2304)
    parser.add_argument(
        "--draft-prefill-layout",
        choices=("packed_b1", "fixed_b8"),
        default="packed_b1",
    )
    parser.add_argument("--draft-batched-vision-shapes", default="8x640,8x768")
    parser.add_argument("--draft-batched-text-shape", default="8x256")
    parser.add_argument("--row-overlap-px", type=int, default=3)
    parser.add_argument(
        "--compact-uint8-preprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--image-resize-backend",
        choices=("pillow", "kornia_rs"),
        default="pillow",
    )
    parser.add_argument(
        "--target-cpu-delay-ms",
        type=float,
        default=0.0,
        help=(
            "Delay full-table CPU preparation so the row-draft frontend can "
            "run first; the work still overlaps draft decoding."
        ),
    )
    parser.add_argument(
        "--overlap-target-cpu-preparation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Overlap full-table CPU preprocessing with draft recognition. "
            "Disable to finish full-table preprocessing before draft-row "
            "preprocessing starts."
        ),
    )

    parser.add_argument("--k-values", default="7,15,31,63")
    parser.add_argument("--initial-k", type=int, default=15)
    parser.add_argument(
        "--verifier-optimization",
        default="combined_apply_spec_prefetch_mrope",
    )
    parser.add_argument(
        "--per-call-device-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--allow-compile", action="store_true")
    parser.add_argument(
        "--token-selection",
        default="greedy",
        choices=TOKEN_SELECTION_CHOICES,
    )
    parser.add_argument("--min-pixels", type=int, default=28224)
    parser.add_argument("--max-pixels", type=int, default=802816)
    parser.add_argument(
        "--vision-buckets",
        default="256,384,512,640,768,1408,1920,2048,2304,2944,4096",
    )
    parser.add_argument("--b1-text-buckets", default="1152")
    parser.add_argument(
        "--decode-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_torchair",
    )
    parser.add_argument(
        "--vision-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_vision_torchair",
    )
    parser.add_argument(
        "--vision-batched-cache-dir",
        type=Path,
        default=(
            REPO_ROOT
            / ".runtime_cache/09_persistent_page_engine_vision_batched_torchair"
        ),
    )
    parser.add_argument(
        "--text-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_text_torchair",
    )
    parser.add_argument(
        "--text-packed-cache-dir",
        type=Path,
        default=(
            REPO_ROOT / ".runtime_cache/09_persistent_page_engine_text_packed_torchair"
        ),
    )
    parser.add_argument(
        "--text-batched-cache-dir",
        type=Path,
        default=(
            REPO_ROOT
            / ".runtime_cache/09_persistent_page_engine_text_batched_torchair"
        ),
    )
    parser.add_argument(
        "--k-cache-root",
        action="append",
        default=[],
        metavar="K=PATH",
    )
    return parser.parse_args()


def _parse_shapes(value: str) -> tuple[tuple[int, int], ...]:
    shapes: list[tuple[int, int]] = []
    for piece in value.split(","):
        batch_text, separator, sequence_text = piece.strip().lower().partition("x")
        if not separator:
            raise ValueError(f"invalid BxS shape: {piece!r}")
        shapes.append((int(batch_text), int(sequence_text)))
    if not shapes:
        raise ValueError("at least one BxS shape is required")
    return tuple(shapes)


def _b1_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        decode_optimization=args.b1_decode_optimization,
        decode_vocab_token_ids=args.b1_decode_vocab_token_ids,
        cache_length=args.b1_cache_length,
        max_new_tokens=args.b1_max_new_tokens,
        token_selection=args.token_selection,
        decode_cache_dir=args.decode_cache_dir,
        vision_cache_dir=args.vision_cache_dir,
        text_cache_dir=args.text_cache_dir,
        vision_buckets=args.b1_vision_buckets,
        text_buckets=args.b1_text_buckets,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        compact_uint8_preprocess=args.compact_uint8_preprocess,
        image_resize_backend=args.image_resize_backend,
    )


def _draft_args(args: argparse.Namespace) -> SimpleNamespace:
    fixed_b8 = args.draft_prefill_layout == "fixed_b8"
    return SimpleNamespace(
        model=args.model,
        decode_optimization=args.draft_decode_optimization,
        decode_vocab_token_ids=args.draft_decode_vocab_token_ids,
        decode_batch_size=args.draft_batch_size,
        cache_length=args.draft_cache_length,
        token_selection=args.token_selection,
        decode_cache_dir=args.decode_cache_dir,
        vision_cache_dir=args.vision_cache_dir,
        text_cache_dir=args.text_cache_dir,
        text_packed_cache_dir=args.text_packed_cache_dir,
        vision_buckets=args.vision_buckets,
        vision_packing=("fixed_batch" if fixed_b8 else args.draft_vision_packing),
        vision_pack_target=args.draft_vision_pack_target,
        vision_batched_cache_dir=(
            args.vision_batched_cache_dir.resolve() if fixed_b8 else None
        ),
        vision_batched_shapes=(
            _parse_shapes(args.draft_batched_vision_shapes) if fixed_b8 else None
        ),
        batched_prefill_require_warm_cache=not args.allow_compile,
        text_packing=("fixed_batch" if fixed_b8 else "production_group"),
        text_batched_shape=(
            _parse_shapes(args.draft_batched_text_shape)[0] if fixed_b8 else None
        ),
        text_batched_cache_dir=(
            args.text_batched_cache_dir.resolve() if fixed_b8 else None
        ),
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        compact_uint8_preprocess=args.compact_uint8_preprocess,
        image_resize_backend=args.image_resize_backend,
    )


def _select_records(
    records: list[dict[str, Any]],
    request_ids: list[str],
) -> list[dict[str, Any]]:
    by_id = {str(record["request_id"]): record for record in records}
    selected_ids = request_ids or list(DEFAULT_TAIL_IDS)
    missing = [request_id for request_id in selected_ids if request_id not in by_id]
    if missing:
        raise KeyError(f"unknown request IDs: {missing}")
    return [by_id[request_id] for request_id in selected_ids]


def _exact_target_crop_from_raw(
    source: dict[str, Any],
    raw_image: Any,
) -> Any:
    crop = raw_image
    trim_box = source.get("trim_box_in_raw_crop")
    if trim_box is not None:
        crop = crop.crop(tuple(int(value) for value in trim_box))
    expected = tuple(int(value) for value in source["crop_size"])
    if crop.size != expected:
        raise ValueError(
            f"{source['request_id']}: reconstructed crop {crop.size} != {expected}"
        )
    return crop


def _prepare_rows(
    source: dict[str, Any],
    raw_image: Any,
    args: argparse.Namespace,
) -> tuple[list[RecognitionRequest], list[tuple[int, int, Any]], dict[str, Any]]:
    if args.draft_row_count <= 0:
        raise ValueError("draft_row_count must be positive")
    strategy_name = f"uniform_{args.draft_row_count}_snapped"
    prepare_started = time.perf_counter()
    prepare_cpu_started = time.thread_time()
    prepared, rotation_cw, source_size, split_s = row_lab.prepare_strategy_inputs(
        source,
        raw_image,
        (strategy_name,),
        resize_full_table_before_split=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    strategy = prepared[strategy_name]
    crop_started = time.perf_counter()
    rows = row_lab.crop_rows(
        strategy["image"],
        strategy["proposal"].boundaries,
        args.row_overlap_px,
        pad_to_factor=28,
        min_pixels=args.min_pixels,
        max_pixels=None,
    )
    row_crop_s = time.perf_counter() - crop_started
    requests: list[RecognitionRequest] = []
    for row_index, (_top, _bottom, row_image) in enumerate(rows):
        row_pixels = int(row_image.width * row_image.height)
        requests.append(
            RecognitionRequest(
                request_id=(
                    f"{source['request_id']}:{strategy_name}:row_{row_index:04d}"
                ),
                crop=row_image,
                prompt="Table Recognition:",
                min_pixels=row_pixels,
                max_pixels=row_pixels,
                source_crop_size=row_image.size,
            )
        )
    return requests, rows, {
        "row_prepare_wall_s": time.perf_counter() - prepare_started,
        "row_prepare_thread_cpu_s": time.thread_time() - prepare_cpu_started,
        "split_cpu_s": float(split_s),
        "row_crop_cpu_s": row_crop_s,
        "row_draft_rotation_cw": int(rotation_cw),
        "row_draft_source_size": source_size,
        "row_draft_crop_size": list(strategy["image"].size),
        "boundaries": list(strategy["proposal"].boundaries),
        "split_diagnostics": strategy["proposal"].diagnostics,
    }


def _run_draft(
    recognizer: Any,
    source: dict[str, Any],
    requests: list[RecognitionRequest],
    rows: list[tuple[int, int, Any]],
    *,
    draft_row_count: int,
) -> tuple[dict[str, Any], float]:
    strategy_name = f"uniform_{draft_row_count}_snapped"
    row_results: list[dict[str, Any]] = []

    def emit(result: Any) -> None:
        payload = asdict(result)
        payload["raw_text"] = payload["text"]
        row_index = int(result.request_id.rsplit("_", 1)[-1])
        payload["row_index"] = row_index
        payload["row_y"] = list(rows[row_index][:2])
        row_results.append(payload)

    started = time.perf_counter()
    schedule = recognizer.run(
        requests,
        schedule_id=f"live-u{draft_row_count}:{source['request_id']}",
        emit_result=emit,
    )
    wall_s = time.perf_counter() - started
    row_results.sort(key=lambda item: int(item["row_index"]))
    return {
        "request_id": str(source["request_id"]),
        "strategy": strategy_name,
        "page_name": source["page_name"],
        "rows": row_results,
        "schedule": asdict(schedule),
    }, wall_s


@torch.inference_mode()
def _timed_cpu_prepare(
    recognizer: Any,
    request: RecognitionRequest,
    submitted_at: float,
    delay_s: float,
) -> tuple[Any, float, float, float]:
    if delay_s > 0:
        time.sleep(delay_s)
    started = time.perf_counter()
    cpu_started = time.thread_time()
    prepared = recognizer._prepare_cpu(request, submitted_at)
    return prepared, started, time.perf_counter(), time.thread_time() - cpu_started


def _timed_matcher(
    draft: dict[str, Any],
    tokenizer: Any,
    *,
    eos_token_id: int,
    block_size: int,
) -> tuple[TableDraftMatcher, float, float, float]:
    started = time.perf_counter()
    cpu_started = time.thread_time()
    # The matcher index is acyclic. Cyclic-GC scans can otherwise add large,
    # input-dependent pauses while this worker overlaps target NPU prefill.
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        matcher = TableDraftMatcher(
            draft,
            tokenizer,
            eos_token_id=eos_token_id,
            block_size=block_size,
        )
    finally:
        if gc_was_enabled:
            gc.enable()
    return matcher, started, time.perf_counter(), time.thread_time() - cpu_started


def _intersection_seconds(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _run_one(
    *,
    source: dict[str, Any],
    raw_image: Any,
    crop_preload_wall_s: float,
    measured: bool,
    args: argparse.Namespace,
    b1_recognizer: Any,
    draft_recognizer: Any,
    runtime: adaptive_lab.AdaptiveKTableSpeculativeDecodeRuntime,
    cpu_executor: ThreadPoolExecutor,
) -> dict[str, Any]:
    request_id = str(source["request_id"])
    # The latency contract begins with an already-materialized table crop.
    e2e_started = time.perf_counter()
    process_cpu_started = time.process_time()
    target_crop = _exact_target_crop_from_raw(source, raw_image)
    row_requests, row_crops, row_timing = _prepare_rows(
        source, raw_image, args
    )

    target_request = fixed_lab.request_for(source, target_crop, _b1_args(args))
    cpu_submitted = time.perf_counter()
    cpu_future = None
    if args.overlap_target_cpu_preparation:
        cpu_future = cpu_executor.submit(
            _timed_cpu_prepare,
            b1_recognizer,
            target_request,
            cpu_submitted,
            max(0.0, float(args.target_cpu_delay_ms) / 1000.0),
        )
    else:
        prepared, cpu_started, cpu_finished, cpu_prepare_thread_s = (
            _timed_cpu_prepare(
                b1_recognizer,
                target_request,
                cpu_submitted,
                max(0.0, float(args.target_cpu_delay_ms) / 1000.0),
            )
        )
    draft_started = time.perf_counter()
    draft, draft_wall_s = _run_draft(
        draft_recognizer,
        source,
        row_requests,
        row_crops,
        draft_row_count=args.draft_row_count,
    )
    draft_finished = time.perf_counter()
    cpu_wait_started = time.perf_counter()
    if cpu_future is not None:
        prepared, cpu_started, cpu_finished, cpu_prepare_thread_s = (
            cpu_future.result()
        )
        cpu_wait_s = time.perf_counter() - cpu_wait_started
    else:
        cpu_wait_s = 0.0

    matcher_future = cpu_executor.submit(
        _timed_matcher,
        draft,
        b1_recognizer.tokenizer,
        eos_token_id=int(b1_recognizer.model.config.eos_token_id),
        block_size=args.initial_k,
    )
    prefill_started = time.perf_counter()
    prefilled = b1_recognizer.prefill_prepared_one(prepared)
    prefill_finished = time.perf_counter()
    target_prefill_timing_s = dict(prefilled.timing_s)
    target_prefill_device_stage_s = dict(prefilled.device_stage_s)
    matcher_wait_started = time.perf_counter()
    matcher, matcher_started, matcher_finished, matcher_thread_s = (
        matcher_future.result()
    )
    matcher_wait_s = time.perf_counter() - matcher_wait_started
    verify_started = time.perf_counter()
    result = runtime.decode(
        prefilled,
        matcher,
        max_new_tokens=args.b1_max_new_tokens,
    )
    verify_finished = time.perf_counter()
    e2e_s = verify_finished - e2e_started
    process_cpu_s = time.process_time() - process_cpu_started

    reference_tokens = fixed_lab.target_tokens(source)
    cpu_prepare_s = cpu_finished - cpu_started
    matcher_build_s = matcher_finished - matcher_started
    explicit_cpu_wall_sum_s = (
        float(row_timing["row_prepare_wall_s"])
        + cpu_prepare_s
        + matcher_build_s
    )
    explicit_cpu_thread_sum_s = (
        float(row_timing["row_prepare_thread_cpu_s"])
        + cpu_prepare_thread_s
        + matcher_thread_s
    )
    cpu_hidden_s = _intersection_seconds(
        cpu_started, cpu_finished, draft_started, draft_finished
    )
    matcher_hidden_s = _intersection_seconds(
        matcher_started, matcher_finished, prefill_started, prefill_finished
    )
    payload = {
        "request_id": request_id,
        "page_name": source["page_name"],
        "measured": bool(measured),
        "crop_size": list(target_crop.size),
        "input_tokens": int(prefilled.input_tokens),
        "projected_image_tokens": int(prefilled.projected_image_tokens),
        "saved_reference_tokens": len(reference_tokens),
        "timing_s": {
            "e2e_complete_wall": e2e_s,
            "table_crop_preload_wall_excluded": float(crop_preload_wall_s),
            "host_process_cpu_total": process_cpu_s,
            "host_process_cpu_per_e2e_wall": (
                process_cpu_s / e2e_s if e2e_s > 0 else None
            ),
            "explicit_cpu_stage_wall_sum": explicit_cpu_wall_sum_s,
            "explicit_cpu_stage_thread_sum": explicit_cpu_thread_sum_s,
            **row_timing,
            "draft_recognition_wall": draft_wall_s,
            "target_cpu_prepare": cpu_prepare_s,
            "target_cpu_prepare_scheduled_delay": (
                max(0.0, float(args.target_cpu_delay_ms) / 1000.0)
            ),
            "target_cpu_prepare_thread_cpu": cpu_prepare_thread_s,
            "target_cpu_prepare_consumer_wait": cpu_wait_s,
            "target_cpu_prepare_hidden_by_draft": cpu_hidden_s,
            "matcher_build": matcher_build_s,
            "matcher_thread_cpu": matcher_thread_s,
            "matcher_consumer_wait": matcher_wait_s,
            "matcher_hidden_by_target_prefill": matcher_hidden_s,
            "target_npu_prefill_wall": prefill_finished - prefill_started,
            "target_verify_wall": verify_finished - verify_started,
            "overlap_hidden_total": cpu_hidden_s + matcher_hidden_s,
            "sequential_work_estimate": e2e_s + cpu_hidden_s + matcher_hidden_s,
        },
        "target_prefill": {
            "timing_s": target_prefill_timing_s,
            "device_stage_s": target_prefill_device_stage_s,
        },
        "draft": draft,
        "speculative": result.to_dict(),
        "gt_html": source.get("gt_html"),
        "pred_html": normalize_recognition_text("table", result.text),
        "exact_saved_reference": result.token_ids == reference_tokens,
        "first_saved_difference": fixed_lab.first_difference(
            result.token_ids, reference_tokens
        ),
    }
    return payload


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
    all_targets = fixed_lab.read_jsonl(args.targets)
    selected = _select_records(all_targets, args.request_id)
    by_id = {str(record["request_id"]): record for record in all_targets}
    if args.cold_request_id not in by_id:
        raise KeyError(f"unknown cold request ID: {args.cold_request_id}")
    cold_source = by_id[args.cold_request_id]

    preloaded_crops: dict[str, Any] = {}
    crop_preload_wall_s: dict[str, float] = {}
    preload_started = time.perf_counter()
    for source in (cold_source, *selected):
        request_id = str(source["request_id"])
        if request_id in preloaded_crops:
            continue
        started = time.perf_counter()
        preloaded_crops[request_id] = load_crop(source, args.images_dir)
        crop_preload_wall_s[request_id] = time.perf_counter() - started
    crop_preload_total_s = time.perf_counter() - preload_started
    print(
        "TABLE_SPEC_LIVE_PROGRESS crops=preloaded "
        f"count={len(preloaded_crops)} wall_s={crop_preload_total_s:.3f}",
        flush=True,
    )

    k_values = adaptive_lab.parse_k_values(args.k_values)
    cache_roots = adaptive_lab.parse_k_cache_roots(
        args.k_cache_root,
        k_values=k_values,
        default=args.decode_cache_dir,
    )
    setup_started = time.perf_counter()
    print("TABLE_SPEC_LIVE_PROGRESS setup=draft_recognizer", flush=True)
    draft_recognizer = row_lab.build_recognizer(_draft_args(args))
    print("TABLE_SPEC_LIVE_PROGRESS setup=b1_recognizer", flush=True)
    b1_recognizer = fixed_lab.build_recognizer(_b1_args(args))
    cache_hits: dict[int, bool] = {}
    for value in k_values:
        spec_cache = torchair_cache_dir_for_spec_shape(
            cache_roots[value],
            draft_length=value,
            cache_length=args.b1_cache_length,
            dtype=b1_recognizer.dtype,
            device=b1_recognizer.device,
            model_dir=b1_recognizer.model_dir,
            linear_weight_format=str(
                b1_recognizer.weight_format["effective_mode"]
            ),
            optimization=args.verifier_optimization,
            token_selection=args.token_selection,
            preferred_token_id=b1_recognizer.math_open_token_id,
            alternate_preferred_token_id=b1_recognizer.math_slash_token_id,
            cell_start_token_ids=b1_recognizer.table_cell_token_ids,
        )
        cache_hits[value] = spec_cache.is_dir() and any(spec_cache.iterdir())
        if not cache_hits[value] and not args.allow_compile:
            raise RuntimeError(
                f"missing K{value}/KV{args.b1_cache_length} verifier cache"
            )
    print(
        "TABLE_SPEC_LIVE_PROGRESS setup=verifiers "
        + " ".join(
            f"K{value}={'hit' if cache_hits[value] else 'compile'}"
            for value in k_values
        ),
        flush=True,
    )
    runtime = adaptive_lab.AdaptiveKTableSpeculativeDecodeRuntime(
        b1_recognizer,
        k_values=k_values,
        initial_k=args.initial_k,
        cache_roots=cache_roots,
        verifier_optimization=args.verifier_optimization,
        record_device_timing=args.per_call_device_timing,
    )
    setup_s = time.perf_counter() - setup_started

    output_path = args.output_dir / "tables.jsonl"
    output_path.write_text("", encoding="utf-8")
    records: list[dict[str, Any]] = []
    warm_pass_records: list[dict[str, Any]] = []
    if args.measurement_passes <= 0:
        raise ValueError("measurement_passes must be positive")
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="table-spec-cpu-overlap",
    ) as cpu_executor:
        print(
            f"TABLE_SPEC_LIVE_PROGRESS cold_start={args.cold_request_id}",
            flush=True,
        )
        cold = _run_one(
            source=cold_source,
            raw_image=preloaded_crops[args.cold_request_id],
            crop_preload_wall_s=crop_preload_wall_s[args.cold_request_id],
            measured=False,
            args=args,
            b1_recognizer=b1_recognizer,
            draft_recognizer=draft_recognizer,
            runtime=runtime,
            cpu_executor=cpu_executor,
        )
        with output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(cold, ensure_ascii=False) + "\n")
        _write_json(args.output_dir / "run_progress.json", {
            "setup_s": setup_s,
            "cold": cold,
            "tables": records,
        })
        if args.settle_s > 0:
            time.sleep(args.settle_s)

        for pass_index in range(1, args.measurement_passes + 1):
            measure_pass = pass_index == args.measurement_passes
            print(
                "TABLE_SPEC_LIVE_PROGRESS "
                f"pass={pass_index}/{args.measurement_passes} "
                f"measured={measure_pass}",
                flush=True,
            )
            for index, source in enumerate(selected, start=1):
                payload = _run_one(
                    source=source,
                    raw_image=preloaded_crops[str(source["request_id"])],
                    crop_preload_wall_s=crop_preload_wall_s[
                        str(source["request_id"])
                    ],
                    measured=measure_pass,
                    args=args,
                    b1_recognizer=b1_recognizer,
                    draft_recognizer=draft_recognizer,
                    runtime=runtime,
                    cpu_executor=cpu_executor,
                )
                payload["measurement_pass"] = pass_index
                (records if measure_pass else warm_pass_records).append(payload)
                with output_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
                _write_json(args.output_dir / "run_progress.json", {
                    "setup_s": setup_s,
                    "cold": cold,
                    "warm_pass_tables": warm_pass_records,
                    "tables": records,
                })
                timing = payload["timing_s"]
                print(
                    "TABLE_SPEC_LIVE_RESULT "
                    f"pass={pass_index}/{args.measurement_passes} "
                    f"table={index}/{len(selected)} id={payload['request_id']} "
                    f"e2e={timing['e2e_complete_wall']:.3f}s "
                    f"draft={timing['draft_recognition_wall']:.3f}s "
                    f"prefill={timing['target_npu_prefill_wall']:.3f}s "
                    f"verify={timing['target_verify_wall']:.3f}s "
                    f"cpu={timing['host_process_cpu_total']:.3f}s "
                    f"hidden={timing['overlap_hidden_total']:.3f}s "
                    f"exact={payload['exact_saved_reference']}",
                    flush=True,
                )
                if args.settle_s > 0 and index < len(selected):
                    time.sleep(args.settle_s)

    e2e_values = [record["timing_s"]["e2e_complete_wall"] for record in records]
    summary = {
        "setup_s": setup_s,
        "crop_preload_total_wall_excluded": crop_preload_total_s,
        "environment": {
            "ascend_rt_visible_devices": os.environ.get(
                "ASCEND_RT_VISIBLE_DEVICES"
            ),
            "device": str(b1_recognizer.device),
        },
        "settings": {
            "cold_request_id": args.cold_request_id,
            "measurement_passes": args.measurement_passes,
            "measured_request_ids": [record["request_id"] for record in records],
            "draft_decode_optimization": args.draft_decode_optimization,
            "draft_vision_packing": args.draft_vision_packing,
            "draft_vision_pack_target": args.draft_vision_pack_target,
            "b1_decode_optimization": args.b1_decode_optimization,
            "verifier_optimization": args.verifier_optimization,
            "k_values": list(k_values),
            "initial_k": args.initial_k,
            "draft_row_count": int(args.draft_row_count),
            "row_strategy": f"uniform_{args.draft_row_count}_snapped",
            "latency_boundary": "in_memory_table_crop_to_verified_output",
            "source_page_load_and_bbox_crop_included": False,
            "target_cpu_overlap": (
                "during_draft_recognition"
                if args.overlap_target_cpu_preparation
                else "disabled_prepare_before_draft"
            ),
            "target_cpu_delay_ms": float(args.target_cpu_delay_ms),
            "matcher_overlap": "during_target_npu_prefill",
            "npu_overlap": False,
        },
        "cold": cold,
        "warm_pass_tables": warm_pass_records,
        "measured_tables": len(records),
        "exact_saved_reference": sum(
            bool(record["exact_saved_reference"]) for record in records
        ),
        "e2e_distribution_s": fixed_lab.distribution(e2e_values),
        "host_process_cpu_distribution_s": fixed_lab.distribution([
            record["timing_s"]["host_process_cpu_total"]
            for record in records
        ]),
        "explicit_cpu_stage_wall_distribution_s": fixed_lab.distribution([
            record["timing_s"]["explicit_cpu_stage_wall_sum"]
            for record in records
        ]),
        "explicit_cpu_stage_thread_distribution_s": fixed_lab.distribution([
            record["timing_s"]["explicit_cpu_stage_thread_sum"]
            for record in records
        ]),
        "tables": records,
    }
    _write_json(args.output_dir / "run_summary.json", summary)
    print(
        "TABLE_SPEC_LIVE_COMPLETE "
        f"tables={len(records)} exact={summary['exact_saved_reference']}/{len(records)} "
        f"output={args.output_dir / 'run_summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
