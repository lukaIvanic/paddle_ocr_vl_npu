#!/usr/bin/env python3
"""Run experiment 6 in memory-safe page chunks and aggregate the results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_SCRIPT = SCRIPT_DIR / "bench_page_pipeline_e2e.py"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def tok_per_s(count: int | float, seconds: float) -> float | None:
    seconds = float(seconds)
    if seconds <= 0:
        return None
    return float(count) / seconds


def stat_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "sum": 0.0, "avg": None, "min": None, "max": None, "p50": None, "p90": None}
    vals = sorted(float(value) for value in values)

    def percentile(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = (len(vals) - 1) * p
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        frac = pos - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac

    return {
        "count": int(len(vals)),
        "sum": float(sum(vals)),
        "avg": float(sum(vals) / float(len(vals))),
        "min": float(vals[0]),
        "max": float(vals[-1]),
        "p50": float(percentile(0.50)),
        "p90": float(percentile(0.90)),
    }


def sum_path(chunks: list[dict[str, Any]], *path: str) -> float:
    total = 0.0
    for chunk in chunks:
        value: Any = chunk
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if isinstance(value, (int, float)):
            total += float(value)
    return total


def int_sum_path(chunks: list[dict[str, Any]], *path: str) -> int:
    return int(round(sum_path(chunks, *path)))


def merge_counters(chunks: list[dict[str, Any]], *path: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for chunk in chunks:
        value: Any = chunk
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if isinstance(value, dict):
            counter.update({str(key): int(val) for key, val in value.items()})
    return dict(sorted(counter.items()))


def weighted_metric(chunks: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    numerator = 0.0
    count = 0
    score_numerator = 0.0
    score_count = 0
    page_numerator = 0.0
    page_count = 0
    lower_is_better = None
    for chunk in chunks:
        metric = (chunk.get("omnidocbench_metrics_without_cdm") or {}).get(metric_name)
        if not isinstance(metric, dict):
            continue
        metric_count = int(metric.get("count") or 0)
        avg = metric.get("avg")
        if avg is not None and metric_count:
            numerator += float(avg) * metric_count
            count += metric_count
        score = metric.get("score")
        if score is not None and metric_count:
            score_numerator += float(score) * metric_count
            score_count += metric_count
        page_avg = metric.get("page_avg")
        if page_avg is not None:
            pages = int(metric.get("page_avg_count") or chunk.get("page_count") or 0)
            if pages:
                page_numerator += float(page_avg) * pages
                page_count += pages
        if metric.get("lower_is_better") is not None:
            lower_is_better = bool(metric.get("lower_is_better"))
    avg = numerator / count if count else None
    score = score_numerator / score_count if score_count else None
    return {
        "avg": avg,
        "count": count,
        "lower_is_better": lower_is_better,
        "page_avg": page_numerator / page_count if page_count else None,
        "page_avg_count": int(page_count),
        "score": score,
        "score_percent": None if score is None else float(score) * 100.0,
    }


def aggregate_child_summary(chunks: list[dict[str, Any]], section_name: str, metric_name: str) -> dict[str, Any]:
    total_count = 0
    total_sum = 0.0
    mins: list[float] = []
    maxes: list[float] = []
    for chunk in chunks:
        metric = (chunk.get(section_name) or {}).get(metric_name)
        if not isinstance(metric, dict):
            continue
        count = int(metric.get("count") or 0)
        total = metric.get("sum")
        if total is not None and count:
            total_count += count
            total_sum += float(total)
        if metric.get("min") is not None:
            mins.append(float(metric["min"]))
        if metric.get("max") is not None:
            maxes.append(float(metric["max"]))
    return {
        "count": int(total_count),
        "sum": float(total_sum),
        "avg": (total_sum / float(total_count)) if total_count else None,
        "min": min(mins) if mins else None,
        "max": max(maxes) if maxes else None,
        "p50": None,
        "p90": None,
        "aggregation_note": "count/sum/avg/min/max are exact from child summaries; global p50/p90 are not reconstructed.",
    }


def aggregate_metrics(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "text_block_Edit_dist",
        "text_diagnostic_Edit_dist_including_title_code",
        "display_formula_Edit_dist",
        "display_formula_BLEU_1_4",
        "table_Edit_dist",
        "table_TEDS",
        "table_TEDS_structure_only",
        "reading_order_Edit_dist",
    ]
    metrics = {name: weighted_metric(chunks, name) for name in metric_names}
    components: list[dict[str, Any]] = []
    text_score = metrics["text_block_Edit_dist"].get("score")
    if text_score is not None:
        components.append(
            {
                "name": "text_block_Edit_dist",
                "score": float(text_score),
                "score_percent": float(text_score) * 100.0,
                "count": int(metrics["text_block_Edit_dist"].get("count") or 0),
            }
        )
    table_score = metrics["table_TEDS"].get("score")
    if table_score is not None:
        components.append(
            {
                "name": "table_TEDS",
                "score": float(table_score),
                "score_percent": float(table_score) * 100.0,
                "count": int(metrics["table_TEDS"].get("count") or 0),
            }
        )
    conclusion = None
    if components:
        conclusion = sum(float(component["score_percent"]) for component in components) / float(len(components))
    metrics.update(
        {
            "enabled": any(bool((chunk.get("omnidocbench_metrics_without_cdm") or {}).get("enabled")) for chunk in chunks),
            "is_official_omnidocbench_metric": False,
            "scope": "chunked_gt_crop_local_metrics_without_cdm",
            "matched_scored_items": int_sum_path(chunks, "omnidocbench_metrics_without_cdm", "matched_scored_items"),
            "leaderboard_overall": None,
            "leaderboard_overall_unavailable_reason": "CDM and official MGAM/TEDS evaluator are intentionally not run.",
            "text_table_conclusion_components": components,
            "text_table_conclusion_mean_score_percent": conclusion,
            "available_non_cdm_component_mean_score_percent": conclusion,
            "available_non_cdm_component_mean_note": (
                "Chunked aggregate. Formula edit/BLEU are diagnostics only because PaddleOCR-VL reports Formula CDM; "
                "this mean includes text Edit_dist score and table TEDS when available."
            ),
        }
    )
    return metrics


def aggregate_rough_accuracy(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    matched = int_sum_path(chunks, "rough_ground_truth_accuracy", "matched_text_items")
    exact = int_sum_path(chunks, "rough_ground_truth_accuracy", "normalized_exact_count")
    seq_num = 0.0
    by_label: dict[str, dict[str, float]] = {}
    for chunk in chunks:
        rough = chunk.get("rough_ground_truth_accuracy") or {}
        count = int(rough.get("matched_text_items") or 0)
        avg = rough.get("avg_sequence_ratio")
        if avg is not None and count:
            seq_num += float(avg) * count
        for label, row in (rough.get("by_layout_label") or {}).items():
            label_count = int(row.get("count") or 0)
            if not label_count:
                continue
            target = by_label.setdefault(str(label), {"count": 0.0, "exact": 0.0, "seq": 0.0})
            target["count"] += float(label_count)
            exact_rate = row.get("normalized_exact_rate")
            seq = row.get("avg_sequence_ratio")
            if exact_rate is not None:
                target["exact"] += float(exact_rate) * label_count
            if seq is not None:
                target["seq"] += float(seq) * label_count
    return {
        "enabled": any(bool((chunk.get("rough_ground_truth_accuracy") or {}).get("enabled")) for chunk in chunks),
        "is_official_omnidocbench_metric": False,
        "scope": "chunked_rough_gt_crop_text_similarity",
        "matched_text_items": matched,
        "normalized_exact_count": exact,
        "normalized_exact_rate": float(exact) / float(matched) if matched else None,
        "avg_sequence_ratio": seq_num / float(matched) if matched else None,
        "by_layout_label": {
            label: {
                "count": int(row["count"]),
                "normalized_exact_rate": row["exact"] / row["count"] if row["count"] else None,
                "avg_sequence_ratio": row["seq"] / row["count"] if row["count"] else None,
            }
            for label, row in sorted(by_label.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--page-start", type=int, default=0)
    parser.add_argument("--num-pages", type=int, default=64)
    parser.add_argument("--page-chunk-size", type=int, default=4)
    parser.add_argument("--child-output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--layout-source", default="omnidocbench_gt", choices=["omnidocbench_gt", "official", "cache"])
    parser.add_argument("--layout-device", default=None)
    parser.add_argument("--layout-model-name", default="PP-DocLayoutV3")
    parser.add_argument("--layout-model-dir", type=Path, default=None)
    parser.add_argument("--layout-threshold", type=float, default=None)
    parser.add_argument("--layout-nms", choices=["true", "false"], default=None)
    parser.add_argument("--layout-unclip-ratio", type=float, default=None)
    parser.add_argument("--layout-merge-bboxes-mode", default=None)
    parser.add_argument("--reuse-layout-cache", action="store_true")
    parser.add_argument("--crop-padding", type=int, default=0)
    parser.add_argument("--min-crop-side", type=int, default=4)
    parser.add_argument("--skip-labels", default="")
    parser.add_argument("--include-ignored-gt", action="store_true")
    parser.add_argument("--include-empty-gt", action="store_true")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--cache-length", type=int, default=2048)
    parser.add_argument("--preprocessor-min-pixels", type=int, default=-1)
    parser.add_argument("--preprocessor-max-pixels", type=int, default=-1)
    parser.add_argument("--active-batch-size", type=int, default=8)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--decode-backend", default="torchair")
    parser.add_argument("--decode-schedule", default="hotswap")
    parser.add_argument("--eos-mode", default="overlap_event_flags")
    parser.add_argument("--npu-jit-compile", default="off")
    parser.add_argument("--torchair-cache-dir", type=Path, default=None)
    parser.add_argument("--validation-items", type=int, default=-1)
    parser.add_argument("--rough-gt-min-iou", type=float, default=0.5)
    parser.add_argument("--expect-layout-source", default="", choices=["", "omnidocbench_gt", "official"])
    parser.add_argument("--expected-recognizer-crops", type=int, default=-1)
    parser.add_argument("--min-recognizer-crops", type=int, default=-1)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    parser.add_argument("--fail-on-length-cap", action="store_true")
    return parser.parse_args()


def child_command(args: argparse.Namespace, *, chunk_start: int, chunk_pages: int, child_json: Path) -> list[str]:
    layout_cache = child_json.with_suffix(".layout_cache.json")
    cmd = [
        str(args.python_bin),
        str(BENCH_SCRIPT),
        "--model",
        str(args.model),
        "--dataset-dir",
        str(args.dataset_dir),
        "--page-start",
        str(chunk_start),
        "--num-pages",
        str(chunk_pages),
        "--layout-source",
        str(args.layout_source),
        "--layout-cache-json",
        str(layout_cache),
        "--device",
        str(args.device),
        "--dtype",
        str(args.dtype),
        "--decode-backend",
        str(args.decode_backend),
        "--active-batch-size",
        str(args.active_batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--cache-length",
        str(args.cache_length),
        "--preprocessor-min-pixels",
        str(args.preprocessor_min_pixels),
        "--preprocessor-max-pixels",
        str(args.preprocessor_max_pixels),
        "--npu-jit-compile",
        str(args.npu_jit_compile),
        "--validation-items",
        str(args.validation_items),
        "--rough-gt-min-iou",
        str(args.rough_gt_min_iou),
        "--json",
    ]
    if args.layout_device is not None:
        cmd += ["--layout-device", str(args.layout_device)]
    if args.layout_model_name:
        cmd += ["--layout-model-name", str(args.layout_model_name)]
    if args.layout_model_dir is not None:
        cmd += ["--layout-model-dir", str(args.layout_model_dir)]
    if args.layout_threshold is not None:
        cmd += ["--layout-threshold", str(args.layout_threshold)]
    if args.layout_nms is not None:
        cmd += ["--layout-nms", str(args.layout_nms)]
    if args.layout_unclip_ratio is not None:
        cmd += ["--layout-unclip-ratio", str(args.layout_unclip_ratio)]
    if args.layout_merge_bboxes_mode is not None:
        cmd += ["--layout-merge-bboxes-mode", str(args.layout_merge_bboxes_mode)]
    if args.reuse_layout_cache:
        cmd += ["--reuse-layout-cache"]
    if int(args.crop_padding) != 0:
        cmd += ["--crop-padding", str(args.crop_padding)]
    if int(args.min_crop_side) != 4:
        cmd += ["--min-crop-side", str(args.min_crop_side)]
    if args.skip_labels:
        cmd += ["--skip-labels", str(args.skip_labels)]
    if args.include_ignored_gt:
        cmd += ["--include-ignored-gt"]
    if args.include_empty_gt:
        cmd += ["--include-empty-gt"]
    if args.prompt is not None:
        cmd += ["--prompt", str(args.prompt)]
    if args.decode_schedule:
        cmd += ["--decode-schedule", str(args.decode_schedule)]
    if args.eos_mode:
        cmd += ["--eos-mode", str(args.eos_mode)]
    if args.torchair_cache_dir is not None:
        cmd += ["--torchair-cache-dir", str(args.torchair_cache_dir)]
    if args.expect_layout_source:
        cmd += ["--expect-layout-source", str(args.expect_layout_source)]
    if int(args.min_recognizer_crops) >= 0:
        # Apply only as a weak per-chunk guard; exact total crop contracts are checked after aggregation.
        cmd += ["--min-recognizer-crops", "1"]
    return cmd


def build_aggregate(args: argparse.Namespace, chunks: list[dict[str, Any]], child_paths: list[Path]) -> dict[str, Any]:
    page_count = int_sum_path(chunks, "page_count")
    crop_count = int_sum_path(chunks, "recognizer_crop_count")
    decode_calls = int_sum_path(chunks, "decode_summary", "decode_calls")
    raw_decode = int_sum_path(chunks, "decode_summary", "raw_decode_token_calls")
    effective_decode = int_sum_path(chunks, "decode_summary", "effective_decode_token_calls")
    generated = int_sum_path(chunks, "decode_summary", "generated_tokens_including_prefill_first")
    output_fingerprints: list[dict[str, Any]] = []
    for chunk in chunks:
        output_fingerprints.extend(chunk.get("output_fingerprints") or [])
    mismatch = int_sum_path(chunks, "correctness", "mismatch_count")
    invalid = int_sum_path(chunks, "correctness", "invalid_token_count")
    length_caps = int_sum_path(chunks, "correctness", "length_cap_hit_count")
    validated = int_sum_path(chunks, "correctness", "validated_items")
    validation_enabled_chunks = sum(1 for chunk in chunks if bool((chunk.get("correctness") or {}).get("enabled")))
    validation_complete = bool(crop_count > 0 and validated == crop_count and validation_enabled_chunks == len(chunks))
    phase = {
        key: sum_path(chunks, "phase_timing_s", key)
        for key in [
            "layout_detection",
            "crop_extract",
            "recognizer_cpu_input_build",
            "recognizer_ready_bank_build",
            "hotswap_external_overlap_buffer_setup",
            "text_decode_queue",
            "decode_output_postprocess",
            "crop_chunk_cleanup",
            "measured_e2e_page_pipeline_excluding_setup_and_validation",
            "validation",
        ]
    }
    token_counts: list[float] = []
    for chunk in chunks:
        counts = (chunk.get("decode_summary") or {}).get("trimmed_new_token_counts") or []
        token_counts.extend(float(value) for value in counts)
    expected_crop = None if int(args.expected_recognizer_crops) < 0 else int(args.expected_recognizer_crops)
    min_crop = None if int(args.min_recognizer_crops) < 0 else int(args.min_recognizer_crops)
    child_layout_sources = sorted(
        {
            str((chunk.get("layout") or {}).get("source"))
            for chunk in chunks
            if (chunk.get("layout") or {}).get("source") is not None
        }
    )
    actual_layout_source = child_layout_sources[0] if len(child_layout_sources) == 1 else "mixed"
    child_cache_modes = sorted(
        {
            str((chunk.get("layout") or {}).get("cache_mode"))
            for chunk in chunks
            if (chunk.get("layout") or {}).get("cache_mode") is not None
        }
    )
    child_layout_inference_measured = sorted(
        {
            str(bool((chunk.get("layout") or {}).get("inference_measured")))
            for chunk in chunks
            if (chunk.get("layout") or {}).get("inference_measured") is not None
        }
    )
    contract_passed = True
    if args.expect_layout_source:
        contract_passed = contract_passed and all(
            str((chunk.get("layout") or {}).get("source")) == str(args.expect_layout_source) for chunk in chunks
        )
    if expected_crop is not None:
        contract_passed = contract_passed and crop_count == expected_crop
    if min_crop is not None:
        contract_passed = contract_passed and crop_count >= min_crop
    required_without_caps = bool(contract_passed and validation_complete and mismatch == 0 and invalid == 0)
    required_with_caps = bool(required_without_caps and length_caps == 0)
    measured = phase["measured_e2e_page_pipeline_excluding_setup_and_validation"]
    decode_s = phase["text_decode_queue"]
    ready_s = phase["recognizer_ready_bank_build"]
    page_load_escaped = int_sum_path(chunks, "page_load", "escaped_path_fallback_count")
    return {
        "experiment": "06_full_page_pipeline_e2e_chunked",
        "page_pipeline_scope": (
            "chunked full pages -> selected layout source -> crops -> per-chunk ready bank -> "
            "hotswap batched text decode -> aggregate JSON"
        ),
        "chunked_execution": True,
        "chunking": {
            "page_chunk_size": int(args.page_chunk_size),
            "chunk_count": int(len(chunks)),
            "child_output_dir": str(args.child_output_dir),
            "child_jsons": [str(path) for path in child_paths],
            "notes": (
                "Each child run keeps only that chunk's ready-bank KV cache on device. "
                "Aggregate measured_e2e sums the child timing windows and excludes each child setup/validation, "
                "matching the single-run timing convention while avoiding all-crops-resident VRAM pressure."
            ),
        },
        "model": str(args.model),
        "dataset_dir": str(args.dataset_dir),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "npu_jit_compile": str(args.npu_jit_compile),
        "page_start": int(args.page_start),
        "page_count": page_count,
        "page_load": {
            "requested_page_start": int(args.page_start),
            "requested_num_pages": int(args.num_pages),
            "loaded_page_count": int(page_count),
            "missing_image_policy": "fatal",
            "escaped_path_fallback_count": int(page_load_escaped),
            "aggregation_note": (
                "Counts are summed from child page chunks. Missing page images are fatal, not skipped."
            ),
        },
        "recognizer_crop_count": crop_count,
        "crop_count_contract": {
            "expect_layout_source": str(args.expect_layout_source),
            "expected_recognizer_crops": expected_crop,
            "min_recognizer_crops": min_crop,
            "actual_layout_source": str(actual_layout_source),
            "actual_recognizer_crops": crop_count,
            "passed": bool(contract_passed),
        },
        "uses_ground_truth_layout_boxes": bool(
            chunks and all(bool(chunk.get("uses_ground_truth_layout_boxes")) for chunk in chunks)
        ),
        "doc_layout_model_measured": bool(
            chunks and any(bool((chunk.get("layout") or {}).get("inference_measured")) for chunk in chunks)
        ),
        "active_batch_size": int(args.active_batch_size),
        "prefill_batch_size": 1,
        "decode_schedule": str(args.decode_schedule),
        "decode_backend": str(args.decode_backend),
        "eos_mode": str(args.eos_mode),
        "max_new_tokens": int(args.max_new_tokens),
        "cache_length": int(args.cache_length),
        "preprocessor": {
            "min_pixels": int(args.preprocessor_min_pixels) if int(args.preprocessor_min_pixels) >= 0 else (
                (chunks[0].get("preprocessor") or {}).get("min_pixels") if chunks else None
            ),
            "max_pixels": int(args.preprocessor_max_pixels) if int(args.preprocessor_max_pixels) >= 0 else (
                (chunks[0].get("preprocessor") or {}).get("max_pixels") if chunks else None
            ),
            "children": [chunk.get("preprocessor") for chunk in chunks],
        },
        "layout": {
            "source": str(actual_layout_source),
            "child_sources": child_layout_sources,
            "child_cache_modes": child_cache_modes,
            "child_inference_measured": child_layout_inference_measured,
            "inference_measured": bool(
                chunks and any(bool((chunk.get("layout") or {}).get("inference_measured")) for chunk in chunks)
            ),
            "requested_layout_source": str(args.layout_source),
            "device": args.layout_device,
            "include_ignored_gt": bool(args.include_ignored_gt),
            "include_empty_gt": bool(args.include_empty_gt),
        },
        "setup_timing_s_chunk_sum": {
            "recognizer_model_load_s": sum_path(chunks, "setup_timing_s", "recognizer_model_load_s"),
            "decode_weight_format_s": sum_path(chunks, "setup_timing_s", "decode_weight_format_s"),
            "compile_wrapper_s": sum_path(chunks, "setup_timing_s", "compile_wrapper_s"),
            "compile_first_call_s": sum_path(chunks, "setup_timing_s", "compile_first_call_s"),
        },
        "phase_timing_s": phase,
        "throughput": {
            "pages_per_s_measured_e2e": tok_per_s(page_count, measured),
            "seconds_per_page_measured_e2e": measured / float(page_count) if page_count else None,
            "crops_per_s_measured_e2e": tok_per_s(crop_count, measured),
            "prefill_crops_per_s": tok_per_s(crop_count, ready_s),
            "decode_crops_per_s": tok_per_s(crop_count, decode_s),
            "decode_calls_per_s": tok_per_s(decode_calls, decode_s),
            "raw_decode_token_calls_per_s": tok_per_s(raw_decode, decode_s),
            "effective_decode_tokens_per_s": tok_per_s(effective_decode, decode_s),
        },
        "crop_summary": {
            "recognizer_crop_count": crop_count,
            "layout_box_count": int_sum_path(chunks, "crop_summary", "layout_box_count"),
            "skipped_count": int_sum_path(chunks, "crop_summary", "skipped_count"),
            "label_counts": merge_counters(chunks, "crop_summary", "label_counts"),
            "prompt_counts": merge_counters(chunks, "crop_summary", "prompt_counts"),
        },
        "ready_item_timing_summary_s": {
            key: aggregate_child_summary(chunks, "ready_item_timing_summary_s", key)
            for key in [
                "native_resolution_visual_encoder_total",
                "vision_encoder",
                "adaptive_mlp_projector",
                "vision_total",
                "vision_projector_total",
                "text_prefill",
                "prefill_lm_head",
                "prefill_argmax",
                "ready_item_total_excluding_device_transfer",
                "ready_item_total_with_device_transfer",
            ]
        },
        "decode_summary": {
            "decode_calls": decode_calls,
            "raw_decode_token_calls": raw_decode,
            "effective_decode_token_calls": effective_decode,
            "generated_tokens_including_prefill_first": generated,
            "swap_event_count": int_sum_path(chunks, "decode_summary", "swap_event_count"),
            "total_swapped_in_items": int_sum_path(chunks, "decode_summary", "total_swapped_in_items"),
            "eos_hit_count": int_sum_path(chunks, "decode_summary", "eos_hit_count"),
            "length_cap_hit_count": length_caps,
            "trimmed_new_tokens": stat_summary(token_counts),
        },
        "correctness": {
            "all_required_checks_passed": required_with_caps,
            "required_checks_excluding_length_caps_passed": required_without_caps,
            "length_cap_is_required_failure": False,
            "mismatch_count": mismatch,
            "invalid_token_count": invalid,
            "length_cap_hit_count": length_caps,
            "validated_items": validated,
            "validation_complete": validation_complete,
            "validation_enabled_chunks": int(validation_enabled_chunks),
            "validation_required_for_pass": True,
            "ground_truth_checked": False,
            "scope": "chunked_queue_output_vs_same_local_static_model",
        },
        "omnidocbench_metrics_without_cdm": aggregate_metrics(chunks),
        "rough_ground_truth_accuracy": aggregate_rough_accuracy(chunks),
        "output_fingerprint_summary": {
            "item_count": int(len(output_fingerprints)),
            "fingerprints_sha256": None,
            "generated_texts_sha256": None,
            "token_ids_sha256": None,
            "note": "Chunk aggregate concatenates child output_fingerprints in page/chunk order.",
        },
        "output_fingerprints": output_fingerprints,
        "chunks": [
            {
                "page_start": int(chunk.get("page_start", 0)),
                "page_count": int(chunk.get("page_count", 0)),
                "page_load": chunk.get("page_load", {}),
                "recognizer_crop_count": int(chunk.get("recognizer_crop_count", 0)),
                "json": str(path),
                "decode_summary": chunk.get("decode_summary", {}),
                "correctness": chunk.get("correctness", {}),
                "phase_timing_s": chunk.get("phase_timing_s", {}),
                "crop_label_counts": (chunk.get("crop_summary") or {}).get("label_counts", {}),
            }
            for chunk, path in zip(chunks, child_paths)
        ],
    }


def main() -> None:
    args = parse_args()
    if int(args.page_chunk_size) <= 0:
        raise ValueError("--page-chunk-size must be positive")
    if int(args.num_pages) <= 0:
        raise ValueError("--num-pages must be positive")
    args.child_output_dir.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    chunks: list[dict[str, Any]] = []
    child_paths: list[Path] = []
    for offset in range(0, int(args.num_pages), int(args.page_chunk_size)):
        chunk_start = int(args.page_start) + offset
        chunk_pages = min(int(args.page_chunk_size), int(args.num_pages) - offset)
        child_json = args.child_output_dir / f"chunk_start_{chunk_start:05d}_pages_{chunk_pages}.json"
        cmd = child_command(args, chunk_start=chunk_start, chunk_pages=chunk_pages, child_json=child_json)
        print(f"CHUNK_RUN start={chunk_start} pages={chunk_pages} output={child_json}", flush=True)
        print("CHUNK_COMMAND " + " ".join(cmd), flush=True)
        start = time.perf_counter()
        with child_json.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.PIPE, text=True)
        elapsed = time.perf_counter() - start
        if proc.returncode != 0:
            print(f"CHUNK_FAILED start={chunk_start} pages={chunk_pages} elapsed_s={elapsed:.3f}", file=sys.stderr)
            print(proc.stderr[-8000:], file=sys.stderr)
            raise SystemExit(proc.returncode)
        chunk = json.loads(child_json.read_text(encoding="utf-8"))
        if chunk.get("error"):
            print(f"CHUNK_ERROR start={chunk_start} payload={json.dumps(chunk, sort_keys=True)[:8000]}", file=sys.stderr)
            raise SystemExit(2)
        if int(chunk.get("page_count") or -1) != int(chunk_pages):
            raise SystemExit(
                f"chunk loaded {chunk.get('page_count')} pages but requested {chunk_pages}; "
                "missing page images and out-of-range page slices must be fixed, not skipped"
            )
        chunks.append(chunk)
        child_paths.append(child_json)
        decode = chunk.get("decode_summary") or {}
        correctness = chunk.get("correctness") or {}
        print(
            "CHUNK_DONE "
            + json.dumps(
                {
                    "page_start": chunk_start,
                    "page_count": chunk.get("page_count"),
                    "page_load": chunk.get("page_load"),
                    "recognizer_crop_count": chunk.get("recognizer_crop_count"),
                    "effective_decode_token_calls": decode.get("effective_decode_token_calls"),
                    "generated_tokens_including_prefill_first": decode.get(
                        "generated_tokens_including_prefill_first"
                    ),
                    "mismatch_count": correctness.get("mismatch_count"),
                    "invalid_token_count": correctness.get("invalid_token_count"),
                    "length_cap_hit_count": correctness.get("length_cap_hit_count"),
                    "elapsed_outer_s": elapsed,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    aggregate = build_aggregate(args, chunks, child_paths)
    args.output_json.write_text(json.dumps(aggregate, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    print("CHUNKED_OUTPUT_JSON", args.output_json, flush=True)
    print(
        "CHUNKED_SUMMARY "
        + json.dumps(
            {
                "page_count": aggregate["page_count"],
                "page_load": aggregate.get("page_load"),
                "recognizer_crop_count": aggregate["recognizer_crop_count"],
                "crop_count_contract": aggregate["crop_count_contract"],
                "effective_decode_token_calls": aggregate["decode_summary"]["effective_decode_token_calls"],
                "generated_tokens_including_prefill_first": aggregate["decode_summary"][
                    "generated_tokens_including_prefill_first"
                ],
                "raw_decode_token_calls": aggregate["decode_summary"]["raw_decode_token_calls"],
                "decode_calls": aggregate["decode_summary"]["decode_calls"],
                "mismatch_count": aggregate["correctness"]["mismatch_count"],
                "invalid_token_count": aggregate["correctness"]["invalid_token_count"],
                "length_cap_hit_count": aggregate["correctness"]["length_cap_hit_count"],
                "required_checks_excluding_length_caps_passed": aggregate["correctness"][
                    "required_checks_excluding_length_caps_passed"
                ],
                "pages_per_s_measured_e2e": aggregate["throughput"]["pages_per_s_measured_e2e"],
                "effective_decode_tokens_per_s": aggregate["throughput"]["effective_decode_tokens_per_s"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if not aggregate["crop_count_contract"]["passed"]:
        raise SystemExit("crop count/layout source contract failed")
    if args.fail_on_mismatch and not aggregate["correctness"]["required_checks_excluding_length_caps_passed"]:
        raise SystemExit("chunked correctness failed: mismatch_count or invalid_token_count is nonzero")
    if args.fail_on_length_cap and int(aggregate["correctness"]["length_cap_hit_count"]) > 0:
        raise SystemExit("chunked correctness failed: length_cap_hit_count is nonzero")


if __name__ == "__main__":
    main()
