#!/usr/bin/env python3
"""Serve the validated height-routed U8 table-speculation lane over HTTP."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import io
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import queue
import signal
from types import SimpleNamespace
from typing import Any
import sys
import threading
import time
import traceback


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
DEFAULT_TARGETS = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/table_spec_full_d1e6d00/"
    "whole/whole/tables.jsonl"
)
DEFAULT_COMPACT_VOCAB = (
    EXPERIMENT_ROOT
    / "presets/table_compact_vocab/b1_verifier_topfreq_16384.json"
)
sys.path.insert(0, str(HERE))

from serve_crop_ocr_api import (  # noqa: E402
    _Handler,
    _Server,
    _State,
    _write_service_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--max-image-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--queue-capacity", type=int, default=256)
    parser.add_argument(
        "--interleaved-tables", type=int, choices=(1, 2, 4), default=None,
        help="Opt-in step-level reference with independent table slots; client controls in-flight requests.",
    )
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
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
    parser.add_argument(
        "--decode-vocab-token-ids",
        type=Path,
        default=DEFAULT_COMPACT_VOCAB,
    )
    parser.add_argument("--height-threshold-px", type=int, default=384)
    parser.add_argument("--experimental-oracle-token-counts", type=Path,
                        help="Benchmark-only frozen ordinary-B1 output-length map; never output token IDs.")
    parser.add_argument("--oracle-min-output-tokens", type=int,
                        help="Speculate only at/above this oracle count AND the existing height threshold.")
    parser.add_argument("--cold-request-id", default="page_000010_table_box_id_1")
    parser.add_argument("--draft-cache-length", type=int, default=768)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--decode-optimization",
        default="combined_apply_complete_layer_prefetch1_rope_lut",
    )
    parser.add_argument(
        "--verifier-optimization",
        default="combined_apply_spec_prefetch_mrope",
    )
    parser.add_argument("--k-values", default="7,15,31,63")
    parser.add_argument("--initial-k", type=int, default=15)
    parser.add_argument(
        "--allow-compile",
        action="store_true",
        help=(
            "Compile missing production graphs during server startup. "
            "Compilation finishes before the ready endpoint becomes available."
        ),
    )
    parser.add_argument("--min-pixels", type=int, default=28224)
    parser.add_argument("--max-pixels", type=int, default=802816)
    parser.add_argument(
        "--decode-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_torchair",
    )
    parser.add_argument(
        "--vision-cache-dir",
        type=Path,
        default=(
            REPO_ROOT
            / ".runtime_cache/09_persistent_page_engine_vision_torchair"
        ),
    )
    parser.add_argument(
        "--text-cache-dir",
        type=Path,
        default=(
            REPO_ROOT
            / ".runtime_cache/09_persistent_page_engine_text_torchair"
        ),
    )
    parser.add_argument(
        "--text-packed-cache-dir",
        type=Path,
        default=(
            REPO_ROOT
            / ".runtime_cache/09_persistent_page_engine_text_packed_torchair"
        ),
    )
    parser.add_argument(
        "--service-summary-output",
        type=Path,
        help=(
            "Write final draft, prefill, and verifier metrics when the "
            "server drains during shutdown."
        ),
    )
    args = parser.parse_args()
    if (args.experimental_oracle_token_counts is None) != (args.oracle_min_output_tokens is None):
        parser.error("oracle count map and minimum token count must be supplied together")
    if args.oracle_min_output_tokens is not None:
        if args.interleaved_tables is None or args.oracle_min_output_tokens < 0:
            parser.error("oracle routing requires --interleaved-tables and a nonnegative threshold")
    return args


def _oracle_counts(path: Path) -> tuple[dict[str, int], str]:
    payload = path.read_bytes()
    document = json.loads(payload)
    if document.get("format") != "ordinary_b1_output_length_oracle_v1":
        raise ValueError("unsupported oracle map format")
    counts = document.get("token_counts")
    if not isinstance(counts, dict) or not counts:
        raise ValueError("oracle map must contain nonempty token_counts")
    if any(not isinstance(k, str) or type(v) is not int or v < 1 for k, v in counts.items()):
        raise ValueError("oracle lengths must be positive integer native-token counts")
    return counts, hashlib.sha256(payload).hexdigest()


def _table_route(height: int, height_threshold: int, source_id: str,
                 counts: dict[str, int] | None, minimum: int | None) -> tuple[str, dict[str, Any]]:
    count = None if counts is None else counts[source_id]  # Missing IDs fail, never silently fall back.
    eligible = height >= height_threshold
    selected = eligible and (minimum is None or count >= minimum)
    return ("spec" if selected else "b1"), {
        "height_px": height, "height_eligible": eligible,
        "oracle_b1_output_tokens": count, "oracle_min_output_tokens": minimum,
        "reason": "below_height" if not eligible else (
            "below_oracle_token_threshold" if not selected else "spec_eligible"),
    }


_SCHEDULE_COUNT_FIELDS = (
    "requests",
    "graph_calls",
    "initial_admissions",
    "hot_swap_admissions",
    "prefill_only_completions",
    "raw_decode_token_slots",
    "active_decode_token_slots",
    "effective_decode_tokens",
    "idle_decode_token_slots",
    "lookahead_decode_token_slots",
    "kv_prefix_bytes_copied",
    "initial_kv_prefix_bytes_copied",
    "hot_swap_kv_prefix_bytes_copied",
)


def _sum_numeric_maps(values: list[dict[str, Any]]) -> dict[str, float]:
    keys = {str(key) for value in values for key in value}
    return {
        key: sum(
            float(value.get(key, 0.0))
            for value in values
            if isinstance(value.get(key, 0.0), (int, float))
        )
        for key in sorted(keys)
    }


def _merge_decode_schedules(
    schedules: list[dict[str, Any]],
) -> dict[str, Any]:
    if not schedules:
        return {}
    counts = {
        key: sum(int(schedule.get(key, 0)) for schedule in schedules)
        for key in _SCHEDULE_COUNT_FIELDS
    }
    timing = _sum_numeric_maps(
        [dict(schedule.get("timing_s") or {}) for schedule in schedules]
    )
    raw_slots = counts["raw_decode_token_slots"]
    active_slots = counts["active_decode_token_slots"]
    effective_tokens = counts["effective_decode_tokens"]
    decode_wall = float(timing.get("continuous_decode_wall", 0.0))
    device_wall = float(timing.get("decode_model_and_argmax_device", 0.0))
    scheduler_wall = float(timing.get("run_scoped_scheduler_wall", 0.0))

    def rate(numerator: int, denominator: float) -> float | None:
        return numerator / denominator if denominator > 0 else None

    return {
        "schedule_count": len(schedules),
        "batch_size": int(schedules[0].get("batch_size", 0)),
        **counts,
        "timing_s": timing,
        "rates": {
            "raw_decode_tok_per_s": rate(raw_slots, decode_wall),
            "effective_decode_tok_per_s": rate(effective_tokens, decode_wall),
            "effective_device_tok_per_s": rate(effective_tokens, device_wall),
            "scheduler_effective_tok_per_s": rate(
                effective_tokens, scheduler_wall
            ),
            "active_slot_fraction": (
                active_slots / raw_slots if raw_slots > 0 else None
            ),
            "idle_slot_fraction": (
                counts["idle_decode_token_slots"] / raw_slots
                if raw_slots > 0
                else None
            ),
        },
    }


def _spec_runtime_metrics(full: dict[str, Any]) -> dict[str, Any]:
    speculative = dict(full["speculative"])
    adaptive = dict(speculative.get("adaptive_k") or {})
    draft_rows = []
    for row in full["draft"]["rows"]:
        draft_rows.append(
            {
                "input_tokens": int(row["input_tokens"]),
                "projected_image_tokens": int(row["projected_image_tokens"]),
                "generated_tokens_including_eos": int(
                    row["generated_tokens_including_eos"]
                ),
                "decode_tokens_after_prefill_including_eos": int(
                    row["decode_tokens_after_prefill_including_eos"]
                ),
                "decode_calls_executed": int(row["decode_calls_executed"]),
                "timing_s": dict(row.get("timing_s") or {}),
                "device_stage_s": dict(row.get("device_stage_s") or {}),
                "vision": dict(row.get("vision") or {}),
                "text_prefill": dict(row.get("text_prefill") or {}),
            }
        )
    return {
        "draft": {
            "rows": draft_rows,
            "schedule": dict(full["draft"]["schedule"]),
        },
        "target_prefill": {
            "input_tokens": int(full["input_tokens"]),
            "projected_image_tokens": int(full["projected_image_tokens"]),
            **dict(full["target_prefill"]),
        },
        "verifier": {
            key: speculative.get(key)
            for key in (
                "target_calls",
                "speculative_calls",
                "fully_accepted_speculative_calls",
                "rejected_speculative_calls",
                "fallback_calls",
                "proposed_draft_tokens",
                "accepted_draft_tokens",
                "accepted_fraction_of_proposed",
                "verifier_device_s",
                "fallback_device_s",
                "wall_s",
                "effective_target_tok_per_s",
            )
        }
        | {
            "output_tokens_after_prefill": max(
                0, len(speculative["token_ids"]) - 1
            ),
            "per_k": dict(adaptive.get("per_k") or {}),
            "transitions": dict(adaptive.get("transitions") or {}),
        },
    }


def _summarize_spec_service(
    spec_metrics: list[dict[str, Any]],
    b1_schedules: list[dict[str, Any]],
) -> dict[str, Any]:
    draft_schedules = [item["draft"]["schedule"] for item in spec_metrics]
    verifier_rows = [item["verifier"] for item in spec_metrics]
    verifier_sum_fields = (
        "target_calls",
        "speculative_calls",
        "fully_accepted_speculative_calls",
        "rejected_speculative_calls",
        "fallback_calls",
        "proposed_draft_tokens",
        "accepted_draft_tokens",
        "output_tokens_after_prefill",
    )
    verifier = {
        key: sum(int(row.get(key) or 0) for row in verifier_rows)
        for key in verifier_sum_fields
    }
    verifier_device_s = sum(
        float(row.get("verifier_device_s") or 0.0) for row in verifier_rows
    )
    fallback_device_s = sum(
        float(row.get("fallback_device_s") or 0.0) for row in verifier_rows
    )
    verifier_wall_s = sum(
        float(row.get("wall_s") or 0.0) for row in verifier_rows
    )
    proposed = verifier["proposed_draft_tokens"]
    physical_verifier_tokens = 0
    per_k: dict[str, dict[str, float]] = {}
    for row in verifier_rows:
        for k_text, values in dict(row.get("per_k") or {}).items():
            target = per_k.setdefault(k_text, {})
            for key, value in dict(values).items():
                if isinstance(value, (int, float)):
                    target[key] = target.get(key, 0.0) + float(value)
            physical_verifier_tokens += int(values.get("calls", 0)) * (
                int(k_text) + 1
            )
    physical_verifier_tokens += verifier["fallback_calls"]
    verifier.update(
        {
            "verifier_device_s": verifier_device_s,
            "fallback_device_s": fallback_device_s,
            "wall_s": verifier_wall_s,
            "accepted_fraction_of_proposed": (
                verifier["accepted_draft_tokens"] / proposed
                if proposed > 0
                else None
            ),
            "physical_verifier_tokens": physical_verifier_tokens,
            "physical_verifier_tok_per_s": (
                physical_verifier_tokens / verifier_device_s
                if verifier_device_s > 0
                else None
            ),
            "physical_verifier_tok_per_verifier_wall_s": (
                physical_verifier_tokens / verifier_wall_s
                if verifier_wall_s > 0
                else None
            ),
            "effective_output_tok_per_s": (
                verifier["output_tokens_after_prefill"] / verifier_wall_s
                if verifier_wall_s > 0
                else None
            ),
            "per_k": per_k,
        }
    )
    return {
        "format": "paddleocr_table_spec_service_metrics_v1",
        "request_count": len(spec_metrics) + len(b1_schedules),
        "route_counts": {
            "spec": len(spec_metrics),
            "b1": len(b1_schedules),
        },
        "draft_decode": _merge_decode_schedules(draft_schedules),
        "verifier": verifier,
        "b1_decode": _merge_decode_schedules(b1_schedules),
    }


def _live_args(config: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        model=Path(config["model"]),
        b1_decode_optimization=config["decode_optimization"],
        b1_decode_vocab_token_ids=Path(config["decode_vocab_token_ids"]),
        b1_cache_length=config["cache_length"],
        b1_max_new_tokens=config["max_new_tokens"],
        b1_vision_buckets="4096",
        b1_text_buckets="1152",
        draft_decode_optimization=config["decode_optimization"],
        draft_decode_vocab_token_ids=Path(config["decode_vocab_token_ids"]),
        draft_cache_length=config["draft_cache_length"],
        draft_row_count=8,
        draft_batch_size=8,
        draft_vision_packing="greedy",
        draft_vision_pack_target=2304,
        draft_prefill_layout="packed_b1",
        draft_batched_vision_shapes="8x640,8x768",
        draft_batched_text_shape="8x256",
        row_overlap_px=3,
        compact_uint8_preprocess=False,
        image_resize_backend="pillow",
        target_cpu_delay_ms=0.0,
        overlap_target_cpu_preparation=False,
        k_values=config["k_values"],
        initial_k=config["initial_k"],
        verifier_optimization=config["verifier_optimization"],
        per_call_device_timing=False,
        allow_compile=bool(config["allow_compile"]),
        token_selection="greedy",
        min_pixels=config["min_pixels"],
        max_pixels=config["max_pixels"],
        vision_buckets="256,384,512,640,768,1408,1920,2048,2304,2944,4096",
        decode_cache_dir=Path(config["decode_cache_dir"]),
        vision_cache_dir=Path(config["vision_cache_dir"]),
        vision_batched_cache_dir=Path(config["vision_cache_dir"]),
        text_cache_dir=Path(config["text_cache_dir"]),
        text_packed_cache_dir=Path(config["text_packed_cache_dir"]),
        text_batched_cache_dir=Path(config["text_cache_dir"]),
        k_cache_root=[],
    )


def _worker_main(jobs: Any, results: Any, config: dict[str, Any]) -> None:
    try:
        sys.path.insert(0, str(EXPERIMENT_ROOT))
        from PIL import Image
        import torch
        import torch_npu  # noqa: F401
        import table_row_ocr_lab as row_lab
        import table_spec_adaptive_k_lab as adaptive_lab
        import table_spec_decode_lab as fixed_lab
        import table_spec_live_u8_adaptive_lab as live_lab
        from paddleocr_vl.model.text_spec_verify import (
            SPEC_VERIFY_ATTENTION,
            torchair_cache_dir_for_spec_shape,
        )
        from pipeline.layout_output import normalize_recognition_text

        torch.npu.config.allow_internal_format = True
        torch.npu.set_compile_mode(jit_compile=False)
        args = _live_args(config)
        if config.get("interleaved_tables") is not None and not args.allow_compile:
            raise ValueError("interleaved reference requires explicit --allow-compile for its startup graph matrix")
        targets = fixed_lab.read_jsonl(Path(config["targets"]))
        targets_by_id = {str(row["request_id"]): row for row in targets}

        print("TABLE_SPEC_API setup=draft_recognizer", flush=True)
        draft_recognizer = row_lab.build_recognizer(live_lab._draft_args(args))
        print("TABLE_SPEC_API setup=b1_recognizer", flush=True)
        b1_recognizer = fixed_lab.build_recognizer(live_lab._b1_args(args))
        if config.get("interleaved_tables") is not None:
            _interleaved_worker_loop(
                jobs, results, config, args, targets_by_id,
                b1_recognizer, draft_recognizer,
            )
            return
        k_values = adaptive_lab.parse_k_values(args.k_values)
        cache_roots = adaptive_lab.parse_k_cache_roots(
            [], k_values=k_values, default=args.decode_cache_dir
        )
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
            if (
                not args.allow_compile
                and (not spec_cache.is_dir() or not any(spec_cache.iterdir()))
            ):
                raise RuntimeError(f"missing K{value} verifier cache: {spec_cache}")
        runtime = adaptive_lab.AdaptiveKTableSpeculativeDecodeRuntime(
            b1_recognizer,
            k_values=k_values,
            initial_k=args.initial_k,
            cache_roots=cache_roots,
            verifier_optimization=args.verifier_optimization,
            record_device_timing=False,
        )
        cpu_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="table-spec-api-cpu-overlap",
        )

        def run_spec(source: dict[str, Any], raw_image: Any) -> dict[str, Any]:
            return live_lab._run_one(
                source=source,
                raw_image=raw_image,
                crop_preload_wall_s=0.0,
                measured=True,
                args=args,
                b1_recognizer=b1_recognizer,
                draft_recognizer=draft_recognizer,
                runtime=runtime,
                cpu_executor=cpu_executor,
            )

        def run_b1(source: dict[str, Any], target_crop: Any) -> dict[str, Any]:
            emitted: list[Any] = []
            request = fixed_lab.request_for(source, target_crop, args)
            schedule = b1_recognizer.run(
                [request],
                schedule_id=f"table-spec-api:b1:{source['request_id']}",
                emit_result=emitted.append,
            )
            if len(emitted) != 1:
                raise RuntimeError(f"B1 route emitted {len(emitted)} results")
            recognition = emitted[0]
            payload = asdict(recognition)
            payload["raw_text"] = payload["text"]
            payload["text"] = normalize_recognition_text(
                "table", payload["raw_text"]
            )
            payload["schedule"] = asdict(schedule)
            return payload

        cold_id = config["cold_request_id"]
        if cold_id:
            cold_source = targets_by_id[cold_id]
            cold_image = live_lab.load_crop(
                cold_source, Path(config["images_dir"])
            )
            print(f"TABLE_SPEC_API cold_start={cold_id}", flush=True)
            run_spec(cold_source, cold_image)

        results.put(
            {
                "kind": "ready",
                "configuration": {
                    "lane": "height_routed_u8_adaptive_k",
                    "height_threshold_px": config["height_threshold_px"],
                    "draft_rows": 8,
                    "adaptive_k": list(k_values),
                    "initial_k": args.initial_k,
                    "decode_optimization": args.b1_decode_optimization,
                    "verifier_optimization": args.verifier_optimization,
                    "spec_verify_attention": SPEC_VERIFY_ATTENTION,
                    "allow_compile": args.allow_compile,
                },
                "worker_pid": __import__("os").getpid(),
            }
        )

        spec_service_metrics: list[dict[str, Any]] = []
        b1_service_schedules: list[dict[str, Any]] = []

        while True:
            job = jobs.get()
            if job is None:
                results.put(
                    {
                        "kind": "service_summary",
                        "payload": _summarize_spec_service(
                            spec_service_metrics,
                            b1_service_schedules,
                        ),
                    }
                )
                break
            internal_id = str(job["request_id"])
            try:
                if job.get("crop_type") != "table":
                    raise ValueError("speculative server accepts only table crops")
                source_id = str(job["source_request_id"])
                source = targets_by_id[source_id]
                with Image.open(io.BytesIO(job["image_bytes"])) as opened:
                    raw_image = opened.convert("RGB")
                service_started = time.perf_counter()
                target_crop = live_lab._exact_target_crop_from_raw(
                    source, raw_image
                )
                if target_crop.height < config["height_threshold_px"]:
                    route_lane = "b1"
                    full = run_b1(source, target_crop)
                    raw_text = str(full["raw_text"])
                    text = str(full["text"])
                    token_ids = list(full["token_ids"])
                    stop_reason = str(full["stop_reason"])
                    exact = token_ids == fixed_lab.target_tokens(source)
                    stage_timing = {}
                    runtime_metrics = {
                        "b1": {
                            "schedule": dict(full["schedule"]),
                            "timing_s": dict(full.get("timing_s") or {}),
                            "device_stage_s": dict(
                                full.get("device_stage_s") or {}
                            ),
                            "vision": dict(full.get("vision") or {}),
                            "text_prefill": dict(
                                full.get("text_prefill") or {}
                            ),
                            "generated_tokens_including_eos": int(
                                full["generated_tokens_including_eos"]
                            ),
                            "decode_tokens_after_prefill_including_eos": int(
                                full[
                                    "decode_tokens_after_prefill_including_eos"
                                ]
                            ),
                            "decode_calls_executed": int(
                                full["decode_calls_executed"]
                            ),
                        }
                    }
                    b1_service_schedules.append(dict(full["schedule"]))
                else:
                    route_lane = "spec"
                    full = run_spec(source, raw_image)
                    speculative = full["speculative"]
                    raw_text = str(speculative["text"])
                    text = str(full["pred_html"])
                    token_ids = list(speculative["token_ids"])
                    stop_reason = str(speculative["stop_reason"])
                    exact = bool(full["exact_saved_reference"])
                    stage_timing = dict(full["timing_s"])
                    runtime_metrics = _spec_runtime_metrics(full)
                    spec_service_metrics.append(runtime_metrics)
                service_wall_s = time.perf_counter() - service_started
                results.put(
                    {
                        "kind": "result",
                        "request_id": internal_id,
                        "ok": True,
                        "payload": {
                            "request_id": source_id,
                            "crop_type": "table",
                            "route_lane": route_lane,
                            "raw_text": raw_text,
                            "text": text,
                            "token_ids": token_ids,
                            "output_tokens": len(token_ids),
                            "stop_reason": stop_reason,
                            "exact_saved_reference": exact,
                            "service_wall_s": service_wall_s,
                            "queue_wait_s": (
                                service_started
                                - float(job["submitted_monotonic_s"])
                            ),
                            "stage_timing_s": stage_timing,
                            "runtime_metrics": runtime_metrics,
                            "worker_wall_s": (
                                time.perf_counter()
                                - float(job["submitted_monotonic_s"])
                            ),
                        },
                    }
                )
            except BaseException as exc:
                results.put(
                    {
                        "kind": "result",
                        "request_id": internal_id,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
        cpu_executor.shutdown(wait=True)
    except BaseException as exc:
        results.put(
            {
                "kind": "startup_error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def _interleaved_worker_loop(
    jobs: Any, results: Any, config: dict[str, Any], args: Any,
    targets_by_id: dict[str, Any], b1: Any, draft: Any,
) -> None:
    """Reference composition only; the established C1 worker above is unchanged."""
    from PIL import Image
    import torch
    import table_spec_decode_lab as fixed_lab
    import table_spec_live_u8_adaptive_lab as live_lab
    from paddleocr_vl.serving.table_interleaved_runtime import InterleavedTableRuntime
    from paddleocr_vl.serving.table_phase_scheduler import PhaseLedger, TablePhasePolicy
    from pipeline.layout_output import normalize_recognition_text

    capacity = int(config["interleaved_tables"])
    counts = config.get("oracle_token_counts")
    oracle_metadata = {
        "enabled": counts is not None,
        "minimum_output_tokens": config.get("oracle_min_output_tokens"),
        "count_map_sha256": config.get("oracle_count_map_sha256"),
        "contract": "benchmark oracle only; position/companion invariant; no saved token IDs used",
    }
    with torch.inference_mode():
        runtime = InterleavedTableRuntime(b1, draft, args, capacity=capacity)
        policy, ledger = TablePhasePolicy(), PhaseLedger()
        matcher_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="table-phase-matcher")
        preparation_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="table-phase-prepare")
        preparing: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        draining = False
        completed = 0
        results.put({
            "kind": "ready", "worker_pid": __import__("os").getpid(),
            "configuration": {
                "lane": "interleaved_height_routed_u8_reference",
                "table_slots": capacity, "draft_rows": 8,
                "height_threshold_px": config["height_threshold_px"],
                "experimental_oracle_routing": oracle_metadata,
                "adaptive_k": list(runtime.k_values), "initial_k": args.initial_k,
                "mixed_execution": "separate_fairly_alternating_calls",
                "batching": "immediate_matching_phase_and_query_only",
                "kv_policy": "stable_request_slots_no_k_migration",
                "nonadjacent_batch_policy": "smallest_covering_power_of_two_with_dummy_prefix_preservation",
                "cpu_preparation": "one_background_worker_with_reserved_table_slots",
                "warmup": "client_must_send_complete_requests_before_measurement",
                "graph_contracts": runtime.metadata,
            },
        })
        print("TABLE_PHASE ready; perform complete client warmups outside measurement", flush=True)

        def snapshot(extra: str | None = None) -> dict[str, str]:
            phases = {key: value["phase"] for key, value in runtime.jobs.items()}
            phases.update({key: "cpu_prepare" for key in preparing})
            if extra is not None:
                phases[extra] = "cpu_prepare"
            return phases

        def account(action: str, owners: list[str], fn: Any, *, extra: str | None = None) -> Any:
            phases = snapshot(extra)
            started = time.perf_counter()
            try:
                return fn()
            finally:
                ledger.record(action, owners=owners, phases=phases, wall_s=time.perf_counter() - started)

        def prepare(job: dict[str, Any]) -> dict[str, Any]:
            started = time.perf_counter()
            cpu_started = time.thread_time()
            key = str(job["request_id"])
            source_id = str(job["source_request_id"])
            if job.get("crop_type") != "table":
                raise ValueError("speculative server accepts only table crops")
            source = dict(targets_by_id[source_id])
            # Keep source identity for the existing orientation/preprocessing
            # contract. HTTP identity is applied only to prepared work items.
            with Image.open(io.BytesIO(job["image_bytes"])) as opened:
                raw = opened.convert("RGB")
            crop = live_lab._exact_target_crop_from_raw(source, raw)
            route, routing = _table_route(crop.height, config["height_threshold_px"], source_id,
                                          counts, config.get("oracle_min_output_tokens"))
            row_requests = None
            row_preparation = None
            if route == "spec":
                row_requests, _, row_preparation = live_lab._prepare_rows(source, raw, args)
                row_requests = [replace(row, request_id=f"{key}:row_{index:04d}")
                                for index, row in enumerate(row_requests)]
            request = replace(
                fixed_lab.request_for(source, crop, live_lab._b1_args(args)), request_id=key,
            )
            prepared = b1._prepare_cpu(request, time.perf_counter())
            groups = draft._iter_packed_prefill_groups(row_requests) if row_requests is not None else None
            return {
                "route": route, "source": source, "prepared": prepared, "groups": groups,
                "metadata": {"job": job, "source_id": source_id, "row_preparation": row_preparation,
                             "routing": routing},
                "timing": {"started": started, "finished": time.perf_counter(),
                           "thread_s": time.thread_time() - cpu_started,
                           "contract": "background CPU span may overlap decode; not additive to E2E"},
            }

        def matcher_factory(record: dict[str, Any]) -> Any:
            return matcher_executor.submit(
                live_lab._timed_matcher,
                record, b1.tokenizer, eos_token_id=int(b1.model.config.eos_token_id),
                block_size=args.initial_k,
            )

        try:
            while not draining or runtime.jobs or preparing:
                # Never wait for a second request while the first has work.
                while not draining and len(runtime.jobs) + len(preparing) < capacity:
                    try:
                        incoming = jobs.get() if not runtime.jobs and not preparing else jobs.get_nowait()
                    except queue.Empty:
                        break
                    if incoming is None:
                        draining = True
                    else:
                        key = str(incoming["request_id"])
                        ledger.admit(key)
                        preparing[key] = preparation_executor.submit(prepare, incoming)
                        if len(runtime.jobs) + len(preparing) > capacity:
                            raise RuntimeError("CPU preparation exceeded table admission capacity")
                        print(f"TABLE_PHASE preparing id={key} admitted={len(runtime.jobs) + len(preparing)}", flush=True)
                for key, future in list(preparing.items()):
                    # With useful model work available, never block on CPU
                    # preparation. Otherwise wait, rather than spinning idle.
                    if runtime.jobs and not future.done():
                        continue
                    try:
                        ready = account("cpu_prepare_wait", [key], future.result)
                        del preparing[key]
                        runtime.add(key, route=ready["route"], payload=ready["source"],
                                    target_prepared=ready["prepared"], row_groups=ready["groups"])
                        metadata[key] = ready["metadata"]
                        metadata[key]["cpu_preparation"] = ready["timing"]
                        print(f"TABLE_PHASE admitted id={key} route={ready['route']} active={len(runtime.jobs)}", flush=True)
                    except Exception as exc:
                        preparing.pop(key, None)
                        ledger.retire(key)
                        results.put({"kind": "result", "request_id": key, "ok": False,
                                     "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                if not runtime.jobs:
                    continue

                for key in list(runtime.jobs):
                    state = runtime.jobs[key]
                    if state["phase"] == "draft" and all(row.stop is not None for row in state["draft_states"]):
                        account("matcher_submit", [key], lambda key=key: runtime.transitions(matcher_factory, key))
                    else:
                        runtime.transitions(matcher_factory, key)
                    if runtime.jobs[key]["phase"] != "done":
                        continue
                    started = time.perf_counter()
                    state = runtime.jobs[key]
                    target = state["target"]
                    raw_text = b1.tokenizer.decode(target.tokens, skip_special_tokens=target.prefilled.skip_special_tokens)
                    text = normalize_recognition_text("table", raw_text)
                    ledger.record("output_postprocess", owners=[key], phases=snapshot(), wall_s=time.perf_counter() - started)
                    scheduling = ledger.retire(key)
                    info = metadata.pop(key)
                    original = targets_by_id[info["source_id"]]
                    response = {
                        "request_id": info["source_id"], "crop_type": "table",
                        "route_lane": state["route"], "raw_text": raw_text, "text": text,
                        "token_ids": list(target.tokens), "output_tokens": len(target.tokens),
                        "stop_reason": target.stop,
                        "exact_saved_reference": target.tokens == fixed_lab.target_tokens(original),
                        "worker_wall_s": time.perf_counter() - float(info["job"]["submitted_monotonic_s"]),
                        "stage_timing_s": {},
                        "runtime_metrics": {
                            "routing": info["routing"],
                            "phase_accounting": scheduling,
                            "prefills": state["prefill_records"],
                            "adaptive_trace": state["trace"],
                            "draft_rows": [{"row_index": row.slot % 8, "token_ids": row.tokens, "stop_reason": row.stop} for row in state["draft_states"]],
                            "target_calls": target.calls,
                            "matcher_timing": state["matcher_timing"],
                            "row_preparation": info["row_preparation"],
                            "cpu_preparation": info["cpu_preparation"],
                        },
                    }
                    runtime.retire(key)
                    policy.retire(key)
                    results.put({"kind": "result", "request_id": key, "ok": True, "payload": response})
                    completed += 1
                    print(f"TABLE_PHASE complete id={key} wall_s={response['worker_wall_s']:.4f} tokens={len(target.tokens)} exact={response['exact_saved_reference']} active={len(runtime.jobs)}", flush=True)

                if not runtime.jobs:
                    continue
                work = account("matcher_propose_control", list(runtime.jobs), runtime.work)
                selected = policy.choose(work)
                if not selected:
                    continue
                owners = [item.request_id for item in selected]
                if selected[0].phase.endswith("prefill"):
                    account(selected[0].phase, owners, lambda: runtime.prefill(owners[0]))
                else:
                    phases = snapshot()
                    started = time.perf_counter()
                    action, device_s = runtime.decode_step(selected)
                    ledger.record(action, owners=owners, phases=phases,
                                  wall_s=time.perf_counter() - started, device_s=device_s, decode=True)
            results.put({"kind": "service_summary", "payload": {
                "completed_requests": completed, **ledger.summary(), "graph_contracts": runtime.metadata,
                "q1_pipeline": runtime.pipeline_statistics(),
                "kv_prefix_preservation": runtime.cache_guard_statistics(),
                "batch_composition": runtime.batch_composition,
                "experimental_oracle_routing": oracle_metadata,
            }})
        except BaseException as exc:
            for key in list(runtime.jobs) + list(preparing):
                results.put({"kind": "result", "request_id": key, "ok": False,
                             "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
            raise
        finally:
            preparation_executor.shutdown(wait=True, cancel_futures=True)
            matcher_executor.shutdown(wait=True, cancel_futures=True)
            runtime.close()


def main() -> None:
    args = parse_args()
    counts, count_hash = (None, None) if args.experimental_oracle_token_counts is None else _oracle_counts(
        args.experimental_oracle_token_counts.expanduser().resolve())
    context = mp.get_context("spawn")
    jobs = context.Queue(maxsize=args.queue_capacity)
    results = context.Queue()
    config = {
        "targets": str(args.targets.expanduser().resolve()),
        "images_dir": str(args.images_dir.expanduser().resolve()),
        "model": str(args.model.expanduser().resolve()),
        "decode_vocab_token_ids": str(
            args.decode_vocab_token_ids.expanduser().resolve()
        ),
        "height_threshold_px": args.height_threshold_px,
        "oracle_token_counts": counts,
        "oracle_count_map_sha256": count_hash,
        "oracle_min_output_tokens": args.oracle_min_output_tokens,
        "cold_request_id": args.cold_request_id,
        "draft_cache_length": args.draft_cache_length,
        "cache_length": args.cache_length,
        "max_new_tokens": args.max_new_tokens,
        "decode_optimization": args.decode_optimization,
        "verifier_optimization": args.verifier_optimization,
        "k_values": args.k_values,
        "initial_k": args.initial_k,
        "allow_compile": args.allow_compile,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "interleaved_tables": args.interleaved_tables,
        "decode_cache_dir": str(args.decode_cache_dir.expanduser().resolve()),
        "vision_cache_dir": str(args.vision_cache_dir.expanduser().resolve()),
        "text_cache_dir": str(args.text_cache_dir.expanduser().resolve()),
        "text_packed_cache_dir": str(
            args.text_packed_cache_dir.expanduser().resolve()
        ),
    }
    worker = context.Process(
        target=_worker_main,
        args=(jobs, results, config),
        name="table-speculative-npu-worker",
    )
    worker.start()
    state = _State(
        jobs=jobs,
        results=results,
        worker=worker,
        timeout_s=args.request_timeout_s,
        max_image_bytes=args.max_image_bytes,
    )
    print(f"Waiting for NPU worker pid={worker.pid}", flush=True)
    state.ready.wait(timeout=args.request_timeout_s)
    if state.startup_error is not None:
        print(state.startup_error["traceback"], file=sys.stderr)
        raise RuntimeError(state.startup_error["error"])
    if not state.ready.is_set() or not worker.is_alive():
        raise RuntimeError("NPU worker did not become ready")

    server = _Server((args.host, args.port), _Handler)
    server.state = state  # type: ignore[attr-defined]
    stop_once = threading.Event()

    def stop_server(signum: int, frame: Any) -> None:
        del signum, frame
        if not stop_once.is_set():
            stop_once.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    print(
        f"READY http://{args.host}:{args.port} worker_pid={state.worker_pid}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        service_summary: dict[str, Any] | None = None
        try:
            service_summary = state.drain()
        except (queue.Empty, queue.Full):
            worker.terminate()
        if args.service_summary_output is not None and service_summary is not None:
            _write_service_summary(
                args.service_summary_output,
                configuration=state.configuration,
                worker_pid=state.worker_pid,
                summary=service_summary,
            )
            print(
                f"SERVICE_SUMMARY {args.service_summary_output.expanduser().resolve()}",
                flush=True,
            )
        worker.join(timeout=10.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5.0)
        state.stopping.set()
        server.server_close()


if __name__ == "__main__":
    main()
