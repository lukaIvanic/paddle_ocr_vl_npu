#!/usr/bin/env python3
"""Benchmark experiment-4 batched static decode schedules.

Experiment 4 starts with the serving-relevant part only: sequential no-padding
prefill for real crops, then true batched static decode over the filled KV
cache rows. It can also prefill a larger NPU-resident ready bank and hot-swap
finished items into fixed compiled decode slots.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from tokenizers import Tokenizer

from local_modeling_paddleocr_vl import (
    DECODE_ATTENTION,
    DECODE_CACHE_UPDATE,
    DECODE_LINEAR_WEIGHT_FORMAT,
    LocalPaddleOCRVLForConditionalGeneration,
    LocalPaddleOCRVLStaticCache,
    _resolve_model_dir,
    cast_decode_linear_weights_to_nz,
)
from probe_static_compile import DEFAULT_TORCHAIR_CACHE_DIR, compile_decode_module, maybe_sync
from run_local_recognition import (
    NPU_JIT_COMPILE_CHOICES,
    build_inputs,
    configure_npu_jit_compile,
    load_preprocessor_config,
    parse_dtype,
    preprocess_image,
    resolve_device,
)


PROFILE_METRIC_CHOICES = ("pipe", "memory", "l2", "memory_access")
EOS_MODE_CHOICES = ("none", "overlap_event_flags")
SCHEDULE_CHOICES = ("fixed_cohort", "hotswap")
STEP_TIMING_CHOICES = ("off", "cpu", "npu", "both")
REPORT_CHOICES = ("full", "summary")
DIAGNOSTIC_SWAP_COPY_CHOICES = ("direct", "clone")
REPO_ROOT = Path(__file__).resolve().parents[1]


def decode_attention_label(device: torch.device) -> str:
    return DECODE_ATTENTION if device.type == "npu" else "manual"


def decode_cache_update_label(device: torch.device) -> str:
    return DECODE_CACHE_UPDATE if device.type == "npu" else "per_row_copy"


@dataclass
class CohortInput:
    entry: dict[str, Any]
    crop_path: Path
    prompt: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor


@dataclass
class BatchedPrefill:
    cache: LocalPaddleOCRVLStaticCache
    rope_deltas: torch.Tensor
    next_cache_position: torch.Tensor


@dataclass
class ReadyBank:
    cache: LocalPaddleOCRVLStaticCache
    rope_deltas: torch.Tensor
    next_cache_position: torch.Tensor
    next_token: torch.Tensor


@dataclass
class DecodeLoopResult:
    ids: torch.Tensor
    last_logits: torch.Tensor | None
    decode_calls: int
    finished: torch.Tensor | None
    eos_steps: torch.Tensor | None
    stopped_all_eos: bool
    step_timing: list[dict[str, Any]] | None = None
    timing_recorder: Any | None = None


@dataclass
class HotSwapDecodeResult:
    ids: torch.Tensor
    lengths: torch.Tensor
    last_logits: torch.Tensor | None
    decode_calls: int
    completed_by_item: list[bool]
    completion_decode_calls: list[int | None]
    eos_hit_by_item: list[bool]
    length_cap_hit_by_item: list[bool]
    swap_events: list[dict[str, Any]]
    step_timing: list[dict[str, Any]] | None
    stopped_all_items: bool
    diagnostics: dict[str, Any] | None = None
    diagnostic_step_trace: list[dict[str, Any]] | None = None
    timing_recorder: Any | None = None


class StepTimingRecorder:
    def __init__(self, mode: str, device: torch.device):
        if mode not in STEP_TIMING_CHOICES:
            raise ValueError(f"unsupported step timing mode: {mode!r}")
        self.mode = mode
        self.device = device
        self.cpu_enabled = mode in ("cpu", "both")
        self.npu_enabled = mode in ("npu", "both")
        self.records: list[dict[str, Any]] = []
        self._event_records: list[tuple[int, dict[str, tuple[Any, Any]]]] = []
        self._torch_npu = None
        if self.npu_enabled:
            if device.type != "npu":
                raise ValueError("--step-timing npu/both requires --device npu:0")
            import torch_npu

            self._torch_npu = torch_npu

    def cpu_now(self) -> int | None:
        if not self.cpu_enabled:
            return None
        return time.perf_counter_ns()

    @staticmethod
    def cpu_elapsed_s(start_ns: int | None) -> float | None:
        if start_ns is None:
            return None
        return (time.perf_counter_ns() - int(start_ns)) / 1e9

    def npu_event(self, stream: Any | None = None) -> Any | None:
        if not self.npu_enabled:
            return None
        assert self._torch_npu is not None
        event = self._torch_npu.npu.Event(enable_timing=True)
        if stream is None:
            event.record(self._torch_npu.npu.current_stream())
        else:
            event.record(stream)
        return event

    def add(self, record: dict[str, Any], events: dict[str, tuple[Any, Any]] | None = None) -> None:
        if self.mode == "off":
            return
        clean_record = {key: value for key, value in record.items() if value is not None}
        self.records.append(clean_record)
        if self.npu_enabled and events:
            event_pairs = {
                name: pair
                for name, pair in events.items()
                if pair[0] is not None and pair[1] is not None
            }
            if event_pairs:
                self._event_records.append((len(self.records) - 1, event_pairs))

    def finalize(self) -> list[dict[str, Any]]:
        if self.npu_enabled and self._event_records:
            assert self._torch_npu is not None
            self._torch_npu.npu.synchronize()
            for record_idx, event_pairs in self._event_records:
                for name, (start_event, end_event) in event_pairs.items():
                    try:
                        self.records[record_idx][f"{name}_ms"] = float(start_event.elapsed_time(end_event))
                    except Exception as exc:
                        self.records[record_idx][f"{name}_ms_error"] = repr(exc)
        return self.records


def finalize_step_timing(result: DecodeLoopResult | HotSwapDecodeResult) -> None:
    recorder = result.timing_recorder
    if recorder is not None:
        result.step_timing = recorder.finalize()
        result.timing_recorder = None


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def timed(device: torch.device, fn: Callable):
    maybe_sync(device)
    start = time.perf_counter()
    result = fn()
    maybe_sync(device)
    return result, time.perf_counter() - start


def resolve_repo_path(path: Path) -> Path:
    path = path.expanduser()
    if path.exists():
        return path
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate
    return path


def load_manifest(path: Path) -> list[dict[str, Any]]:
    path = resolve_repo_path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def select_manifest_entries(
    manifest: list[dict[str, Any]],
    *,
    batch_size: int,
    num_items: int | None = None,
    crop_ids: list[str] | None,
    schedule: str = "fixed_cohort",
) -> list[dict[str, Any]]:
    if schedule not in SCHEDULE_CHOICES:
        raise ValueError(f"unsupported --schedule={schedule!r}")
    if batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {batch_size}")
    if crop_ids:
        by_id = {str(entry["id"]): entry for entry in manifest}
        missing = [crop_id for crop_id in crop_ids if crop_id not in by_id]
        if missing:
            raise ValueError(f"unknown --crop-ids entries: {missing}")
        if schedule == "fixed_cohort" and len(crop_ids) != batch_size:
            raise ValueError(f"--batch-size={batch_size} but --crop-ids contains {len(crop_ids)} ids")
        if schedule == "hotswap" and len(crop_ids) < batch_size:
            raise ValueError(f"--schedule hotswap requires at least --batch-size ids, got {len(crop_ids)}")
        if num_items is not None and len(crop_ids) != int(num_items):
            raise ValueError(f"--num-items={num_items} but --crop-ids contains {len(crop_ids)} ids")
        return [by_id[crop_id] for crop_id in crop_ids]

    selected_count = int(batch_size) if schedule == "fixed_cohort" else int(num_items or len(manifest))
    if schedule == "hotswap" and selected_count < int(batch_size):
        raise ValueError(f"--schedule hotswap requires --num-items >= --batch-size, got {selected_count} < {batch_size}")
    if selected_count > len(manifest):
        raise ValueError(f"requested {selected_count} real crops, but manifest has {len(manifest)}")
    return manifest[:selected_count]


def build_cohort_inputs(
    *,
    entries: list[dict[str, Any]],
    manifest_path: Path,
    tokenizer: Tokenizer,
    pre_cfg: dict[str, Any],
    prompt_override: str | None,
) -> list[CohortInput]:
    manifest_path = resolve_repo_path(manifest_path)
    crops_dir = manifest_path.parent
    cohort = []
    for entry in entries:
        crop_path = Path(str(entry["file"]))
        if not crop_path.is_absolute():
            crop_path = crops_dir / crop_path
        if not crop_path.exists():
            raise FileNotFoundError(f"crop not found for {entry.get('id')}: {crop_path}")
        prompt = str(prompt_override if prompt_override is not None else entry.get("suggested_prompt", "OCR:"))
        pixel_values, image_grid_thw = preprocess_image(crop_path, pre_cfg)
        input_ids, attention_mask = build_inputs(
            tokenizer,
            image_grid_thw,
            prompt,
            merge_size=int(pre_cfg["merge_size"]),
        )
        cohort.append(
            CohortInput(
                entry=entry,
                crop_path=crop_path,
                prompt=prompt,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        )
    return cohort


def hits_eos(token_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    return token_ids.reshape(-1) == int(eos_token_id)


def row_token_lists(ids: torch.Tensor) -> list[list[int]]:
    return [[int(value) for value in row] for row in ids.detach().cpu().tolist()]


def row_trimmed_token_lists(ids: torch.Tensor, eos_token_id: int) -> list[list[int]]:
    rows = row_token_lists(ids)
    trimmed = []
    for row in rows:
        try:
            first_eos = row.index(int(eos_token_id))
        except ValueError:
            trimmed.append(row)
        else:
            trimmed.append(row[: first_eos + 1])
    return trimmed


def safe_decode_tokens(tokenizer: Tokenizer, row: list[int]) -> str:
    try:
        return tokenizer.decode(row, skip_special_tokens=True)
    except Exception as exc:
        return f"<decode_error {type(exc).__name__}: {exc}>"


def length_trimmed_token_lists(ids: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    rows = row_token_lists(ids)
    length_values = [int(value) for value in lengths.detach().cpu().tolist()]
    return [row[: max(0, length)] for row, length in zip(rows, length_values)]


def token_id_range_summary(
    ids: torch.Tensor,
    *,
    vocab_size: int,
    lengths: torch.Tensor | None = None,
    max_samples: int = 8,
) -> dict[str, Any]:
    rows = length_trimmed_token_lists(ids, lengths) if lengths is not None else row_token_lists(ids)
    min_id = None
    max_id = None
    invalid_samples = []
    token_count = 0
    for row_idx, row in enumerate(rows):
        for col_idx, value in enumerate(row):
            token_count += 1
            min_id = int(value) if min_id is None else min(min_id, int(value))
            max_id = int(value) if max_id is None else max(max_id, int(value))
            if (int(value) < 0 or int(value) >= int(vocab_size)) and len(invalid_samples) < int(max_samples):
                invalid_samples.append(
                    {
                        "row": int(row_idx),
                        "position": int(col_idx),
                        "value": int(value),
                    }
                )
    invalid_count = 0
    for row in rows:
        invalid_count += sum(1 for value in row if int(value) < 0 or int(value) >= int(vocab_size))
    return {
        "vocab_size": int(vocab_size),
        "token_count": int(token_count),
        "min_id": None if min_id is None else int(min_id),
        "max_id": None if max_id is None else int(max_id),
        "invalid_count": int(invalid_count),
        "invalid_samples": invalid_samples,
    }


def first_mismatch_index(left: list[int], right: list[int]) -> int | None:
    for idx, (left_value, right_value) in enumerate(zip(left, right)):
        if int(left_value) != int(right_value):
            return int(idx)
    if len(left) != len(right):
        return int(min(len(left), len(right)))
    return None


def token_comparison_summary(left: list[int], right: list[int]) -> dict[str, Any]:
    mismatch = first_mismatch_index(left, right)
    return {
        "equal": bool(mismatch is None),
        "left_len": int(len(left)),
        "right_len": int(len(right)),
        "first_mismatch": None if mismatch is None else int(mismatch),
        "left_from_mismatch": [] if mismatch is None else [int(value) for value in left[mismatch : mismatch + 8]],
        "right_from_mismatch": [] if mismatch is None else [int(value) for value in right[mismatch : mismatch + 8]],
    }


def tensor_diff_summary(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if tuple(left.shape) != tuple(right.shape):
        return {
            "same_shape": False,
            "left_shape": [int(value) for value in left.shape],
            "right_shape": [int(value) for value in right.shape],
        }
    exact = bool(torch.equal(left, right))
    diff = (left.float() - right.float()).abs()
    return {
        "same_shape": True,
        "exact_equal": exact,
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
    }


def cache_diff_summary(left: LocalPaddleOCRVLStaticCache, right: LocalPaddleOCRVLStaticCache) -> dict[str, Any]:
    max_abs = 0.0
    sum_abs = 0.0
    count = 0
    exact = True
    for left_tensor, right_tensor in zip(left.key_caches + left.value_caches, right.key_caches + right.value_caches):
        if tuple(left_tensor.shape) != tuple(right_tensor.shape):
            return {
                "same_shape": False,
                "exact_equal": False,
                "max_abs": None,
                "mean_abs": None,
            }
        exact = exact and bool(torch.equal(left_tensor, right_tensor))
        diff = (left_tensor.float() - right_tensor.float()).abs()
        if diff.numel():
            max_abs = max(max_abs, float(diff.max().item()))
            sum_abs += float(diff.sum().item())
            count += int(diff.numel())
    return {
        "same_shape": True,
        "exact_equal": bool(exact),
        "max_abs": float(max_abs),
        "mean_abs": float(sum_abs / count) if count else 0.0,
    }


def summarize_step_timing(records: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not records:
        return None

    def percentile(ordered: list[float], q: float) -> float:
        if not ordered:
            raise ValueError("percentile requires at least one value")
        idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
        return float(ordered[idx])

    def stats_for(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        ordered = sorted(values)
        return {
            "count": int(len(values)),
            "sum": float(sum(values)),
            "avg": float(sum(values) / len(values)),
            "min": float(ordered[0]),
            "p50": percentile(ordered, 0.50),
            "p90": percentile(ordered, 0.90),
            "p95": percentile(ordered, 0.95),
            "p99": percentile(ordered, 0.99),
            "max": float(ordered[-1]),
        }

    def collect(group: list[dict[str, Any]]) -> dict[str, Any]:
        keys = [
            "host_iter_s",
            "host_wait_prev_flag_s",
            "host_swap_s",
            "host_decode_enqueue_s",
            "host_flag_copy_enqueue_s",
            "npu_iter_ms",
            "npu_swap_ms",
            "npu_decode_ms",
            "npu_flag_copy_ms",
        ]
        output: dict[str, Any] = {"count": int(len(group))}
        for key in keys:
            values = [float(record[key]) for record in group if key in record]
            key_stats = stats_for(values)
            if key_stats is not None:
                output[key] = key_stats
        return output

    def group_by_int(field: str) -> dict[str, Any]:
        groups: dict[int, list[dict[str, Any]]] = {}
        for record in records:
            value = int(record.get(field, 0))
            groups.setdefault(value, []).append(record)
        return {str(value): collect(group) for value, group in sorted(groups.items())}

    def top_records(key: str, limit: int = 8) -> list[dict[str, Any]]:
        candidates = [record for record in records if key in record]
        candidates.sort(key=lambda record: float(record[key]), reverse=True)
        fields = [
            "step",
            "decode_call",
            "swap_count",
            "finished_slot_count",
            "deactivated_slot_count",
            "completed_item_ids",
            "swapped_slots",
            "swapped_in_item_ids",
            "host_iter_s",
            "host_wait_prev_flag_s",
            "host_swap_s",
            "host_decode_enqueue_s",
            "host_flag_copy_enqueue_s",
            "npu_iter_ms",
            "npu_swap_ms",
            "npu_decode_ms",
            "npu_flag_copy_ms",
        ]
        return [
            {field: record[field] for field in fields if field in record}
            for record in candidates[:limit]
        ]

    swap_records = [record for record in records if int(record.get("swap_count", 0)) > 0]
    no_swap_records = [record for record in records if int(record.get("swap_count", 0)) == 0]
    return {
        "all": collect(records),
        "no_swap": collect(no_swap_records),
        "swap": collect(swap_records),
        "by_swap_count": group_by_int("swap_count"),
        "by_finished_slot_count": group_by_int("finished_slot_count"),
        "top_host_iter_s": top_records("host_iter_s"),
        "top_npu_iter_ms": top_records("npu_iter_ms"),
        "top_npu_swap_ms": top_records("npu_swap_ms"),
        "top_host_swap_s": top_records("host_swap_s"),
        "top_host_wait_prev_flag_s": top_records("host_wait_prev_flag_s"),
        "swap_steps": [int(record["step"]) for record in swap_records if "step" in record],
    }


def step_timing_accounting(
    records: list[dict[str, Any]] | None,
    *,
    wall_s: float,
    batch_size: int,
    raw_token_calls: int,
    effective_token_calls: int,
) -> dict[str, Any] | None:
    if not records:
        return None

    def sum_key(key: str) -> float:
        return float(sum(float(record[key]) for record in records if key in record))

    host_iter_sum_s = sum_key("host_iter_s")
    npu_iter_sum_s = sum_key("npu_iter_ms") / 1000.0
    npu_decode_sum_s = sum_key("npu_decode_ms") / 1000.0
    npu_swap_sum_s = sum_key("npu_swap_ms") / 1000.0
    npu_flag_copy_sum_s = sum_key("npu_flag_copy_ms") / 1000.0
    host_swap_sum_s = sum_key("host_swap_s")
    host_wait_sum_s = sum_key("host_wait_prev_flag_s")
    host_decode_enqueue_sum_s = sum_key("host_decode_enqueue_s")
    host_flag_copy_enqueue_sum_s = sum_key("host_flag_copy_enqueue_s")

    return {
        "step_records": int(len(records)),
        "batch_size": int(batch_size),
        "wall_decode_s": float(wall_s),
        "wall_avg_step_s": float(wall_s / len(records)) if records else 0.0,
        "wall_decode_steps_per_s": tok_per_s(len(records), wall_s),
        "wall_raw_batch_tokens_per_s": tok_per_s(raw_token_calls, wall_s),
        "wall_effective_tokens_per_s": tok_per_s(effective_token_calls, wall_s),
        "host_iter_sum_s": host_iter_sum_s,
        "host_iter_sum_to_wall_ratio": float(host_iter_sum_s / wall_s) if wall_s > 0 else float("inf"),
        "wall_minus_host_iter_sum_s": float(wall_s - host_iter_sum_s),
        "host_region_sums": {
            "swap": host_swap_sum_s,
            "wait_prev_flag": host_wait_sum_s,
            "decode_enqueue": host_decode_enqueue_sum_s,
            "flag_copy_enqueue": host_flag_copy_enqueue_sum_s,
        },
        "npu_event_region_sums_s": {
            "iter": npu_iter_sum_s,
            "decode": npu_decode_sum_s,
            "swap": npu_swap_sum_s,
            "flag_copy": npu_flag_copy_sum_s,
        },
        "npu_iter_sum_to_wall_ratio": float(npu_iter_sum_s / wall_s) if wall_s > 0 else float("inf"),
    }


def slim_stats(stats: dict[str, Any] | None) -> dict[str, Any] | None:
    if not stats:
        return None
    keys = ("count", "avg", "p50", "p90", "p95", "max")
    return {key: stats[key] for key in keys if key in stats}


def slim_step_timing_summary(step_summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not step_summary:
        return None
    metric_keys = (
        "host_iter_s",
        "host_wait_prev_flag_s",
        "host_swap_s",
        "npu_iter_ms",
        "npu_decode_ms",
        "npu_swap_ms",
        "npu_flag_copy_ms",
    )
    groups: dict[str, Any] = {}
    for group_name in ("all", "no_swap", "swap"):
        group = step_summary.get(group_name)
        if not group:
            continue
        slim_group: dict[str, Any] = {"count": int(group.get("count", 0))}
        for metric_key in metric_keys:
            metric_stats = slim_stats(group.get(metric_key))
            if metric_stats is not None:
                slim_group[metric_key] = metric_stats
        groups[group_name] = slim_group
    groups["swap_steps_count"] = int(len(step_summary.get("swap_steps", [])))
    return groups


def benchmark_report(summary: dict[str, Any]) -> dict[str, Any]:
    common = {
        "experiment": summary.get("experiment"),
        "schedule": summary.get("schedule", "fixed_cohort"),
        "backend": summary.get("backend"),
        "device": summary.get("device"),
        "dtype": summary.get("dtype"),
        "batch_size": summary.get("batch_size"),
        "num_items": summary.get("num_items", summary.get("batch_size")),
        "npu_jit_compile": summary.get("npu_jit_compile"),
        "decode_attention": summary.get("decode_attention"),
        "decode_cache_update": summary.get("decode_cache_update"),
        "eos_mode": summary.get("eos_mode"),
        "step_timing_mode": summary.get("step_timing_mode"),
        "cache_length": summary.get("cache_length"),
        "timing_s": summary.get("timing_s"),
    }
    if summary.get("schedule") == "hotswap":
        hotswap_loop = summary.get("loop", {}).get("hotswap", {})
        hotswap_matches = summary.get("matches", {}).get("hotswap_vs_single_refs", {})
        return {
            **common,
            "history_write_mode": summary.get("history_write_mode"),
            "slot_control_write_mode": summary.get("slot_control_write_mode"),
            "correctness": {
                "all_required_checks_passed": summary.get("matches", {}).get("all_required_checks_passed"),
                "all_trimmed_match": hotswap_matches.get("all_trimmed_match"),
                "mismatch_count": hotswap_matches.get("mismatch_count"),
                "invalid_count": summary.get("token_ids", {}).get("invalid_count"),
                "first_mismatches": hotswap_matches.get("first_mismatches", [])[:2],
            },
            "tok_per_s": summary.get("tok_per_s"),
            "timing_accounting": summary.get("timing_accounting"),
            "loop": {
                "decode_calls": hotswap_loop.get("decode_calls"),
                "raw_decode_token_calls": hotswap_loop.get("raw_decode_token_calls"),
                "effective_decode_token_calls": hotswap_loop.get("effective_decode_token_calls"),
                "swap_event_count": hotswap_loop.get("swap_event_count"),
                "total_swapped_in_items": hotswap_loop.get("total_swapped_in_items"),
                "step_timing_summary": slim_step_timing_summary(hotswap_loop.get("step_timing_summary")),
            },
        }
    compiled_loop = summary.get("loop", {}).get("compiled", {})
    static_loop = summary.get("loop", {}).get("static_eager", {})
    return {
        **common,
        "correctness": {
            "all_required_checks_passed": summary.get("matches", {}).get("all_required_checks_passed"),
            "static_eager_vs_compiled": summary.get("matches", {}).get("static_eager_vs_compiled"),
            "static_eager_vs_compiled_trimmed": summary.get("matches", {}).get("static_eager_vs_compiled_trimmed"),
            "compare_loop_static_vs_compiled": summary.get("matches", {}).get("compare_loop_static_vs_compiled"),
            "static_eager_vs_single_refs_all_trimmed_match": summary.get("matches", {})
            .get("static_eager_vs_single_refs", {})
            .get("all_trimmed_match"),
        },
        "logit_diff_static_eager_vs_compiled_decode": summary.get("logit_diff_static_eager_vs_compiled_decode"),
        "tok_per_s": summary.get("tok_per_s"),
        "timing_accounting": {
            "static_eager": summary.get("timing_accounting", {}).get("static_eager"),
            "compiled": summary.get("timing_accounting", {}).get("compiled"),
        },
        "loop": {
            "static_eager": {
                "decode_calls": static_loop.get("decode_calls"),
                "raw_decode_token_calls": static_loop.get("raw_decode_token_calls"),
                "effective_decode_token_calls": static_loop.get("effective_decode_token_calls"),
                "step_timing_summary": slim_step_timing_summary(static_loop.get("step_timing_summary")),
            },
            "compiled": {
                "decode_calls": compiled_loop.get("decode_calls"),
                "raw_decode_token_calls": compiled_loop.get("raw_decode_token_calls"),
                "effective_decode_token_calls": compiled_loop.get("effective_decode_token_calls"),
                "step_timing_summary": slim_step_timing_summary(compiled_loop.get("step_timing_summary")),
            },
        },
    }


def decode_loop_summary(result: DecodeLoopResult, *, eos_token_id: int) -> dict[str, Any]:
    trimmed_rows = row_trimmed_token_lists(result.ids, eos_token_id)
    effective_decode_calls_by_row = [max(0, len(row) - 1) for row in trimmed_rows]
    finished = None if result.finished is None else [bool(value) for value in result.finished.detach().cpu().tolist()]
    eos_steps = None
    if result.eos_steps is not None:
        eos_steps = []
        for value in result.eos_steps.detach().cpu().tolist():
            eos_steps.append(None if int(value) < 0 else int(value))
    return {
        "batch_size": int(result.ids.shape[0]),
        "generated_new_tokens_per_row": [int(result.ids.shape[1])] * int(result.ids.shape[0]),
        "trimmed_new_tokens_per_row": [len(row) for row in trimmed_rows],
        "decode_calls": int(result.decode_calls),
        "raw_decode_token_calls": int(result.decode_calls) * int(result.ids.shape[0]),
        "effective_decode_calls_by_row": effective_decode_calls_by_row,
        "effective_decode_token_calls": int(sum(effective_decode_calls_by_row)),
        "finished_by_row": finished,
        "eos_steps_by_row": eos_steps,
        "stopped_all_eos": bool(result.stopped_all_eos),
        "step_timing_summary": summarize_step_timing(result.step_timing),
    }


def hotswap_loop_summary(result: HotSwapDecodeResult, *, batch_size: int) -> dict[str, Any]:
    lengths = [int(value) for value in result.lengths.detach().cpu().tolist()]
    effective_decode_calls_by_item = [max(0, length - 1) for length in lengths]
    return {
        "batch_size": int(batch_size),
        "num_items": int(result.ids.shape[0]),
        "generated_new_tokens_by_item": lengths,
        "decode_calls": int(result.decode_calls),
        "raw_decode_token_calls": int(result.decode_calls) * int(batch_size),
        "effective_decode_calls_by_item": effective_decode_calls_by_item,
        "effective_decode_token_calls": int(sum(effective_decode_calls_by_item)),
        "completed_by_item": [bool(value) for value in result.completed_by_item],
        "completion_decode_calls": result.completion_decode_calls,
        "eos_hit_by_item": [bool(value) for value in result.eos_hit_by_item],
        "length_cap_hit_by_item": [bool(value) for value in result.length_cap_hit_by_item],
        "stopped_all_items": bool(result.stopped_all_items),
        "swap_event_count": int(len(result.swap_events)),
        "total_swapped_in_items": int(sum(len(event.get("swapped_in_item_ids", [])) for event in result.swap_events)),
        "step_timing_summary": summarize_step_timing(result.step_timing),
    }


def make_single_prefill_from_ready(ready: ReadyBank, item_idx: int) -> BatchedPrefill:
    return BatchedPrefill(
        cache=LocalPaddleOCRVLStaticCache(
            tuple(cache[item_idx : item_idx + 1].clone().contiguous() for cache in ready.cache.key_caches),
            tuple(cache[item_idx : item_idx + 1].clone().contiguous() for cache in ready.cache.value_caches),
            int(ready.cache.cache_length),
        ),
        rope_deltas=ready.rope_deltas[item_idx : item_idx + 1].clone().contiguous(),
        next_cache_position=ready.next_cache_position[item_idx : item_idx + 1].clone().contiguous(),
    )


@torch.inference_mode()
def validate_hotswap_against_single_refs(
    *,
    flat_decode: Callable,
    ready: ReadyBank,
    hotswap_ids: torch.Tensor,
    hotswap_lengths: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
) -> dict[str, Any]:
    hotswap_rows = length_trimmed_token_lists(hotswap_ids, hotswap_lengths)
    reference_rows = []
    first_mismatches = []

    for item_idx in range(int(ready.next_token.shape[0])):
        single_prefill = make_single_prefill_from_ready(ready, item_idx)
        single_result = static_flat_decode_loop(
            flat_decode,
            single_prefill,
            ready.next_token[item_idx : item_idx + 1].clone().contiguous(),
            max_new_tokens=int(max_new_tokens),
            eos_mode="overlap_event_flags",
            eos_token_id=int(eos_token_id),
            step_timing="off",
        )
        reference_row = row_trimmed_token_lists(single_result.ids, int(eos_token_id))[0]
        reference_rows.append(reference_row)
        if hotswap_rows[item_idx] != reference_row and len(first_mismatches) < 8:
            first_mismatches.append(
                {
                    "item": int(item_idx),
                    "hotswap": hotswap_rows[item_idx],
                    "reference": reference_row,
                    "hotswap_len": int(len(hotswap_rows[item_idx])),
                    "reference_len": int(len(reference_row)),
                }
            )

    matches_by_item = [
        hotswap_row == reference_row
        for hotswap_row, reference_row in zip(hotswap_rows, reference_rows)
    ]
    return {
        "reference": "single_item_static_eager_from_contiguous_ready_bank_clone",
        "why_clone": "ready bank rows are non-contiguous views from a larger NPU cache; validation must use normal B=1 cache tensors",
        "items_checked": int(len(matches_by_item)),
        "trimmed_matches_by_item": [bool(value) for value in matches_by_item],
        "all_trimmed_match": bool(all(matches_by_item)),
        "mismatch_count": int(sum(1 for value in matches_by_item if not value)),
        "first_mismatches": first_mismatches,
    }


@torch.inference_mode()
def diagnose_hotswap_item_traces(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    flat_decode: Callable,
    ready: ReadyBank,
    cohort: list[CohortInput],
    hotswap_ids: torch.Tensor,
    hotswap_lengths: torch.Tensor,
    item_indices: list[int],
    cache_length: int,
    max_new_tokens: int,
    eos_token_id: int,
    device: torch.device,
    tokenizer: Tokenizer,
) -> list[dict[str, Any]]:
    hotswap_rows = length_trimmed_token_lists(hotswap_ids, hotswap_lengths)
    traces = []
    for item_idx in item_indices:
        if int(item_idx) < 0 or int(item_idx) >= len(cohort):
            raise ValueError(f"--diagnostic-trace-items contains out-of-range item index {item_idx}; num_items={len(cohort)}")

        item = cohort[int(item_idx)]
        ready_single_prefill = make_single_prefill_from_ready(ready, int(item_idx))
        ready_single_result = static_flat_decode_loop(
            flat_decode,
            ready_single_prefill,
            ready.next_token[int(item_idx) : int(item_idx) + 1].clone().contiguous(),
            max_new_tokens=int(max_new_tokens),
            eos_mode="overlap_event_flags",
            eos_token_id=int(eos_token_id),
            step_timing="off",
        )
        direct_prefill, direct_next_token = make_static_prefill(
            model,
            item.input_ids.to(device),
            item.attention_mask.to(device),
            item.pixel_values.to(device),
            item.image_grid_thw.to(device),
            cache_length=int(cache_length),
        )
        direct_result = static_flat_decode_loop(
            flat_decode,
            direct_prefill,
            direct_next_token,
            max_new_tokens=int(max_new_tokens),
            eos_mode="overlap_event_flags",
            eos_token_id=int(eos_token_id),
            step_timing="off",
        )

        hotswap_row = hotswap_rows[int(item_idx)]
        ready_single_row = row_trimmed_token_lists(ready_single_result.ids, int(eos_token_id))[0]
        direct_row = row_trimmed_token_lists(direct_result.ids, int(eos_token_id))[0]
        traces.append(
            {
                "item": int(item_idx),
                "id": str(item.entry.get("id")),
                "file": str(item.crop_path),
                "category_type": item.entry.get("category_type"),
                "prompt": item.prompt,
                "prompt_tokens": int(item.input_ids.shape[1]),
                "generated": {
                    "hotswap": [int(value) for value in hotswap_row],
                    "single_ready_clone": [int(value) for value in ready_single_row],
                    "single_direct_prefill": [int(value) for value in direct_row],
                },
                "texts": {
                    "hotswap": safe_decode_tokens(tokenizer, hotswap_row),
                    "single_ready_clone": safe_decode_tokens(tokenizer, ready_single_row),
                    "single_direct_prefill": safe_decode_tokens(tokenizer, direct_row),
                },
                "comparisons": {
                    "hotswap_vs_single_ready_clone": token_comparison_summary(hotswap_row, ready_single_row),
                    "hotswap_vs_single_direct_prefill": token_comparison_summary(hotswap_row, direct_row),
                    "single_ready_clone_vs_single_direct_prefill": token_comparison_summary(ready_single_row, direct_row),
                },
                "prefill_compare_ready_bank_vs_direct": {
                    "next_token": tensor_diff_summary(
                        ready.next_token[int(item_idx) : int(item_idx) + 1].clone().contiguous(),
                        direct_next_token,
                    ),
                    "next_cache_position": tensor_diff_summary(
                        ready.next_cache_position[int(item_idx) : int(item_idx) + 1].clone().contiguous(),
                        direct_prefill.next_cache_position.reshape(1),
                    ),
                    "rope_deltas": tensor_diff_summary(
                        ready.rope_deltas[int(item_idx) : int(item_idx) + 1].clone().contiguous(),
                        direct_prefill.rope_deltas,
                    ),
                    "kv_cache": cache_diff_summary(ready_single_prefill.cache, direct_prefill.cache),
                },
            }
        )
    return traces


@torch.inference_mode()
def static_flat_decode_loop(
    decode_fn: Callable,
    prefill: BatchedPrefill,
    next_token: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_mode: str = "none",
    eos_token_id: int | None = None,
    step_timing: str = "off",
) -> DecodeLoopResult:
    if eos_mode not in EOS_MODE_CHOICES:
        raise ValueError(f"unsupported eos_mode={eos_mode!r}")
    if eos_mode != "none" and eos_token_id is None:
        raise ValueError(f"eos_mode={eos_mode!r} requires eos_token_id")
    if int(max_new_tokens) <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

    batch_size = int(next_token.shape[0])
    generated = [next_token]
    cache_position = prefill.next_cache_position
    flat_cache = prefill.cache.flat_tensors()
    last_logits = None
    max_decode_calls = max(0, int(max_new_tokens) - 1)
    decode_calls = 0
    stopped_all_eos = False
    finished = None
    eos_steps = None
    async_cpu_flags = None
    copy_stream = None
    pending_eos_event = None
    pending_eos_step = None
    timing = StepTimingRecorder(step_timing, next_token.device)

    use_npu_overlap = eos_mode == "overlap_event_flags" and next_token.device.type == "npu"
    if eos_mode == "overlap_event_flags":
        finished = hits_eos(next_token, int(eos_token_id))
        eos_steps = torch.full((batch_size,), -1, device=next_token.device, dtype=torch.int64)
        eos_steps = torch.where(finished, torch.zeros_like(eos_steps), eos_steps)
        if use_npu_overlap:
            import torch_npu

            async_cpu_flags = torch.zeros((max_decode_calls, batch_size), dtype=torch.bool, pin_memory=True)
            copy_stream = torch_npu.npu.Stream(device=next_token.device)

    for step in range(max_decode_calls):
        record: dict[str, Any] = {
            "step": int(step),
            "decode_call": int(step) + 1,
            "swap_count": 0,
        }
        events: dict[str, tuple[Any, Any]] = {}
        host_iter_start = timing.cpu_now()
        npu_iter_start = timing.npu_event()
        host_decode_start = timing.cpu_now()
        npu_decode_start = timing.npu_event()
        last_logits = decode_fn(next_token, cache_position, prefill.rope_deltas, *flat_cache)
        npu_decode_end = timing.npu_event()
        decode_enqueue_s = timing.cpu_elapsed_s(host_decode_start)
        if decode_enqueue_s is not None:
            record["host_decode_enqueue_s"] = decode_enqueue_s
        events["npu_decode"] = (npu_decode_start, npu_decode_end)
        next_token = torch.argmax(last_logits[:, -1, :].float(), dim=-1, keepdim=True)
        if eos_mode == "overlap_event_flags":
            assert finished is not None
            assert eos_steps is not None
            active_before_step = ~finished
            eos_fill = torch.full_like(next_token, int(eos_token_id))
            next_token = torch.where(active_before_step.view(-1, 1), next_token, eos_fill)
            new_hits = hits_eos(next_token, int(eos_token_id)) & active_before_step
            eos_step_values = torch.full_like(eos_steps, int(step) + 1)
            eos_steps = torch.where((eos_steps < 0) & new_hits, eos_step_values, eos_steps)
            finished = finished | new_hits
        generated.append(next_token)
        cache_position = cache_position + 1
        decode_calls += 1

        if use_npu_overlap and async_cpu_flags is not None and copy_stream is not None:
            host_flag_copy_start = timing.cpu_now()
            eos_ready_event = torch_npu.npu.current_stream().record_event()
            copy_done_event = torch_npu.npu.Event()
            npu_flag_copy_start = None
            npu_flag_copy_end = None
            with torch_npu.npu.stream(copy_stream):
                copy_stream.wait_event(eos_ready_event)
                npu_flag_copy_start = timing.npu_event(copy_stream)
                async_cpu_flags[step].copy_(finished, non_blocking=True)
                npu_flag_copy_end = timing.npu_event(copy_stream)
                copy_done_event.record(copy_stream)
            flag_copy_enqueue_s = timing.cpu_elapsed_s(host_flag_copy_start)
            if flag_copy_enqueue_s is not None:
                record["host_flag_copy_enqueue_s"] = flag_copy_enqueue_s
            events["npu_flag_copy"] = (npu_flag_copy_start, npu_flag_copy_end)
            if pending_eos_event is not None and pending_eos_step is not None:
                host_wait_start = timing.cpu_now()
                pending_eos_event.synchronize()
                wait_s = timing.cpu_elapsed_s(host_wait_start)
                if wait_s is not None:
                    record["host_wait_prev_flag_s"] = wait_s
                if bool(async_cpu_flags[pending_eos_step].all().item()):
                    stopped_all_eos = True
                    npu_iter_end = timing.npu_event()
                    iter_s = timing.cpu_elapsed_s(host_iter_start)
                    if iter_s is not None:
                        record["host_iter_s"] = iter_s
                    events["npu_iter"] = (npu_iter_start, npu_iter_end)
                    timing.add(record, events)
                    break
            pending_eos_event = copy_done_event
            pending_eos_step = int(step)
        elif eos_mode == "overlap_event_flags":
            if bool(finished.detach().cpu().all().item()):
                stopped_all_eos = True
                npu_iter_end = timing.npu_event()
                iter_s = timing.cpu_elapsed_s(host_iter_start)
                if iter_s is not None:
                    record["host_iter_s"] = iter_s
                events["npu_iter"] = (npu_iter_start, npu_iter_end)
                timing.add(record, events)
                break

        npu_iter_end = timing.npu_event()
        iter_s = timing.cpu_elapsed_s(host_iter_start)
        if iter_s is not None:
            record["host_iter_s"] = iter_s
        events["npu_iter"] = (npu_iter_start, npu_iter_end)
        timing.add(record, events)

    if eos_mode == "overlap_event_flags" and not stopped_all_eos and pending_eos_event is not None and pending_eos_step is not None:
        pending_eos_event.synchronize()
        if bool(async_cpu_flags[pending_eos_step].all().item()):
            stopped_all_eos = True

    return DecodeLoopResult(
        ids=torch.cat(generated, dim=1),
        last_logits=last_logits,
        decode_calls=decode_calls,
        finished=finished,
        eos_steps=eos_steps,
        stopped_all_eos=stopped_all_eos,
        step_timing=timing.records,
        timing_recorder=timing if step_timing != "off" else None,
    )


@torch.inference_mode()
def make_static_prefill(
    model: LocalPaddleOCRVLForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    cache_length: int,
):
    prefill = model.forward_static_prefill(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        cache_length=cache_length,
        logits_to_keep=1,
    )
    next_token = torch.argmax(prefill.logits[:, -1, :].float(), dim=-1, keepdim=True)
    return prefill, next_token


@torch.inference_mode()
def make_batched_prefill(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    cohort: list[CohortInput],
    cache_length: int,
    device: torch.device,
):
    slot_prefills = []
    slot_next_tokens = []
    for item in cohort:
        prefill, next_token = make_static_prefill(
            model,
            item.input_ids.to(device),
            item.attention_mask.to(device),
            item.pixel_values.to(device),
            item.image_grid_thw.to(device),
            cache_length=cache_length,
        )
        slot_prefills.append(prefill)
        slot_next_tokens.append(next_token)

    num_layers = int(model.config.text_config.num_hidden_layers)
    key_caches = tuple(
        torch.cat([slot.cache.key_caches[layer_idx] for slot in slot_prefills], dim=0).contiguous()
        for layer_idx in range(num_layers)
    )
    value_caches = tuple(
        torch.cat([slot.cache.value_caches[layer_idx] for slot in slot_prefills], dim=0).contiguous()
        for layer_idx in range(num_layers)
    )
    batched_prefill = BatchedPrefill(
        cache=LocalPaddleOCRVLStaticCache(key_caches, value_caches, int(cache_length)),
        rope_deltas=torch.cat([slot.rope_deltas for slot in slot_prefills], dim=0).contiguous(),
        next_cache_position=torch.cat([slot.next_cache_position.reshape(1) for slot in slot_prefills], dim=0).contiguous(),
    )
    return batched_prefill, torch.cat(slot_next_tokens, dim=0).contiguous(), slot_prefills, slot_next_tokens


@torch.inference_mode()
def make_ready_bank(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    cohort: list[CohortInput],
    cache_length: int,
    device: torch.device,
) -> tuple[ReadyBank, list[Any]]:
    if not cohort:
        raise ValueError("hot-swap ready bank requires at least one crop")
    cache_dtype = next(model.parameters()).dtype
    bank_cache = model.allocate_static_cache(
        batch_size=len(cohort),
        cache_length=cache_length,
        device=device,
        dtype=cache_dtype,
        init_mode="zeros",
    )
    slot_prefills = []
    next_token_rows = []
    cache_position_rows = []
    rope_delta_rows = []
    for item_idx, item in enumerate(cohort):
        row_cache = LocalPaddleOCRVLStaticCache(
            tuple(cache[item_idx : item_idx + 1] for cache in bank_cache.key_caches),
            tuple(cache[item_idx : item_idx + 1] for cache in bank_cache.value_caches),
            int(cache_length),
        )
        prefill = model.forward_static_prefill(
            input_ids=item.input_ids.to(device),
            attention_mask=item.attention_mask.to(device),
            pixel_values=item.pixel_values.to(device),
            image_grid_thw=item.image_grid_thw.to(device),
            cache_length=cache_length,
            cache=row_cache,
            logits_to_keep=1,
        )
        next_token = torch.argmax(prefill.logits[:, -1, :].float(), dim=-1, keepdim=True)
        slot_prefills.append(prefill)
        next_token_rows.append(next_token)
        cache_position_rows.append(prefill.next_cache_position.reshape(1))
        rope_delta_rows.append(prefill.rope_deltas)

    ready = ReadyBank(
        cache=bank_cache,
        rope_deltas=torch.cat(rope_delta_rows, dim=0).contiguous(),
        next_cache_position=torch.cat(cache_position_rows, dim=0).contiguous(),
        next_token=torch.cat(next_token_rows, dim=0).contiguous(),
    )
    return ready, slot_prefills


@torch.inference_mode()
def copy_batch_row_(
    target: torch.Tensor,
    slot: int,
    source_row: torch.Tensor,
) -> None:
    slot = int(slot)
    row_shape = (1, *target.shape[1:])
    row = source_row.to(device=target.device, dtype=target.dtype).reshape(row_shape).contiguous()
    if target.device.type == "npu":
        indices = torch.tensor([slot], device=target.device, dtype=torch.int64)
        target.index_copy_(0, indices, row)
        return
    target[slot : slot + 1].copy_(row)


@torch.inference_mode()
def fill_batch_row_(
    target: torch.Tensor,
    slot: int,
    value: int | bool,
) -> None:
    row = torch.full((1, *target.shape[1:]), value, device=target.device, dtype=target.dtype)
    copy_batch_row_(target, int(slot), row)


@torch.inference_mode()
def copy_vector_position_(
    target: torch.Tensor,
    position: int,
    source_value: torch.Tensor,
) -> None:
    position = int(position)
    value = source_value.to(device=target.device, dtype=target.dtype).reshape(1).contiguous()
    if target.device.type == "npu":
        indices = torch.tensor([position], device=target.device, dtype=torch.int64)
        target.index_copy_(0, indices, value)
        return
    target[position : position + 1].copy_(value)


@torch.inference_mode()
def copy_ready_item_to_active_slot_(
    *,
    ready: ReadyBank,
    active: BatchedPrefill,
    active_next_token: torch.Tensor,
    active_item_indices: torch.Tensor,
    active_mask: torch.Tensor,
    slot: int,
    item_idx: int,
    copy_mode: str = "direct",
) -> None:
    if copy_mode not in DIAGNOSTIC_SWAP_COPY_CHOICES:
        raise ValueError(f"unsupported hot-swap copy mode: {copy_mode!r}")
    slot_slice = slice(int(slot), int(slot) + 1)
    item_slice = slice(int(item_idx), int(item_idx) + 1)
    for layer_idx in range(len(active.cache.key_caches)):
        key_row = ready.cache.key_caches[layer_idx][item_slice]
        value_row = ready.cache.value_caches[layer_idx][item_slice]
        if copy_mode == "clone":
            key_row = key_row.clone().contiguous()
            value_row = value_row.clone().contiguous()
        active.cache.key_caches[layer_idx][slot_slice].copy_(key_row)
        active.cache.value_caches[layer_idx][slot_slice].copy_(value_row)
    rope_row = ready.rope_deltas[item_slice]
    position_row = ready.next_cache_position[item_slice]
    next_token_row = ready.next_token[item_slice]
    if copy_mode == "clone":
        rope_row = rope_row.clone().contiguous()
        position_row = position_row.clone().contiguous()
        next_token_row = next_token_row.clone().contiguous()
    copy_batch_row_(active.rope_deltas, int(slot), rope_row)
    copy_batch_row_(active.next_cache_position, int(slot), position_row)
    copy_batch_row_(active_next_token, int(slot), next_token_row)
    fill_batch_row_(active_item_indices, int(slot), int(item_idx))
    fill_batch_row_(active_mask, int(slot), True)


def verify_ready_item_in_active_slot(
    *,
    ready: ReadyBank,
    active: BatchedPrefill,
    active_next_token: torch.Tensor,
    slot: int,
    item_idx: int,
    stage: str,
    max_failures: int,
) -> list[dict[str, Any]]:
    if max_failures <= 0:
        return []
    slot_slice = slice(int(slot), int(slot) + 1)
    item_slice = slice(int(item_idx), int(item_idx) + 1)
    failures: list[dict[str, Any]] = []

    def check_tensor(name: str, active_tensor: torch.Tensor, ready_tensor: torch.Tensor, layer_idx: int | None = None) -> None:
        if len(failures) >= max_failures:
            return
        if torch.equal(active_tensor, ready_tensor):
            return
        diff = (active_tensor.float() - ready_tensor.float()).abs()
        failure = {
            "stage": stage,
            "slot": int(slot),
            "item": int(item_idx),
            "tensor": name,
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
        }
        if layer_idx is not None:
            failure["layer"] = int(layer_idx)
        failures.append(failure)

    for layer_idx in range(len(active.cache.key_caches)):
        if len(failures) >= max_failures:
            break
        check_tensor(
            "key_cache",
            active.cache.key_caches[layer_idx][slot_slice],
            ready.cache.key_caches[layer_idx][item_slice],
            layer_idx,
        )
        check_tensor(
            "value_cache",
            active.cache.value_caches[layer_idx][slot_slice],
            ready.cache.value_caches[layer_idx][item_slice],
            layer_idx,
        )
    check_tensor("rope_deltas", active.rope_deltas[slot_slice], ready.rope_deltas[item_slice])
    check_tensor("next_cache_position", active.next_cache_position[slot_slice], ready.next_cache_position[item_slice])
    check_tensor("next_token", active_next_token[slot_slice], ready.next_token[item_slice])
    return failures


@torch.inference_mode()
def static_hotswap_decode_loop(
    decode_fn: Callable,
    ready: ReadyBank,
    *,
    batch_size: int,
    max_new_tokens: int,
    eos_mode: str,
    eos_token_id: int,
    step_timing: str = "off",
    diagnostic_swap_copy_mode: str = "direct",
    diagnostic_verify_swap_copies: bool = False,
    diagnostic_sync_finished_flags: bool = False,
    diagnostic_step_trace: bool = False,
    diagnostic_step_trace_items: list[int] | None = None,
    diagnostic_step_trace_print: bool = False,
) -> HotSwapDecodeResult:
    if eos_mode != "overlap_event_flags":
        raise ValueError("--schedule hotswap requires --eos-mode overlap_event_flags")
    if int(batch_size) <= 0:
        raise ValueError(f"--batch-size must be positive, got {batch_size}")
    if int(max_new_tokens) <= 0:
        raise ValueError(f"--max-new-tokens must be positive, got {max_new_tokens}")

    device = ready.next_token.device
    use_npu_overlap = device.type == "npu" and not bool(diagnostic_sync_finished_flags)
    if use_npu_overlap:
        import torch_npu

    num_items = int(ready.next_token.shape[0])
    if int(batch_size) > num_items:
        raise ValueError(f"hot-swap batch_size={batch_size} exceeds num_items={num_items}")

    active_cache_shape = (
        int(batch_size),
        int(ready.cache.key_caches[0].shape[1]),
        int(ready.cache.cache_length),
        int(ready.cache.key_caches[0].shape[3]),
    )
    active_key_caches = tuple(
        torch.zeros(active_cache_shape, device=device, dtype=ready.cache.key_caches[layer_idx].dtype)
        for layer_idx in range(len(ready.cache.key_caches))
    )
    active_value_caches = tuple(
        torch.zeros(active_cache_shape, device=device, dtype=ready.cache.value_caches[layer_idx].dtype)
        for layer_idx in range(len(ready.cache.value_caches))
    )
    active_cache = LocalPaddleOCRVLStaticCache(
        active_key_caches,
        active_value_caches,
        int(ready.cache.cache_length),
    )
    active_rope_deltas = torch.empty((int(batch_size), int(ready.rope_deltas.shape[1])), device=device, dtype=ready.rope_deltas.dtype)
    active_cache_position = torch.empty((int(batch_size),), device=device, dtype=ready.next_cache_position.dtype)
    active_next_token = torch.empty((int(batch_size), 1), device=device, dtype=ready.next_token.dtype)
    active_item_indices = torch.full((int(batch_size),), -1, device=device, dtype=torch.int64)
    active_mask = torch.zeros((int(batch_size),), device=device, dtype=torch.bool)
    slot_finished = torch.zeros((int(batch_size),), device=device, dtype=torch.bool)
    generated_rows = [
        torch.full((int(max_new_tokens),), int(eos_token_id), device=device, dtype=ready.next_token.dtype)
        for _ in range(num_items)
    ]
    ready_first_tokens_cpu = [int(value) for value in ready.next_token.reshape(-1).detach().cpu().tolist()]
    generated_lengths_cpu = [0 for _ in range(num_items)]
    item_eos_hit_cpu = [False for _ in range(num_items)]
    item_length_cap_hit_cpu = [False for _ in range(num_items)]
    active = BatchedPrefill(
        cache=active_cache,
        rope_deltas=active_rope_deltas,
        next_cache_position=active_cache_position,
    )
    flat_cache = active.cache.flat_tensors()
    active_item_indices_cpu = [-1 for _ in range(int(batch_size))]
    slot_finished_cpu = [False for _ in range(int(batch_size))]
    completed_by_item = [False for _ in range(num_items)]
    completion_decode_calls: list[int | None] = [None for _ in range(num_items)]
    completed_count = 0
    next_ready_idx = 0
    swap_events: list[dict[str, Any]] = []
    last_logits = None
    decode_calls = 0
    timing = StepTimingRecorder(step_timing, device)
    copy_verification_failures: list[dict[str, Any]] = []
    copy_verification_checks = 0
    max_copy_verification_failures = 16
    step_trace: list[dict[str, Any]] = []
    trace_item_indices = (
        [int(value) for value in diagnostic_step_trace_items]
        if diagnostic_step_trace_items
        else []
    )
    for trace_item in trace_item_indices:
        if trace_item < 0 or trace_item >= num_items:
            raise ValueError(f"--diagnostic-step-trace-items contains out-of-range item index {trace_item}; num_items={num_items}")

    def snapshot_step(stage: str, *, extra: dict[str, Any] | None = None) -> None:
        if not diagnostic_step_trace and not trace_item_indices:
            return
        active_item_indices_tensor = [int(value) for value in active_item_indices.detach().cpu().tolist()]
        active_mask_values = [bool(value) for value in active_mask.detach().cpu().tolist()]
        slot_finished_values = [bool(value) for value in slot_finished.detach().cpu().tolist()]
        cache_position_values = [int(value) for value in active.next_cache_position.detach().cpu().tolist()]
        active_next_token_values = [int(row[0]) for row in active_next_token.detach().cpu().tolist()]
        selected_items = list(range(num_items)) if diagnostic_step_trace else trace_item_indices
        generated_full = None
        generated_trimmed = None
        if diagnostic_step_trace:
            generated_full = [[int(value) for value in row.detach().cpu().tolist()] for row in generated_rows]
            generated_trimmed = [
                row[: max(0, int(length))]
                for row, length in zip(generated_full, generated_lengths_cpu)
            ]
        selected_generated_full = {
            str(item_idx): [int(value) for value in generated_rows[item_idx].detach().cpu().tolist()]
            for item_idx in selected_items
        }
        selected_generated_trimmed = {
            str(item_idx): selected_generated_full[str(item_idx)][: max(0, int(generated_lengths_cpu[item_idx]))]
            for item_idx in selected_items
        }
        record = {
            "stage": stage,
            "decode_calls": int(decode_calls),
            "completed_count": int(completed_count),
            "next_ready_idx": int(next_ready_idx),
            "active_item_indices_cpu": [int(value) for value in active_item_indices_cpu],
            "active_item_indices_tensor": active_item_indices_tensor,
            "active_mask": active_mask_values,
            "slot_finished_cpu": [bool(value) for value in slot_finished_cpu],
            "slot_finished_tensor": slot_finished_values,
            "active_next_token": active_next_token_values,
            "active_cache_position": cache_position_values,
            "generated_lengths": [int(value) for value in generated_lengths_cpu],
            "selected_items": [int(value) for value in selected_items],
            "selected_generated_ids_full": selected_generated_full,
            "selected_generated_ids_trimmed": selected_generated_trimmed,
        }
        if diagnostic_step_trace:
            record["generated_ids_full"] = generated_full
            record["generated_ids_trimmed"] = generated_trimmed
        if extra:
            record.update(extra)
        step_trace.append(record)
        if diagnostic_step_trace_print:
            print("DIAGNOSTIC_STEP_TRACE " + json.dumps(record, sort_keys=True, default=json_default), flush=True)

    def load_item_to_slot(slot: int, item_idx: int) -> None:
        copy_ready_item_to_active_slot_(
            ready=ready,
            active=active,
            active_next_token=active_next_token,
            active_item_indices=active_item_indices,
            active_mask=active_mask,
            slot=slot,
            item_idx=item_idx,
            copy_mode=diagnostic_swap_copy_mode,
        )
        nonlocal copy_verification_checks
        if diagnostic_verify_swap_copies:
            copy_verification_checks += 1
            if len(copy_verification_failures) < max_copy_verification_failures:
                copy_verification_failures.extend(
                    verify_ready_item_in_active_slot(
                        ready=ready,
                        active=active,
                        active_next_token=active_next_token,
                        slot=slot,
                        item_idx=item_idx,
                        stage="load_item_to_slot",
                        max_failures=max_copy_verification_failures - len(copy_verification_failures),
                    )
                )
        active_item_indices_cpu[int(slot)] = int(item_idx)
        copy_vector_position_(generated_rows[int(item_idx)], 0, ready.next_token[int(item_idx) : int(item_idx) + 1])
        generated_lengths_cpu[int(item_idx)] = 1
        first_token_eos = ready_first_tokens_cpu[int(item_idx)] == int(eos_token_id)
        first_token_cap = int(max_new_tokens) <= 1
        initial_finished = bool(first_token_eos or first_token_cap)
        slot_finished_cpu[int(slot)] = initial_finished
        fill_batch_row_(slot_finished, int(slot), initial_finished)
        item_eos_hit_cpu[int(item_idx)] = bool(first_token_eos)
        item_length_cap_hit_cpu[int(item_idx)] = bool(first_token_cap)

    def deactivate_slot(slot: int) -> None:
        active_item_indices_cpu[int(slot)] = -1
        fill_batch_row_(active_item_indices, int(slot), -1)
        fill_batch_row_(active_mask, int(slot), False)
        slot_finished_cpu[int(slot)] = False
        fill_batch_row_(slot_finished, int(slot), False)
        fill_batch_row_(active_next_token, int(slot), int(eos_token_id))
        fill_batch_row_(active.next_cache_position, int(slot), 0)
        fill_batch_row_(active.rope_deltas, int(slot), 0)

    def consume_finished_slots(finished_flags: list[bool], completion_decode_call: int, record: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal completed_count
        nonlocal next_ready_idx

        finished_slots = [
            slot
            for slot, flag in enumerate(finished_flags)
            if bool(flag) and active_item_indices_cpu[slot] >= 0
        ]
        completed_item_ids = []
        swapped_slots = []
        swapped_in_item_ids = []
        deactivated_slots = []
        for slot in finished_slots:
            finished_item = active_item_indices_cpu[slot]
            if not completed_by_item[finished_item]:
                completed_by_item[finished_item] = True
                completion_decode_calls[finished_item] = int(completion_decode_call)
                completed_count += 1
                completed_item_ids.append(int(finished_item))
                if not item_length_cap_hit_cpu[int(finished_item)]:
                    item_eos_hit_cpu[int(finished_item)] = True
            if next_ready_idx < num_items:
                new_item = int(next_ready_idx)
                next_ready_idx += 1
                load_item_to_slot(slot, new_item)
                swapped_slots.append(int(slot))
                swapped_in_item_ids.append(int(new_item))
            else:
                deactivate_slot(slot)
                deactivated_slots.append(int(slot))

        event = {
            "completion_decode_call": int(completion_decode_call),
            "finished_slots": [int(slot) for slot in finished_slots],
            "completed_item_ids": completed_item_ids,
            "swapped_slots": swapped_slots,
            "swapped_in_item_ids": swapped_in_item_ids,
            "deactivated_slots": deactivated_slots,
            "completed_count": int(completed_count),
            "remaining_ready_items": int(num_items - next_ready_idx),
        }
        if finished_slots:
            swap_events.append(event)
        if record is not None:
            record["swap_count"] = int(len(swapped_slots))
            record["finished_slot_count"] = int(len(finished_slots))
            record["deactivated_slot_count"] = int(len(deactivated_slots))
            record["completed_item_ids"] = completed_item_ids
            record["swapped_slots"] = swapped_slots
            record["swapped_in_item_ids"] = swapped_in_item_ids
        return event

    for slot in range(int(batch_size)):
        load_item_to_slot(slot, next_ready_idx)
        next_ready_idx += 1
    snapshot_step("after_initial_loads")

    # Handle the rare case where the prefill-selected first token is already EOS
    # or the run intentionally requests only one generated token.
    while True:
        initial_finished_flags = [bool(value) for value in slot_finished.detach().cpu().tolist()]
        if not any(initial_finished_flags):
            break
        initial_event = consume_finished_slots(initial_finished_flags, completion_decode_call=0)
        snapshot_step("after_initial_finished_consume", extra={"consume_event": initial_event})
        if completed_count >= num_items:
            break

    max_decode_calls_per_item = max(0, int(max_new_tokens) - 1)
    max_decode_call_budget = max(1, int(num_items) * max_decode_calls_per_item + max_decode_calls_per_item + int(batch_size))
    async_cpu_flags = None
    copy_stream = None
    if use_npu_overlap:
        async_cpu_flags = torch.zeros((max_decode_call_budget, int(batch_size)), dtype=torch.bool, pin_memory=True)
        copy_stream = torch_npu.npu.Stream(device=device)
    pending_flag_event = None
    pending_flag_row = None
    pending_finished_flags = None
    pending_completion_decode_call = None

    while completed_count < num_items:
        if decode_calls >= max_decode_call_budget:
            raise RuntimeError(
                f"hot-swap decode exceeded safety budget {max_decode_call_budget}; "
                f"completed {completed_count}/{num_items} items"
            )
        if not any(item_idx >= 0 for item_idx in active_item_indices_cpu):
            break

        record: dict[str, Any] = {
            "step": int(decode_calls),
            "decode_call": int(decode_calls) + 1,
            "swap_count": 0,
            "finished_slot_count": 0,
            "deactivated_slot_count": 0,
        }
        events: dict[str, tuple[Any, Any]] = {}
        host_iter_start = timing.cpu_now()
        npu_iter_start = timing.npu_event()

        if pending_completion_decode_call is not None and (
            (use_npu_overlap and pending_flag_event is not None and pending_flag_row is not None)
            or (not use_npu_overlap and pending_finished_flags is not None)
        ):
            if use_npu_overlap:
                assert async_cpu_flags is not None
                host_wait_start = timing.cpu_now()
                pending_flag_event.synchronize()
                wait_s = timing.cpu_elapsed_s(host_wait_start)
                if wait_s is not None:
                    record["host_wait_prev_flag_s"] = wait_s
                finished_flags = [bool(value) for value in async_cpu_flags[int(pending_flag_row)].tolist()]
            else:
                finished_flags = list(pending_finished_flags)
            host_swap_start = timing.cpu_now()
            npu_swap_start = timing.npu_event()
            consume_event = consume_finished_slots(finished_flags, int(pending_completion_decode_call), record)
            npu_swap_end = timing.npu_event()
            swap_s = timing.cpu_elapsed_s(host_swap_start)
            if swap_s is not None:
                record["host_swap_s"] = swap_s
            events["npu_swap"] = (npu_swap_start, npu_swap_end)
            snapshot_step(
                "after_pending_finished_consume",
                extra={
                    "pending_completion_decode_call": int(pending_completion_decode_call),
                    "finished_flags": [bool(value) for value in finished_flags],
                    "consume_event": consume_event,
                },
            )
            pending_flag_event = None
            pending_flag_row = None
            pending_finished_flags = None
            pending_completion_decode_call = None
            if completed_count >= num_items:
                npu_iter_end = timing.npu_event()
                iter_s = timing.cpu_elapsed_s(host_iter_start)
                if iter_s is not None:
                    record["host_iter_s"] = iter_s
                events["npu_iter"] = (npu_iter_start, npu_iter_end)
                timing.add(record, events)
                break

        active_before_step = active_mask & ~slot_finished
        active_before_step_cpu = [
            int(item_idx) >= 0 and not bool(slot_finished_cpu[slot])
            for slot, item_idx in enumerate(active_item_indices_cpu)
        ]
        snapshot_step(
            "before_decode",
            extra={
                "next_decode_call": int(decode_calls) + 1,
                "active_before_step_cpu": [bool(value) for value in active_before_step_cpu],
            },
        )
        host_decode_start = timing.cpu_now()
        npu_decode_start = timing.npu_event()
        last_logits = decode_fn(active_next_token, active.next_cache_position, active.rope_deltas, *flat_cache)
        npu_decode_end = timing.npu_event()
        decode_enqueue_s = timing.cpu_elapsed_s(host_decode_start)
        if decode_enqueue_s is not None:
            record["host_decode_enqueue_s"] = decode_enqueue_s
        events["npu_decode"] = (npu_decode_start, npu_decode_end)

        sampled_token = torch.argmax(last_logits[:, -1, :].float(), dim=-1, keepdim=True)
        eos_fill = torch.full_like(sampled_token, int(eos_token_id))
        active_next_token.copy_(torch.where(active_before_step.view(-1, 1), sampled_token, eos_fill))

        # Keep output history writes host-indexed. NPU boolean advanced
        # indexing here can accidentally involve inactive -1 slots.
        length_cap_slots = []
        for slot, should_write in enumerate(active_before_step_cpu):
            if not should_write:
                continue
            item_idx = int(active_item_indices_cpu[slot])
            position = int(generated_lengths_cpu[item_idx])
            if position >= int(max_new_tokens):
                continue
            copy_vector_position_(generated_rows[item_idx], position, active_next_token[slot : slot + 1])
            new_length = position + 1
            generated_lengths_cpu[item_idx] = new_length
            if new_length >= int(max_new_tokens):
                length_cap_slots.append(int(slot))
                slot_finished_cpu[int(slot)] = True
                item_length_cap_hit_cpu[item_idx] = True
        for slot in length_cap_slots:
            fill_batch_row_(slot_finished, int(slot), True)

        new_eos_by_slot = active_before_step & hits_eos(active_next_token, int(eos_token_id))
        slot_finished.logical_or_(new_eos_by_slot)
        active.next_cache_position.add_(active_before_step.to(dtype=active.next_cache_position.dtype))
        decode_calls += 1
        snapshot_step(
            "after_decode_write",
            extra={
                "decode_call": int(decode_calls),
                "active_before_step_cpu": [bool(value) for value in active_before_step_cpu],
                "sampled_token": [int(row[0]) for row in sampled_token.detach().cpu().tolist()],
                "written_next_token": [int(row[0]) for row in active_next_token.detach().cpu().tolist()],
                "length_cap_slots": [int(slot) for slot in length_cap_slots],
                "new_eos_by_slot": [bool(value) for value in new_eos_by_slot.detach().cpu().tolist()],
            },
        )

        if use_npu_overlap:
            assert async_cpu_flags is not None
            assert copy_stream is not None
            host_flag_copy_start = timing.cpu_now()
            flag_ready_event = torch_npu.npu.current_stream().record_event()
            copy_done_event = torch_npu.npu.Event()
            npu_flag_copy_start = None
            npu_flag_copy_end = None
            with torch_npu.npu.stream(copy_stream):
                copy_stream.wait_event(flag_ready_event)
                npu_flag_copy_start = timing.npu_event(copy_stream)
                async_cpu_flags[int(decode_calls) - 1].copy_(slot_finished, non_blocking=True)
                npu_flag_copy_end = timing.npu_event(copy_stream)
                copy_done_event.record(copy_stream)
            flag_copy_enqueue_s = timing.cpu_elapsed_s(host_flag_copy_start)
            if flag_copy_enqueue_s is not None:
                record["host_flag_copy_enqueue_s"] = flag_copy_enqueue_s
            events["npu_flag_copy"] = (npu_flag_copy_start, npu_flag_copy_end)
            pending_flag_event = copy_done_event
            pending_flag_row = int(decode_calls) - 1
        else:
            pending_finished_flags = [bool(value) for value in slot_finished.detach().cpu().tolist()]
        pending_completion_decode_call = int(decode_calls)

        npu_iter_end = timing.npu_event()
        iter_s = timing.cpu_elapsed_s(host_iter_start)
        if iter_s is not None:
            record["host_iter_s"] = iter_s
        events["npu_iter"] = (npu_iter_start, npu_iter_end)
        timing.add(record, events)

    if completed_count < num_items and pending_completion_decode_call is not None and (
        (use_npu_overlap and pending_flag_event is not None and pending_flag_row is not None)
        or (not use_npu_overlap and pending_finished_flags is not None)
    ):
        if use_npu_overlap:
            assert async_cpu_flags is not None
            pending_flag_event.synchronize()
            finished_flags = [bool(value) for value in async_cpu_flags[int(pending_flag_row)].tolist()]
        else:
            finished_flags = list(pending_finished_flags)
        final_event = consume_finished_slots(finished_flags, int(pending_completion_decode_call))
        snapshot_step(
            "after_final_finished_consume",
            extra={
                "pending_completion_decode_call": int(pending_completion_decode_call),
                "finished_flags": [bool(value) for value in finished_flags],
                "consume_event": final_event,
            },
        )

    generated_ids = torch.stack(generated_rows, dim=0)
    generated_lengths = torch.tensor(generated_lengths_cpu, device=device, dtype=torch.int64)
    snapshot_step("final")

    return HotSwapDecodeResult(
        ids=generated_ids,
        lengths=generated_lengths,
        last_logits=last_logits,
        decode_calls=int(decode_calls),
        completed_by_item=completed_by_item,
        completion_decode_calls=completion_decode_calls,
        eos_hit_by_item=item_eos_hit_cpu,
        length_cap_hit_by_item=item_length_cap_hit_cpu,
        swap_events=swap_events,
        step_timing=timing.records,
        stopped_all_items=bool(completed_count >= num_items),
        diagnostics={
            "decode_attention_mode": decode_attention_label(device),
            "decode_cache_update_mode": decode_cache_update_label(device),
            "swap_copy_mode": diagnostic_swap_copy_mode,
            "verify_swap_copies": bool(diagnostic_verify_swap_copies),
            "sync_finished_flags": bool(diagnostic_sync_finished_flags),
            "history_storage": "per_item_device_rows",
            "slot_control_write_mode": "index_copy_on_npu_slice_copy_elsewhere",
            "deactivated_slot_state": "eos_token_cache_pos_0_rope_delta_0",
            "step_trace_enabled": bool(diagnostic_step_trace or trace_item_indices),
            "step_trace_full_generated_ids": bool(diagnostic_step_trace),
            "step_trace_items": [int(value) for value in trace_item_indices],
            "step_trace_syncs_device": bool(diagnostic_step_trace or trace_item_indices),
            "step_trace_print": bool(diagnostic_step_trace_print),
            "copy_verification_checks": int(copy_verification_checks),
            "copy_verification_failure_count": int(len(copy_verification_failures)),
            "copy_verification_failures": copy_verification_failures,
        },
        diagnostic_step_trace=step_trace if (diagnostic_step_trace or trace_item_indices) else None,
        timing_recorder=timing if step_timing != "off" else None,
    )


@torch.inference_mode()
def compare_static_logits(
    eager_decode: Callable,
    compiled_decode: Callable,
    eager_prefill: BatchedPrefill,
    compiled_prefill: BatchedPrefill,
    eager_next_token: torch.Tensor,
    compiled_next_token: torch.Tensor,
    *,
    max_new_tokens: int,
):
    max_abs = 0.0
    mean_abs_sum = 0.0
    compared_steps = 0
    eager_generated = [eager_next_token]
    compiled_generated = [compiled_next_token]
    eager_cache_position = eager_prefill.next_cache_position
    compiled_cache_position = compiled_prefill.next_cache_position
    eager_flat_cache = eager_prefill.cache.flat_tensors()
    compiled_flat_cache = compiled_prefill.cache.flat_tensors()
    for _ in range(max(0, int(max_new_tokens) - 1)):
        eager_logits = eager_decode(eager_next_token, eager_cache_position, eager_prefill.rope_deltas, *eager_flat_cache)
        compiled_logits = compiled_decode(
            compiled_next_token,
            compiled_cache_position,
            compiled_prefill.rope_deltas,
            *compiled_flat_cache,
        )
        diff = (eager_logits.float() - compiled_logits.float()).abs()
        max_abs = max(max_abs, float(diff.max()))
        mean_abs_sum += float(diff.mean())
        compared_steps += 1
        eager_next_token = torch.argmax(eager_logits[:, -1, :].float(), dim=-1, keepdim=True)
        compiled_next_token = torch.argmax(compiled_logits[:, -1, :].float(), dim=-1, keepdim=True)
        eager_generated.append(eager_next_token)
        compiled_generated.append(compiled_next_token)
        eager_cache_position = eager_cache_position + 1
        compiled_cache_position = compiled_cache_position + 1
    mean_abs = mean_abs_sum / compared_steps if compared_steps else 0.0
    return torch.cat(eager_generated, dim=1), torch.cat(compiled_generated, dim=1), max_abs, mean_abs, compared_steps


@torch.inference_mode()
def compare_batched_to_single_refs(
    *,
    flat_decode: Callable,
    slot_prefills: list[Any],
    slot_next_tokens: list[torch.Tensor],
    batched_ids: torch.Tensor,
    max_new_tokens: int,
    eos_mode: str,
    eos_token_id: int,
) -> dict[str, Any]:
    single_rows = []
    single_trimmed_rows = []
    for slot_prefill, slot_next_token in zip(slot_prefills, slot_next_tokens):
        single_prefill = BatchedPrefill(
            cache=slot_prefill.cache,
            rope_deltas=slot_prefill.rope_deltas,
            next_cache_position=slot_prefill.next_cache_position,
        )
        single_result = static_flat_decode_loop(
            flat_decode,
            single_prefill,
            slot_next_token,
            max_new_tokens=max_new_tokens,
            eos_mode=eos_mode,
            eos_token_id=eos_token_id,
        )
        single_rows.append(row_token_lists(single_result.ids)[0])
        single_trimmed_rows.append(row_trimmed_token_lists(single_result.ids, eos_token_id)[0])

    batched_rows = row_token_lists(batched_ids)
    batched_trimmed_rows = row_trimmed_token_lists(batched_ids, eos_token_id)
    full_matches = [batch_row == single_row for batch_row, single_row in zip(batched_rows, single_rows)]
    trimmed_matches = [
        batch_row == single_row
        for batch_row, single_row in zip(batched_trimmed_rows, single_trimmed_rows)
    ]
    first_mismatches = []
    for row_idx, (batch_row, single_row) in enumerate(zip(batched_trimmed_rows, single_trimmed_rows)):
        if batch_row != single_row and len(first_mismatches) < 8:
            first_mismatches.append(
                {
                    "row": int(row_idx),
                    "batched": [int(value) for value in batch_row],
                    "single_ref": [int(value) for value in single_row],
                    "comparison": token_comparison_summary(batch_row, single_row),
                }
            )
    return {
        "full_matches_by_row": full_matches,
        "trimmed_matches_by_row": trimmed_matches,
        "all_full_match": bool(all(full_matches)),
        "all_trimmed_match": bool(all(trimmed_matches)),
        "first_mismatches": first_mismatches,
    }


def tok_per_s(tokens: int, seconds: float) -> float:
    return float(tokens) / float(seconds) if seconds > 0 else float("inf")


def npu_profiler_config(metric: str):
    import torch_npu.profiler as npu_prof

    metrics = {
        "pipe": npu_prof.AiCMetrics.PipeUtilization,
        "memory": npu_prof.AiCMetrics.Memory,
        "l2": npu_prof.AiCMetrics.L2Cache,
        "memory_access": npu_prof.AiCMetrics.MemoryAccess,
    }
    return npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1,
        aic_metrics=metrics[metric],
        l2_cache=metric == "l2",
        export_type=npu_prof.ExportType.Text,
    )


def make_profile_run_dir(root: Path, *, backend: str, eos_mode: str, batch_size: int, max_new_tokens: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return root / f"bench_batched_decode_{timestamp}_{backend}_{DECODE_ATTENTION}_{eos_mode}_bs{batch_size}_{max_new_tokens}tok"


@torch.inference_mode()
def profile_compiled_decode(
    *,
    args: argparse.Namespace,
    model: LocalPaddleOCRVLForConditionalGeneration,
    compiled_decode: Callable,
    cohort: list[CohortInput],
    cache_length: int,
    tokenizer: Tokenizer,
    device: torch.device,
) -> dict[str, Any]:
    if device.type != "npu":
        raise ValueError("--profile-dir requires --device npu:0; torch_npu profiler is NPU-only.")
    if args.backend != "torchair":
        raise ValueError("--profile-dir requires --backend torchair so the profile captures compiled NPU decode.")
    if int(args.max_new_tokens) >= 16:
        raise ValueError("--profile-dir requires --max-new-tokens < 16 to keep profiler JSON bounded.")

    import torch_npu.profiler as npu_prof

    profile_dir = make_profile_run_dir(
        args.profile_dir.expanduser().resolve(),
        backend=args.backend,
        eos_mode=args.eos_mode,
        batch_size=len(cohort),
        max_new_tokens=int(args.max_new_tokens),
    )
    shutil.rmtree(profile_dir, ignore_errors=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    warm_cohort = cohort[: int(args.batch_size)]
    warm_prefill, warm_next_token, _warm_slots, _warm_nexts = make_batched_prefill(
        model=model,
        cohort=warm_cohort,
        cache_length=cache_length,
        device=device,
    )
    _profile_warmup_result, profile_warmup_s = timed(
        device,
        lambda: static_flat_decode_loop(
            compiled_decode,
            warm_prefill,
            warm_next_token,
            max_new_tokens=args.max_new_tokens,
            eos_mode=args.eos_mode,
            eos_token_id=int(model.config.eos_token_id),
        ),
    )

    prof_prefill, prof_next_token, _prof_slots, _prof_nexts = make_batched_prefill(
        model=model,
        cohort=cohort,
        cache_length=cache_length,
        device=device,
    )
    schedule = npu_prof.schedule(wait=0, warmup=0, active=1, repeat=1)
    maybe_sync(device)
    start = time.perf_counter()
    with npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        schedule=schedule,
        experimental_config=npu_profiler_config(args.profile_metric),
        on_trace_ready=npu_prof.tensorboard_trace_handler(str(profile_dir), analyse_flag=True),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    ) as profiler:
        with torch.profiler.record_function("paddle_ocr_vl.batched_compiled_decode_profile"):
            profile_result = static_flat_decode_loop(
                compiled_decode,
                prof_prefill,
                prof_next_token,
                max_new_tokens=args.max_new_tokens,
                eos_mode=args.eos_mode,
                eos_token_id=int(model.config.eos_token_id),
            )
        maybe_sync(device)
        profiler.step()
    maybe_sync(device)
    profile_wall_s = time.perf_counter() - start

    profile_summary = {
        "profile_dir": str(profile_dir),
        "metric": args.profile_metric,
        "batch_size": int(len(cohort)),
        "linear_weight_format": DECODE_LINEAR_WEIGHT_FORMAT,
        "decode_attention": decode_attention_label(device),
        "eos_mode": args.eos_mode,
        "with_stack": True,
        "record_shapes": True,
        "profile_memory": False,
        "profile_warmup_s": float(profile_warmup_s),
        "profile_wall_s": float(profile_wall_s),
        "profiled_generated_tokens_per_row": int(args.max_new_tokens),
        "profiled_decode_steps": int(profile_result.decode_calls),
        "loop": decode_loop_summary(profile_result, eos_token_id=int(model.config.eos_token_id)),
        "generated_ids": row_token_lists(profile_result.ids),
        "generated_texts": [
            tokenizer.decode(row, skip_special_tokens=True)
            for row in row_token_lists(profile_result.ids)
        ],
    }
    (profile_dir / "bench_profile_summary.json").write_text(json.dumps(profile_summary, indent=2, default=json_default), encoding="utf-8")
    return profile_summary


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6")
    parser.add_argument("--manifest", type=Path, default=Path("crops") / "manifest.json")
    parser.add_argument("--schedule", default="fixed_cohort", choices=SCHEDULE_CHOICES)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-items", type=int, default=None, help="Number of manifest items for --schedule hotswap. Defaults to the full manifest.")
    parser.add_argument("--crop-ids", nargs="*", default=None, help="Optional explicit manifest ids. Fixed cohort requires count == --batch-size; hot-swap requires count >= --batch-size.")
    parser.add_argument("--prompt", default=None, help="Optional prompt override. By default, each crop uses its manifest suggested_prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cache-length", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--backend", default="eager", choices=["raw_eager", "eager", "aot_eager", "inductor", "default", "torchair"])
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--torchair-cache-dir", type=Path, default=DEFAULT_TORCHAIR_CACHE_DIR, help="TorchAir GE cache root. Reuse this for warm runs; use a fresh path after model code changes.")
    parser.add_argument("--eos-mode", default="none", choices=EOS_MODE_CHOICES)
    parser.add_argument("--step-timing", default="off", choices=STEP_TIMING_CHOICES, help="Use 'both' for NPU speed runs; JSON output includes per-step host/NPU decode, swap, and flag-copy timing.")
    parser.add_argument(
        "--diagnostic-swap-copy-mode",
        default="direct",
        choices=DIAGNOSTIC_SWAP_COPY_CHOICES,
        help="Diagnostic only: clone ready-bank rows before copying them into active slots.",
    )
    parser.add_argument(
        "--diagnostic-verify-swap-copies",
        action="store_true",
        help="Diagnostic only: compare active slot rows against ready-bank rows after each load/swap.",
    )
    parser.add_argument(
        "--diagnostic-sync-finished-flags",
        action="store_true",
        help="Diagnostic only: on NPU, read finished-slot flags synchronously instead of using the overlap copy stream.",
    )
    parser.add_argument(
        "--diagnostic-trace-items",
        nargs="*",
        type=int,
        default=None,
        help="Diagnostic only for --schedule hotswap: emit token traces for selected 0-indexed item indices.",
    )
    parser.add_argument(
        "--diagnostic-step-trace",
        action="store_true",
        help="Diagnostic only for --schedule hotswap: snapshot the full generated_ids matrix at every hot-swap loop stage.",
    )
    parser.add_argument(
        "--diagnostic-step-trace-items",
        nargs="*",
        type=int,
        default=None,
        help="Diagnostic only for --schedule hotswap: snapshot selected generated_ids rows at every hot-swap loop stage.",
    )
    parser.add_argument(
        "--diagnostic-step-trace-print",
        action="store_true",
        help="Diagnostic only for --schedule hotswap: print each step-trace snapshot immediately as a JSON line.",
    )
    parser.add_argument("--profile-dir", type=Path, default=None, help="Fixed-cohort only: write one post-warmup torch_npu profiler capture for compiled batched decode with --max-new-tokens < 16.")
    parser.add_argument("--profile-metric", default="pipe", choices=PROFILE_METRIC_CHOICES)
    parser.add_argument("--report", default="full", choices=REPORT_CHOICES, help="Use --report summary --json for a pasteable correctness and speed report without post-processing scripts.")
    parser.add_argument("--json", action="store_true", help="Print a compact JSON summary instead of human-readable lines.")
    args = parser.parse_args()

    if int(args.max_new_tokens) <= 0:
        raise ValueError(f"--max-new-tokens must be positive, got {args.max_new_tokens}")
    if args.schedule == "hotswap" and args.eos_mode != "overlap_event_flags":
        raise ValueError("--schedule hotswap requires --eos-mode overlap_event_flags")
    if args.schedule == "hotswap" and args.profile_dir is not None:
        raise ValueError("--profile-dir currently profiles fixed-cohort decode only; run hot-swap without profiler first.")
    if args.schedule != "hotswap" and args.diagnostic_trace_items:
        raise ValueError("--diagnostic-trace-items is currently available for --schedule hotswap only")
    if args.schedule != "hotswap" and args.diagnostic_step_trace:
        raise ValueError("--diagnostic-step-trace is currently available for --schedule hotswap only")
    if args.schedule != "hotswap" and args.diagnostic_step_trace_items:
        raise ValueError("--diagnostic-step-trace-items is currently available for --schedule hotswap only")
    if args.schedule != "hotswap" and args.diagnostic_step_trace_print:
        raise ValueError("--diagnostic-step-trace-print is currently available for --schedule hotswap only")
    if args.diagnostic_step_trace_print and args.json:
        raise ValueError("--diagnostic-step-trace-print writes live stdout lines; run without --json")
    if args.diagnostic_step_trace_print and not args.diagnostic_step_trace and not args.diagnostic_step_trace_items:
        raise ValueError("--diagnostic-step-trace-print requires --diagnostic-step-trace or --diagnostic-step-trace-items")

    model_dir = _resolve_model_dir(args.model)
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    configure_npu_jit_compile(args.npu_jit_compile, device)

    pre_cfg = load_preprocessor_config(model_dir)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    manifest = load_manifest(args.manifest)
    entries = select_manifest_entries(
        manifest,
        batch_size=int(args.batch_size),
        num_items=args.num_items,
        crop_ids=args.crop_ids,
        schedule=args.schedule,
    )
    cohort = build_cohort_inputs(
        entries=entries,
        manifest_path=args.manifest,
        tokenizer=tokenizer,
        pre_cfg=pre_cfg,
        prompt_override=args.prompt,
    )
    prompt_tokens = [int(item.input_ids.shape[1]) for item in cohort]
    cache_length = int(args.cache_length or (max(prompt_tokens) + int(args.max_new_tokens)))
    if cache_length < max(prompt_tokens):
        raise ValueError(f"--cache-length={cache_length} is smaller than max prompt length {max(prompt_tokens)}")

    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, device=device)
    eos_token_id = int(model.config.eos_token_id)
    maybe_sync(device)
    weight_format_start = time.perf_counter()
    weight_format_meta = cast_decode_linear_weights_to_nz(model)
    maybe_sync(device)
    weight_format_meta["setup_s"] = time.perf_counter() - weight_format_start
    flat_decode = model.make_flat_static_decode_module().eval()

    maybe_sync(device)
    compile_start = time.perf_counter()
    compiled_decode, compile_meta = compile_decode_module(
        flat_decode,
        backend_name=args.backend,
        device=device,
        cache_root=args.torchair_cache_dir,
        batch_size=int(args.batch_size),
        cache_length=cache_length,
    )
    maybe_sync(device)
    compile_wrapper_s = time.perf_counter() - compile_start

    warm_cohort = cohort[: int(args.batch_size)]
    warm_prefill, warm_next_token, _warm_slots, _warm_nexts = make_batched_prefill(
        model=model,
        cohort=warm_cohort,
        cache_length=cache_length,
        device=device,
    )
    _, compile_first_s = timed(
        device,
        lambda: compiled_decode(
            warm_next_token,
            warm_prefill.next_cache_position,
            warm_prefill.rope_deltas,
            *warm_prefill.cache.flat_tensors(),
        ),
    )

    if args.schedule == "hotswap":
        ready_bank_result, ready_bank_prefill_s = timed(
            device,
            lambda: make_ready_bank(
                model=model,
                cohort=cohort,
                cache_length=cache_length,
                device=device,
            ),
        )
        ready_bank, _ready_slot_prefills = ready_bank_result
        hotswap_result, hotswap_decode_s = timed(
            device,
            lambda: static_hotswap_decode_loop(
                compiled_decode,
                ready_bank,
                batch_size=int(args.batch_size),
                max_new_tokens=int(args.max_new_tokens),
                eos_mode=args.eos_mode,
                eos_token_id=eos_token_id,
                step_timing=args.step_timing,
                diagnostic_swap_copy_mode=args.diagnostic_swap_copy_mode,
                diagnostic_verify_swap_copies=bool(args.diagnostic_verify_swap_copies),
                diagnostic_sync_finished_flags=bool(args.diagnostic_sync_finished_flags),
                diagnostic_step_trace=bool(args.diagnostic_step_trace),
                diagnostic_step_trace_items=[int(value) for value in args.diagnostic_step_trace_items] if args.diagnostic_step_trace_items else None,
                diagnostic_step_trace_print=bool(args.diagnostic_step_trace_print),
            ),
        )
        finalize_step_timing(hotswap_result)
        hotswap_validation, hotswap_validation_s = timed(
            device,
            lambda: validate_hotswap_against_single_refs(
                flat_decode=flat_decode,
                ready=ready_bank,
                hotswap_ids=hotswap_result.ids,
                hotswap_lengths=hotswap_result.lengths,
                max_new_tokens=int(args.max_new_tokens),
                eos_token_id=eos_token_id,
            ),
        )
        diagnostic_item_traces = []
        diagnostic_item_trace_s = 0.0
        if args.diagnostic_trace_items:
            diagnostic_item_traces, diagnostic_item_trace_s = timed(
                device,
                lambda: diagnose_hotswap_item_traces(
                    model=model,
                    flat_decode=flat_decode,
                    ready=ready_bank,
                    cohort=cohort,
                    hotswap_ids=hotswap_result.ids,
                    hotswap_lengths=hotswap_result.lengths,
                    item_indices=[int(value) for value in args.diagnostic_trace_items],
                    cache_length=cache_length,
                    max_new_tokens=int(args.max_new_tokens),
                    eos_token_id=eos_token_id,
                    device=device,
                    tokenizer=tokenizer,
                ),
            )
        hotswap_rows = length_trimmed_token_lists(hotswap_result.ids, hotswap_result.lengths)
        hotswap_token_ids = token_id_range_summary(
            hotswap_result.ids,
            lengths=hotswap_result.lengths,
            vocab_size=int(model.config.vocab_size),
        )
        hotswap_loop = hotswap_loop_summary(hotswap_result, batch_size=int(args.batch_size))
        raw_token_calls = int(hotswap_loop["raw_decode_token_calls"])
        effective_token_calls = int(hotswap_loop["effective_decode_token_calls"])
        hotswap_timing_accounting = step_timing_accounting(
            hotswap_result.step_timing,
            wall_s=hotswap_decode_s,
            batch_size=int(args.batch_size),
            raw_token_calls=raw_token_calls,
            effective_token_calls=effective_token_calls,
        )
        hotswap_required_checks_passed = bool(
            hotswap_validation["all_trimmed_match"]
            and int(hotswap_token_ids["invalid_count"]) == 0
        )
        summary = {
            "experiment": "04_batched_fixed_cohort_decode",
            "schedule": args.schedule,
            "backend": args.backend,
            "device": str(device),
            "dtype": str(dtype),
            "npu_jit_compile": args.npu_jit_compile,
            "decode_attention": decode_attention_label(device),
            "default_decode_attention": DECODE_ATTENTION,
            "eos_mode": args.eos_mode,
            "step_timing_mode": args.step_timing,
            "eos_token_id": eos_token_id,
            "batch_size": int(args.batch_size),
            "num_items": int(len(cohort)),
            "cohort": [
                {
                    "item": idx,
                    "id": str(item.entry.get("id")),
                    "file": str(item.crop_path),
                    "prompt": item.prompt,
                    "prompt_tokens": int(item.input_ids.shape[1]),
                    "category_type": item.entry.get("category_type"),
                }
                for idx, item in enumerate(cohort)
            ],
            "linear_weight_format": weight_format_meta,
            "compile": compile_meta,
            "cache_update": f"prefill_slice_decode_{decode_cache_update_label(device)}",
            "decode_cache_update": decode_cache_update_label(device),
            "history_write_mode": "per_item_device_rows",
            "slot_control_write_mode": "index_copy_on_npu_slice_copy_elsewhere",
            "prompt_tokens": {
                "per_item": prompt_tokens,
                "min": int(min(prompt_tokens)),
                "max": int(max(prompt_tokens)),
            },
            "token_ids": hotswap_token_ids,
            "generated_tokens_per_item": int(args.max_new_tokens),
            "cache_length": int(cache_length),
            "loop": {
                "hotswap": hotswap_loop,
            },
            "matches": {
                "hotswap_vs_single_refs": hotswap_validation,
                "all_required_checks_passed": hotswap_required_checks_passed,
            },
            "timing_s": {
                "compile_wrapper": float(compile_wrapper_s),
                "compile_first_call": float(compile_first_s),
                "ready_bank_prefill": float(ready_bank_prefill_s),
                "hotswap_decode": float(hotswap_decode_s),
                "hotswap_validation": float(hotswap_validation_s),
                "diagnostic_item_trace": float(diagnostic_item_trace_s),
            },
            "timing_accounting": hotswap_timing_accounting,
            "tok_per_s": {
                "hotswap_decode_steps": tok_per_s(hotswap_result.decode_calls, hotswap_decode_s),
                "hotswap_raw_batch_tokens": tok_per_s(raw_token_calls, hotswap_decode_s),
                "hotswap_effective_item_tokens": tok_per_s(effective_token_calls, hotswap_decode_s),
            },
            "swap_events": hotswap_result.swap_events,
            "step_timing": hotswap_result.step_timing,
            "diagnostics": hotswap_result.diagnostics,
            "diagnostic_step_trace": hotswap_result.diagnostic_step_trace or [],
            "diagnostic_item_traces": diagnostic_item_traces,
            "texts": {
                "hotswap_trimmed": (
                    [
                        safe_decode_tokens(tokenizer, row)
                        for row in hotswap_rows
                    ]
                    if int(hotswap_token_ids["invalid_count"]) == 0
                    else []
                ),
            },
        }
        if args.json:
            output = benchmark_report(summary) if args.report == "summary" else summary
            print(json.dumps(output, indent=2, sort_keys=True, default=json_default))
            if not hotswap_required_checks_passed:
                raise SystemExit(1)
            return

        if args.report == "summary":
            print("benchmark_report=" + json.dumps(benchmark_report(summary), sort_keys=True, default=json_default))
            if not hotswap_required_checks_passed:
                raise SystemExit(1)
            return

        print(
            f"experiment={summary['experiment']} schedule={summary['schedule']} backend={summary['backend']} "
            f"device={summary['device']} dtype={summary['dtype']} batch_size={summary['batch_size']} "
            f"num_items={summary['num_items']} npu_jit_compile={summary['npu_jit_compile']} "
            f"decode_attention={summary['decode_attention']} eos_mode={summary['eos_mode']}"
        )
        print("linear_weight_format=" + json.dumps(summary["linear_weight_format"], sort_keys=True))
        print("cache_update=" + summary["cache_update"])
        print("decode_cache_update=" + summary["decode_cache_update"])
        print("history_write_mode=" + summary["history_write_mode"])
        print("slot_control_write_mode=" + summary["slot_control_write_mode"])
        print(
            f"prompt_tokens={{'min': {summary['prompt_tokens']['min']}, 'max': {summary['prompt_tokens']['max']}}} "
            f"generated_tokens_per_item={summary['generated_tokens_per_item']} cache_length={summary['cache_length']}"
        )
        print("loop=" + json.dumps(summary["loop"], sort_keys=True))
        print("matches=" + json.dumps(summary["matches"], sort_keys=True))
        print("timing_s=" + json.dumps(summary["timing_s"], sort_keys=True))
        print("timing_accounting=" + json.dumps(summary["timing_accounting"], sort_keys=True))
        print("tok_per_s=" + json.dumps(summary["tok_per_s"], sort_keys=True))
        print("step_timing_summary=" + json.dumps(hotswap_loop["step_timing_summary"], sort_keys=True))
        print("swap_events_sample=" + json.dumps(summary["swap_events"][:8], sort_keys=True))
        if diagnostic_item_traces:
            print("diagnostic_item_traces=" + json.dumps(diagnostic_item_traces, sort_keys=True))
        if hotswap_result.diagnostic_step_trace:
            print("diagnostic_step_trace=" + json.dumps(hotswap_result.diagnostic_step_trace, sort_keys=True))
        print("hotswap_texts_sample=" + repr(summary["texts"]["hotswap_trimmed"][: min(8, len(hotswap_rows))]))
        if not hotswap_required_checks_passed:
            raise SystemExit(1)
        return

    static_prefill, static_next_token, static_slot_prefills, static_slot_next_tokens = make_batched_prefill(
        model=model,
        cohort=cohort,
        cache_length=cache_length,
        device=device,
    )
    static_result, static_decode_s = timed(
        device,
        lambda: static_flat_decode_loop(
            flat_decode,
            static_prefill,
            static_next_token,
            max_new_tokens=args.max_new_tokens,
            eos_mode=args.eos_mode,
            eos_token_id=eos_token_id,
            step_timing=args.step_timing,
        ),
    )
    finalize_step_timing(static_result)
    static_ids = static_result.ids

    single_ref_matches = compare_batched_to_single_refs(
        flat_decode=flat_decode,
        slot_prefills=static_slot_prefills,
        slot_next_tokens=static_slot_next_tokens,
        batched_ids=static_ids,
        max_new_tokens=args.max_new_tokens,
        eos_mode=args.eos_mode,
        eos_token_id=eos_token_id,
    )

    compiled_prefill, compiled_next_token, _compiled_slots, _compiled_nexts = make_batched_prefill(
        model=model,
        cohort=cohort,
        cache_length=cache_length,
        device=device,
    )
    compiled_result, compiled_decode_s = timed(
        device,
        lambda: static_flat_decode_loop(
            compiled_decode,
            compiled_prefill,
            compiled_next_token,
            max_new_tokens=args.max_new_tokens,
            eos_mode=args.eos_mode,
            eos_token_id=eos_token_id,
            step_timing=args.step_timing,
        ),
    )
    finalize_step_timing(compiled_result)
    compiled_ids = compiled_result.ids

    compare_eager_prefill, compare_eager_next, _compare_eager_slots, _compare_eager_nexts = make_batched_prefill(
        model=model,
        cohort=cohort,
        cache_length=cache_length,
        device=device,
    )
    compare_compiled_prefill, compare_compiled_next, _compare_compiled_slots, _compare_compiled_nexts = make_batched_prefill(
        model=model,
        cohort=cohort,
        cache_length=cache_length,
        device=device,
    )
    compare_static_ids, compare_compiled_ids, max_abs, mean_abs, compared_steps = compare_static_logits(
        flat_decode,
        compiled_decode,
        compare_eager_prefill,
        compare_compiled_prefill,
        compare_eager_next,
        compare_compiled_next,
        max_new_tokens=args.max_new_tokens,
    )

    profile_summary = None
    if args.profile_dir is not None:
        profile_summary = profile_compiled_decode(
            args=args,
            model=model,
            compiled_decode=compiled_decode,
            cohort=cohort,
            cache_length=cache_length,
            tokenizer=tokenizer,
            device=device,
        )

    static_trimmed_rows = row_trimmed_token_lists(static_ids, eos_token_id)
    compiled_trimmed_rows = row_trimmed_token_lists(compiled_ids, eos_token_id)
    static_loop_summary = decode_loop_summary(static_result, eos_token_id=eos_token_id)
    compiled_loop_summary = decode_loop_summary(compiled_result, eos_token_id=eos_token_id)
    static_raw_token_calls = int(static_loop_summary["raw_decode_token_calls"])
    compiled_raw_token_calls = int(compiled_loop_summary["raw_decode_token_calls"])
    static_effective_token_calls = int(static_loop_summary["effective_decode_token_calls"])
    compiled_effective_token_calls = int(compiled_loop_summary["effective_decode_token_calls"])
    static_timing_accounting = step_timing_accounting(
        static_result.step_timing,
        wall_s=static_decode_s,
        batch_size=int(args.batch_size),
        raw_token_calls=static_raw_token_calls,
        effective_token_calls=static_effective_token_calls,
    )
    compiled_timing_accounting = step_timing_accounting(
        compiled_result.step_timing,
        wall_s=compiled_decode_s,
        batch_size=int(args.batch_size),
        raw_token_calls=compiled_raw_token_calls,
        effective_token_calls=compiled_effective_token_calls,
    )
    single_ref_required_match = bool(single_ref_matches["all_trimmed_match"])
    fixed_required_checks_passed = bool(
        torch.equal(static_ids, compiled_ids)
        and static_trimmed_rows == compiled_trimmed_rows
        and torch.equal(compare_static_ids, compare_compiled_ids)
        and single_ref_required_match
    )
    summary = {
        "experiment": "04_batched_fixed_cohort_decode",
        "backend": args.backend,
        "device": str(device),
        "dtype": str(dtype),
        "npu_jit_compile": args.npu_jit_compile,
        "decode_attention": decode_attention_label(device),
        "default_decode_attention": DECODE_ATTENTION,
        "eos_mode": args.eos_mode,
        "step_timing_mode": args.step_timing,
        "eos_token_id": eos_token_id,
        "batch_size": int(args.batch_size),
        "cohort": [
            {
                "row": idx,
                "id": str(item.entry.get("id")),
                "file": str(item.crop_path),
                "prompt": item.prompt,
                "prompt_tokens": int(item.input_ids.shape[1]),
                "category_type": item.entry.get("category_type"),
            }
            for idx, item in enumerate(cohort)
        ],
        "linear_weight_format": weight_format_meta,
        "compile": compile_meta,
        "cache_update": f"prefill_slice_decode_{decode_cache_update_label(device)}",
        "decode_cache_update": decode_cache_update_label(device),
        "prompt_tokens": {
            "per_row": prompt_tokens,
            "min": int(min(prompt_tokens)),
            "max": int(max(prompt_tokens)),
        },
        "generated_tokens_per_row": int(args.max_new_tokens),
        "requested_decode_steps": max(0, int(args.max_new_tokens) - 1),
        "cache_length": int(cache_length),
        "loop": {
            "static_eager": static_loop_summary,
            "compiled": compiled_loop_summary,
        },
        "matches": {
            "static_eager_vs_compiled": bool(torch.equal(static_ids, compiled_ids)),
            "static_eager_vs_compiled_trimmed": bool(static_trimmed_rows == compiled_trimmed_rows),
            "compare_loop_static_vs_compiled": bool(torch.equal(compare_static_ids, compare_compiled_ids)),
            "static_eager_vs_single_refs": single_ref_matches,
            "all_required_checks_passed": fixed_required_checks_passed,
        },
        "logit_diff_static_eager_vs_compiled_decode": {
            "steps": int(compared_steps),
            "max_abs": float(max_abs),
            "mean_abs": float(mean_abs),
        },
        "timing_s": {
            "compile_wrapper": float(compile_wrapper_s),
            "compile_first_call": float(compile_first_s),
            "static_eager_decode": float(static_decode_s),
            "compiled_decode": float(compiled_decode_s),
        },
        "timing_accounting": {
            "static_eager": static_timing_accounting,
            "compiled": compiled_timing_accounting,
        },
        "tok_per_s": {
            "static_eager_decode_steps": tok_per_s(static_result.decode_calls, static_decode_s),
            "static_eager_raw_batch_tokens": tok_per_s(static_raw_token_calls, static_decode_s),
            "static_eager_effective_batch_tokens": tok_per_s(static_effective_token_calls, static_decode_s),
            "compiled_decode_steps": tok_per_s(compiled_result.decode_calls, compiled_decode_s),
            "compiled_raw_batch_tokens": tok_per_s(compiled_raw_token_calls, compiled_decode_s),
            "compiled_effective_batch_tokens": tok_per_s(compiled_effective_token_calls, compiled_decode_s),
        },
        "step_timing": {
            "static_eager": static_result.step_timing,
            "compiled": compiled_result.step_timing,
        },
        "texts": {
            "static_eager": [
                tokenizer.decode(row, skip_special_tokens=True)
                for row in row_token_lists(static_ids)
            ],
            "compiled": [
                tokenizer.decode(row, skip_special_tokens=True)
                for row in row_token_lists(compiled_ids)
            ],
            "static_eager_trimmed": [
                tokenizer.decode(row, skip_special_tokens=True)
                for row in static_trimmed_rows
            ],
            "compiled_trimmed": [
                tokenizer.decode(row, skip_special_tokens=True)
                for row in compiled_trimmed_rows
            ],
        },
    }
    if profile_summary is not None:
        summary["profile"] = profile_summary

    if args.json:
        output = benchmark_report(summary) if args.report == "summary" else summary
        print(json.dumps(output, indent=2, sort_keys=True, default=json_default))
        if not fixed_required_checks_passed:
            raise SystemExit(1)
        return

    if args.report == "summary":
        print("benchmark_report=" + json.dumps(benchmark_report(summary), sort_keys=True, default=json_default))
        if not fixed_required_checks_passed:
            raise SystemExit(1)
        return

    print(
        f"experiment={summary['experiment']} backend={summary['backend']} device={summary['device']} "
        f"dtype={summary['dtype']} batch_size={summary['batch_size']} npu_jit_compile={summary['npu_jit_compile']} "
        f"decode_attention={summary['decode_attention']} eos_mode={summary['eos_mode']}"
    )
    print("cohort=" + json.dumps(summary["cohort"], sort_keys=True))
    print("linear_weight_format=" + json.dumps(summary["linear_weight_format"], sort_keys=True))
    print("cache_update=" + summary["cache_update"])
    print("decode_cache_update=" + summary["decode_cache_update"])
    print(
        f"prompt_tokens={summary['prompt_tokens']} generated_tokens_per_row={summary['generated_tokens_per_row']} "
        f"requested_decode_steps={summary['requested_decode_steps']} cache_length={summary['cache_length']}"
    )
    print("loop=" + json.dumps(summary["loop"], sort_keys=True))
    print("matches=" + json.dumps(summary["matches"], sort_keys=True))
    print("logit_diff_static_eager_vs_compiled_decode=" + json.dumps(summary["logit_diff_static_eager_vs_compiled_decode"], sort_keys=True))
    print("timing_s=" + json.dumps(summary["timing_s"], sort_keys=True))
    print("timing_accounting=" + json.dumps(summary["timing_accounting"], sort_keys=True))
    print("tok_per_s=" + json.dumps(summary["tok_per_s"], sort_keys=True))
    if args.step_timing != "off":
        print("step_timing_summary=" + json.dumps({
            "static_eager": static_loop_summary["step_timing_summary"],
            "compiled": compiled_loop_summary["step_timing_summary"],
        }, sort_keys=True))
    if profile_summary is not None:
        print("profile=" + json.dumps(profile_summary, sort_keys=True, default=json_default))
    print("compiled_texts=" + repr(summary["texts"]["compiled_trimmed"]))
    if not fixed_required_checks_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
