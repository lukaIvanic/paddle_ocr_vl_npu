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
REPO_ROOT = Path(__file__).resolve().parents[1]


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


def length_trimmed_token_lists(ids: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    rows = row_token_lists(ids)
    length_values = [int(value) for value in lengths.detach().cpu().tolist()]
    return [row[: max(0, length)] for row, length in zip(rows, length_values)]


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
        single_prefill = BatchedPrefill(
            cache=LocalPaddleOCRVLStaticCache(
                tuple(cache[item_idx : item_idx + 1].clone().contiguous() for cache in ready.cache.key_caches),
                tuple(cache[item_idx : item_idx + 1].clone().contiguous() for cache in ready.cache.value_caches),
                int(ready.cache.cache_length),
            ),
            rope_deltas=ready.rope_deltas[item_idx : item_idx + 1].clone().contiguous(),
            next_cache_position=ready.next_cache_position[item_idx : item_idx + 1].clone().contiguous(),
        )
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
def copy_ready_item_to_active_slot_(
    *,
    ready: ReadyBank,
    active: BatchedPrefill,
    active_next_token: torch.Tensor,
    active_item_indices: torch.Tensor,
    active_mask: torch.Tensor,
    slot: int,
    item_idx: int,
) -> None:
    slot_slice = slice(int(slot), int(slot) + 1)
    item_slice = slice(int(item_idx), int(item_idx) + 1)
    for layer_idx in range(len(active.cache.key_caches)):
        active.cache.key_caches[layer_idx][slot_slice].copy_(ready.cache.key_caches[layer_idx][item_slice])
        active.cache.value_caches[layer_idx][slot_slice].copy_(ready.cache.value_caches[layer_idx][item_slice])
    active.rope_deltas[slot_slice].copy_(ready.rope_deltas[item_slice])
    active.next_cache_position[slot_slice].copy_(ready.next_cache_position[item_slice])
    active_next_token[slot_slice].copy_(ready.next_token[item_slice])
    active_item_indices[slot_slice].fill_(int(item_idx))
    active_mask[slot_slice].fill_(True)


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
) -> HotSwapDecodeResult:
    if eos_mode != "overlap_event_flags":
        raise ValueError("--schedule hotswap requires --eos-mode overlap_event_flags")
    if int(batch_size) <= 0:
        raise ValueError(f"--batch-size must be positive, got {batch_size}")
    if int(max_new_tokens) <= 0:
        raise ValueError(f"--max-new-tokens must be positive, got {max_new_tokens}")

    device = ready.next_token.device
    use_npu_overlap = device.type == "npu"
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
    generated_ids = torch.full((num_items, int(max_new_tokens)), int(eos_token_id), device=device, dtype=ready.next_token.dtype)
    generated_lengths = torch.zeros((num_items,), device=device, dtype=torch.int64)
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

    def load_item_to_slot(slot: int, item_idx: int) -> None:
        copy_ready_item_to_active_slot_(
            ready=ready,
            active=active,
            active_next_token=active_next_token,
            active_item_indices=active_item_indices,
            active_mask=active_mask,
            slot=slot,
            item_idx=item_idx,
        )
        active_item_indices_cpu[int(slot)] = int(item_idx)
        generated_ids[int(item_idx), 0].copy_(ready.next_token[int(item_idx), 0])
        generated_lengths_cpu[int(item_idx)] = 1
        generated_lengths[int(item_idx)].fill_(1)
        first_token_eos = ready_first_tokens_cpu[int(item_idx)] == int(eos_token_id)
        first_token_cap = int(max_new_tokens) <= 1
        initial_finished = bool(first_token_eos or first_token_cap)
        slot_finished_cpu[int(slot)] = initial_finished
        slot_finished[int(slot) : int(slot) + 1].fill_(initial_finished)
        item_eos_hit_cpu[int(item_idx)] = bool(first_token_eos)
        item_length_cap_hit_cpu[int(item_idx)] = bool(first_token_cap)

    def deactivate_slot(slot: int) -> None:
        active_item_indices_cpu[int(slot)] = -1
        active_item_indices[int(slot) : int(slot) + 1].fill_(-1)
        active_mask[int(slot) : int(slot) + 1].fill_(False)
        slot_finished_cpu[int(slot)] = False
        slot_finished[int(slot) : int(slot) + 1].fill_(False)
        active_next_token[int(slot) : int(slot) + 1].fill_(int(eos_token_id))

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

    # Handle the rare case where the prefill-selected first token is already EOS
    # or the run intentionally requests only one generated token.
    while True:
        initial_finished_flags = [bool(value) for value in slot_finished.detach().cpu().tolist()]
        if not any(initial_finished_flags):
            break
        consume_finished_slots(initial_finished_flags, completion_decode_call=0)
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
            consume_finished_slots(finished_flags, int(pending_completion_decode_call), record)
            npu_swap_end = timing.npu_event()
            swap_s = timing.cpu_elapsed_s(host_swap_start)
            if swap_s is not None:
                record["host_swap_s"] = swap_s
            events["npu_swap"] = (npu_swap_start, npu_swap_end)
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
            generated_ids[item_idx, position].copy_(active_next_token[slot, 0])
            new_length = position + 1
            generated_lengths_cpu[item_idx] = new_length
            generated_lengths[item_idx : item_idx + 1].fill_(new_length)
            if new_length >= int(max_new_tokens):
                length_cap_slots.append(int(slot))
                slot_finished_cpu[int(slot)] = True
                item_length_cap_hit_cpu[item_idx] = True
        for slot in length_cap_slots:
            slot_finished[int(slot) : int(slot) + 1].fill_(True)

        new_eos_by_slot = active_before_step & hits_eos(active_next_token, int(eos_token_id))
        slot_finished.logical_or_(new_eos_by_slot)
        active.next_cache_position.add_(active_before_step.to(dtype=active.next_cache_position.dtype))
        decode_calls += 1

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
        consume_finished_slots(finished_flags, int(pending_completion_decode_call))

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
    return {
        "full_matches_by_row": full_matches,
        "trimmed_matches_by_row": trimmed_matches,
        "all_full_match": bool(all(full_matches)),
        "all_trimmed_match": bool(all(trimmed_matches)),
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
        "decode_attention": DECODE_ATTENTION,
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
    parser.add_argument("--torchair-cache-dir", type=Path, default=DEFAULT_TORCHAIR_CACHE_DIR)
    parser.add_argument("--eos-mode", default="none", choices=EOS_MODE_CHOICES)
    parser.add_argument("--step-timing", default="off", choices=STEP_TIMING_CHOICES)
    parser.add_argument("--profile-dir", type=Path, default=None, help="Write one post-warmup torch_npu profiler capture for compiled batched decode.")
    parser.add_argument("--profile-metric", default="pipe", choices=PROFILE_METRIC_CHOICES)
    parser.add_argument("--json", action="store_true", help="Print a compact JSON summary instead of human-readable lines.")
    args = parser.parse_args()

    if int(args.max_new_tokens) <= 0:
        raise ValueError(f"--max-new-tokens must be positive, got {args.max_new_tokens}")
    if args.schedule == "hotswap" and args.eos_mode != "overlap_event_flags":
        raise ValueError("--schedule hotswap requires --eos-mode overlap_event_flags")
    if args.schedule == "hotswap" and args.profile_dir is not None:
        raise ValueError("--profile-dir currently profiles fixed-cohort decode only; run hot-swap without profiler first.")

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
        hotswap_rows = length_trimmed_token_lists(hotswap_result.ids, hotswap_result.lengths)
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
        hotswap_required_checks_passed = bool(hotswap_validation["all_trimmed_match"])
        summary = {
            "experiment": "04_batched_fixed_cohort_decode",
            "schedule": args.schedule,
            "backend": args.backend,
            "device": str(device),
            "dtype": str(dtype),
            "npu_jit_compile": args.npu_jit_compile,
            "decode_attention": DECODE_ATTENTION,
            "eos_mode": args.eos_mode,
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
            "cache_update": "prefill_slice_decode_npu_scatter",
            "history_write_mode": "host_indexed_per_slot_copy",
            "prompt_tokens": {
                "per_item": prompt_tokens,
                "min": int(min(prompt_tokens)),
                "max": int(max(prompt_tokens)),
            },
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
            },
            "timing_accounting": hotswap_timing_accounting,
            "tok_per_s": {
                "hotswap_decode_steps": tok_per_s(hotswap_result.decode_calls, hotswap_decode_s),
                "hotswap_raw_batch_tokens": tok_per_s(raw_token_calls, hotswap_decode_s),
                "hotswap_effective_item_tokens": tok_per_s(effective_token_calls, hotswap_decode_s),
            },
            "swap_events": hotswap_result.swap_events,
            "step_timing": hotswap_result.step_timing,
            "texts": {
                "hotswap_trimmed": [
                    tokenizer.decode(row, skip_special_tokens=True)
                    for row in hotswap_rows
                ],
            },
        }
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
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
        print("history_write_mode=" + summary["history_write_mode"])
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
        "decode_attention": DECODE_ATTENTION,
        "eos_mode": args.eos_mode,
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
        "cache_update": "prefill_slice_decode_npu_scatter",
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
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
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
