#!/usr/bin/env python3
"""Benchmark only the PaddleOCR-VL vision prefill path on real page crops.

This is an experiment-6 side harness. It intentionally reuses the same
OmniDocBench page loading, GT layout crop extraction, crop preprocessing, and
prompt/input construction as the full page pipeline, then stops after the
native-resolution vision encoder plus adaptive MLP projector.

Measured vision call per crop:

    CPU preprocessed crop tensor -> device transfer -> model.get_image_features()

No text-token embedding, image embedding scatter, KV prefill, LM head, decode,
or output validation is run here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from tokenizers import Tokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EXP5_DIR = REPO_ROOT / "05_full_recognizer_optimizations"
if str(EXP5_DIR) not in sys.path:
    sys.path.insert(0, str(EXP5_DIR))

from bench_page_pipeline_e2e import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    aggregate_timing_dicts,
    build_detected_crops,
    build_omnidocbench_gt_layout_pages,
    build_queue_inputs_from_crops,
    clean_json,
    load_pages_result,
    page_load_summary,
    resolve_dataset_dir,
    tok_per_s,
)
from bench_recognizer_queue import QueueInput, json_default, stats  # noqa: E402
from local_modeling_paddleocr_vl import (  # noqa: E402
    VISION_ATTENTION_CHOICES,
    VISION_ATTENTION_ENV,
    VISION_PROMPT_FA_LAYOUT_CHOICES,
    VISION_PROMPT_FA_LAYOUT_ENV,
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
    get_vision_attention_impl,
    get_vision_prompt_fa_layout,
)
from probe_static_compile import import_torchair, maybe_sync  # noqa: E402
from run_local_recognition import (  # noqa: E402
    NPU_JIT_COMPILE_CHOICES,
    configure_npu_jit_compile,
    load_preprocessor_config,
    resolve_device,
)


MODE_CHOICES = ("sync_per_crop", "unsynced_loop")
PROFILE_METRIC_CHOICES = ("pipe", "memory", "l2", "memory_access")
CROP_SAMPLE_CHOICES = ("all", "small_medium_large", "small_only")
VISION_COMPILE_BACKEND_CHOICES = ("none", "default", "aot_eager", "inductor", "torchair")


def parse_vision_dtype(name: str) -> torch.dtype:
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def parse_modes(raw: str) -> list[str]:
    modes = [item.strip() for item in str(raw).replace(",", " ").split() if item.strip()]
    if not modes:
        raise ValueError("--modes must select at least one mode")
    bad = [mode for mode in modes if mode not in MODE_CHOICES]
    if bad:
        raise ValueError(f"unsupported mode(s): {bad}; choices={MODE_CHOICES}")
    deduped: list[str] = []
    for mode in modes:
        if mode not in deduped:
            deduped.append(mode)
    return deduped


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


def make_profile_run_dir(
    root: Path,
    *,
    mode: str,
    attention: str,
    dtype: torch.dtype,
    crop_count: int,
    metric: str,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dtype_name = str(dtype).replace("torch.", "")
    safe_attention = str(attention).replace("/", "_")
    return root / f"vision_prefill_{timestamp}_{mode}_{safe_attention}_{dtype_name}_{crop_count}crops_{metric}"


def tensor_grid(item: QueueInput) -> list[int]:
    return [int(value) for value in item.image_grid_thw.reshape(-1).tolist()]


def vision_tokens(item: QueueInput) -> int:
    return int(item.image_grid_thw.prod().item())


def projected_tokens(item: QueueInput, *, merge_size: int) -> int:
    return int(vision_tokens(item) // int(merge_size) // int(merge_size))


def summarize_inputs(inputs: list[QueueInput], *, merge_size: int) -> dict[str, Any]:
    vision_counts = [vision_tokens(item) for item in inputs]
    projected_counts = [projected_tokens(item, merge_size=merge_size) for item in inputs]
    input_counts = [int(item.input_ids.shape[1]) for item in inputs]
    grids = [tuple(tensor_grid(item)) for item in inputs]
    grid_counts = Counter(grids)
    label_counts = Counter(str(item.entry.get("layout_label", "unknown")) for item in inputs)
    prompt_counts = Counter(str(item.prompt) for item in inputs)
    crop_sizes = [
        [int(value) for value in clean_json(item.entry.get("crop_size", [0, 0]))]
        for item in inputs
    ]
    return {
        "count": int(len(inputs)),
        "total_vision_tokens": int(sum(vision_counts)),
        "total_projected_image_tokens": int(sum(projected_counts)),
        "total_input_tokens": int(sum(input_counts)),
        "vision_tokens": {
            "stats": stats([float(value) for value in vision_counts]),
            "unique_count": int(len(set(vision_counts))),
        },
        "projected_image_tokens": {
            "stats": stats([float(value) for value in projected_counts]),
            "unique_count": int(len(set(projected_counts))),
        },
        "input_tokens": {
            "stats": stats([float(value) for value in input_counts]),
            "unique_count": int(len(set(input_counts))),
        },
        "image_grid_thw": {
            "unique_count": int(len(grid_counts)),
            "top_buckets": [
                {"grid": [int(value) for value in grid], "count": int(count)}
                for grid, count in sorted(grid_counts.items(), key=lambda item: (-item[1], item[0]))[:16]
            ],
        },
        "label_counts": dict(sorted(label_counts.items())),
        "prompt_counts": dict(sorted(prompt_counts.items())),
        "crop_size_samples": crop_sizes[:16],
    }


def clone_input_with_sample_meta(
    item: QueueInput,
    *,
    bucket: str,
    selected_rank: int,
    original_idx: int,
) -> QueueInput:
    entry = dict(item.entry)
    entry["crop_sample_bucket"] = bucket
    entry["crop_sample_selected_rank"] = int(selected_rank)
    entry["crop_sample_original_idx"] = int(original_idx)
    return QueueInput(
        entry=entry,
        crop_path=item.crop_path,
        prompt=item.prompt,
        input_ids=item.input_ids,
        attention_mask=item.attention_mask,
        pixel_values=item.pixel_values,
        image_grid_thw=item.image_grid_thw,
        timing_s=item.timing_s,
    )


def select_profile_crop_sample(inputs: list[QueueInput], *, strategy: str) -> tuple[list[QueueInput], dict[str, Any]]:
    if strategy not in CROP_SAMPLE_CHOICES:
        raise ValueError(f"unsupported crop sample strategy: {strategy}; choices={CROP_SAMPLE_CHOICES}")
    if strategy == "all":
        return inputs, {
            "strategy": "all",
            "input_count_before_sample": int(len(inputs)),
            "selected_count": int(len(inputs)),
            "note": "No crop-size sampling was applied.",
        }
    if not inputs:
        raise ValueError("cannot sample representative crops from an empty input list")

    sorted_pairs = sorted(
        enumerate(inputs),
        key=lambda pair: (
            vision_tokens(pair[1]),
            int(pair[1].input_ids.shape[1]),
            str(pair[1].entry.get("id", "")),
        ),
    )
    if strategy == "small_only":
        original_idx, item = sorted_pairs[0]
        cloned = clone_input_with_sample_meta(
            item,
            bucket="small",
            selected_rank=0,
            original_idx=original_idx,
        )
        return [cloned], {
            "strategy": strategy,
            "input_count_before_sample": int(len(inputs)),
            "selected_count": 1,
            "selection_basis": "lowest post-preprocessing vision token count over real extracted/preprocessed OCR crops",
            "selected": [
                {
                    "bucket": "small",
                    "selected_rank": 0,
                    "original_idx": int(original_idx),
                    "sorted_position": 0,
                    "id": str(item.entry.get("id")),
                    "page_index": int(item.entry.get("page_index", 0)),
                    "layout_label": str(item.entry.get("layout_label", "")),
                    "vision_tokens": int(vision_tokens(item)),
                    "input_tokens": int(item.input_ids.shape[1]),
                    "image_grid_thw": tensor_grid(item),
                    "crop_size": clean_json(item.entry.get("crop_size", [0, 0])),
                }
            ],
        }
    n = len(sorted_pairs)
    targets = [
        ("small", 0),
        ("medium", (n - 1) // 2),
        ("large", n - 1),
    ]
    used_positions: set[int] = set()
    selected: list[QueueInput] = []
    selected_rows: list[dict[str, Any]] = []

    for rank, (bucket, target_pos) in enumerate(targets):
        if len(used_positions) >= n:
            break
        best_pos = min(
            (pos for pos in range(n) if pos not in used_positions),
            key=lambda pos: (abs(pos - target_pos), pos),
        )
        used_positions.add(best_pos)
        original_idx, item = sorted_pairs[best_pos]
        cloned = clone_input_with_sample_meta(
            item,
            bucket=bucket,
            selected_rank=rank,
            original_idx=original_idx,
        )
        selected.append(cloned)
        selected_rows.append(
            {
                "bucket": bucket,
                "selected_rank": int(rank),
                "original_idx": int(original_idx),
                "sorted_position": int(best_pos),
                "id": str(item.entry.get("id")),
                "page_index": int(item.entry.get("page_index", 0)),
                "layout_label": str(item.entry.get("layout_label", "")),
                "vision_tokens": int(vision_tokens(item)),
                "input_tokens": int(item.input_ids.shape[1]),
                "image_grid_thw": tensor_grid(item),
                "crop_size": clean_json(item.entry.get("crop_size", [0, 0])),
            }
        )

    return selected, {
        "strategy": strategy,
        "input_count_before_sample": int(len(inputs)),
        "selected_count": int(len(selected)),
        "selection_basis": "vision_tokens sorted ascending over real extracted/preprocessed OCR crops",
        "selected": selected_rows,
    }


class SingleCropVisionFeatureModule(torch.nn.Module):
    """Shape-specialized wrapper for compiling one real crop's vision path."""

    def __init__(self, model: LocalPaddleOCRVLForConditionalGeneration, image_grid_thw: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("image_grid_thw_const", image_grid_thw.detach().clone(), persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.get_image_features(pixel_values, self.image_grid_thw_const)


def vision_compile_backend(name: str, device: torch.device):
    if name == "default":
        return None
    if name == "torchair":
        if device.type != "npu":
            raise ValueError("--vision-compile-backend torchair requires --device npu:0")
        torchair, CompilerConfig = import_torchair()
        config = CompilerConfig()
        return torchair.get_npu_backend(compiler_config=config)
    return name


def compile_single_crop_vision_forward(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    item: QueueInput,
    device: torch.device,
    backend_name: str,
) -> tuple[Callable[[torch.Tensor], torch.Tensor] | None, dict[str, Any]]:
    backend_name = str(backend_name)
    if backend_name == "none":
        return None, {
            "enabled": False,
            "backend": backend_name,
            "compile_api": None,
        }

    wrapper = SingleCropVisionFeatureModule(model, item.image_grid_thw).eval()
    compile_kwargs: dict[str, Any] = {
        "fullgraph": True,
        "dynamic": False,
    }
    backend = vision_compile_backend(backend_name, device)
    if backend is not None:
        compile_kwargs["backend"] = backend

    import torch._dynamo

    old_capture_scalar_outputs = bool(torch._dynamo.config.capture_scalar_outputs)
    torch._dynamo.config.capture_scalar_outputs = True
    maybe_sync(device)
    compile_start = time.perf_counter()
    compiled = torch.compile(wrapper, **compile_kwargs)
    maybe_sync(device)
    compile_wrapper_s = time.perf_counter() - compile_start

    return compiled, {
        "enabled": True,
        "backend": backend_name,
        "compile_api": "torch.compile",
        "fullgraph": True,
        "dynamic": False,
        "capture_scalar_outputs": True,
        "capture_scalar_outputs_previous": old_capture_scalar_outputs,
        "capture_scalar_outputs_scope": (
            "set globally for this process before constructing the compiled callable and kept enabled "
            "through first-call tracing because the vision path reads image_grid_thw with Tensor.item()."
        ),
        "compile_wrapper_s": float(compile_wrapper_s),
        "crop_id": str(item.entry.get("id")),
        "crop_sample_bucket": item.entry.get("crop_sample_bucket"),
        "vision_tokens": int(vision_tokens(item)),
        "image_grid_thw": tensor_grid(item),
        "note": (
            "The compiled callable is shape-specialized to exactly one selected crop. "
            "The input tensor is the already CPU-preprocessed crop patch tensor after transfer to the target device; "
            "the compiled graph covers model.get_image_features(), i.e. native-resolution visual encoder plus adaptive MLP projector."
        ),
    }


@torch.inference_mode()
def run_vision_one(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    item: QueueInput,
    device: torch.device,
    vision_forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    pixel_values = item.pixel_values.to(device=device, dtype=model.visual.dtype)
    if vision_forward is not None:
        return vision_forward(pixel_values)
    return model.get_image_features(pixel_values, item.image_grid_thw)


@torch.inference_mode()
def warmup_vision(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    inputs: list[QueueInput],
    device: torch.device,
    warmup_items: int,
    vision_forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, Any]:
    count = min(max(0, int(warmup_items)), len(inputs))
    if count <= 0:
        return {"count": 0, "elapsed_s": 0.0, "item_ids": []}
    maybe_sync(device)
    start = time.perf_counter()
    outputs = []
    for item in inputs[:count]:
        outputs.append(run_vision_one(model=model, item=item, device=device, vision_forward=vision_forward))
    maybe_sync(device)
    elapsed = time.perf_counter() - start
    return {
        "count": int(count),
        "elapsed_s": float(elapsed),
        "item_ids": [str(item.entry.get("id")) for item in inputs[:count]],
        "projected_shapes": [[int(dim) for dim in output.shape] for output in outputs],
    }


@torch.inference_mode()
def run_sync_per_crop(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    inputs: list[QueueInput],
    device: torch.device,
    repeat_count: int = 1,
    vision_forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_start = time.perf_counter()
    for repeat_idx in range(max(1, int(repeat_count))):
        for original_idx, item in enumerate(inputs):
            idx = repeat_idx * len(inputs) + original_idx
            maybe_sync(device)
            start = time.perf_counter()
            output = run_vision_one(model=model, item=item, device=device, vision_forward=vision_forward)
            maybe_sync(device)
            elapsed = time.perf_counter() - start
            rows.append(make_forward_row(idx, item, output, repeat_idx=repeat_idx, original_idx=original_idx, elapsed_s=elapsed))
    total_s = time.perf_counter() - total_start
    return summarize_mode("sync_per_crop", rows=rows, total_s=total_s)


@torch.inference_mode()
def run_unsynced_loop(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    inputs: list[QueueInput],
    device: torch.device,
    repeat_count: int = 1,
    vision_forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    maybe_sync(device)
    start = time.perf_counter()
    for repeat_idx in range(max(1, int(repeat_count))):
        for original_idx, item in enumerate(inputs):
            idx = repeat_idx * len(inputs) + original_idx
            output = run_vision_one(model=model, item=item, device=device, vision_forward=vision_forward)
            rows.append(make_forward_row(idx, item, output, repeat_idx=repeat_idx, original_idx=original_idx))
    maybe_sync(device)
    total_s = time.perf_counter() - start
    return summarize_mode("unsynced_loop", rows=rows, total_s=total_s)


def make_forward_row(
    idx: int,
    item: QueueInput,
    output: torch.Tensor,
    *,
    repeat_idx: int,
    original_idx: int,
    elapsed_s: float | None = None,
    profile_step: int | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "idx": int(idx),
        "repeat_idx": int(repeat_idx),
        "original_idx": int(original_idx),
        "id": str(item.entry.get("id")),
        "page_index": int(item.entry.get("page_index", 0)),
        "layout_label": str(item.entry.get("layout_label", "")),
        "crop_sample_bucket": item.entry.get("crop_sample_bucket"),
        "vision_tokens": int(vision_tokens(item)),
        "projected_image_tokens": int(output.shape[0]),
        "input_tokens": int(item.input_ids.shape[1]),
        "image_grid_thw": tensor_grid(item),
    }
    if elapsed_s is not None:
        row["elapsed_s"] = float(elapsed_s)
    if profile_step is not None:
        row["profile_step"] = int(profile_step)
    return row


def summarize_mode(mode: str, *, rows: list[dict[str, Any]], total_s: float) -> dict[str, Any]:
    total_vision_tokens = int(sum(int(row.get("vision_tokens", 0)) for row in rows))
    total_projected_tokens = int(sum(int(row.get("projected_image_tokens", 0)) for row in rows))
    input_tokens = int(sum(int(row.get("input_tokens", 0)) for row in rows))
    elapsed_rows = [float(row["elapsed_s"]) for row in rows if "elapsed_s" in row]
    return {
        "mode": mode,
        "measurement_scope": (
            "per crop: CPU preprocessed pixel tensor -> device transfer -> native-resolution visual encoder "
            "+ post layernorm + adaptive MLP projector"
        ),
        "sync_policy": (
            "device synchronize before and after every crop"
            if mode == "sync_per_crop"
            else "one device synchronize before the loop and one after the full loop; no per-crop sync"
        ),
        "count": int(len(rows)),
        "total_s": float(total_s),
        "items_per_s": tok_per_s(len(rows), total_s),
        "vision_tokens_per_s": tok_per_s(total_vision_tokens, total_s),
        "projected_image_tokens_per_s": tok_per_s(total_projected_tokens, total_s),
        "input_tokens_per_s": tok_per_s(input_tokens, total_s),
        "total_vision_tokens": int(total_vision_tokens),
        "total_projected_image_tokens": int(total_projected_tokens),
        "total_input_tokens": int(input_tokens),
        "per_crop_elapsed_s": stats(elapsed_rows) if elapsed_rows else None,
        "samples": rows[:16],
    }


@torch.inference_mode()
def run_mode(
    mode: str,
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    inputs: list[QueueInput],
    device: torch.device,
    repeat_count: int = 1,
    vision_forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, Any]:
    if mode == "sync_per_crop":
        return run_sync_per_crop(
            model=model,
            inputs=inputs,
            device=device,
            repeat_count=repeat_count,
            vision_forward=vision_forward,
        )
    if mode == "unsynced_loop":
        return run_unsynced_loop(
            model=model,
            inputs=inputs,
            device=device,
            repeat_count=repeat_count,
            vision_forward=vision_forward,
        )
    raise AssertionError(mode)


@torch.inference_mode()
def profile_vision_mode(
    *,
    profile_root: Path,
    profile_mode: str,
    profile_metric: str,
    model: LocalPaddleOCRVLForConditionalGeneration,
    inputs: list[QueueInput],
    device: torch.device,
    dtype: torch.dtype,
    warmup_repeats: int,
    active_repeats: int,
    vision_forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, Any]:
    if device.type != "npu":
        raise ValueError("--profile-dir requires --device npu:0; torch_npu profiler is NPU-only.")

    import torch_npu.profiler as npu_prof

    profile_run_dir = make_profile_run_dir(
        profile_root.expanduser().resolve(),
        mode=profile_mode,
        attention=get_vision_attention_impl(),
        dtype=dtype,
        crop_count=len(inputs),
        metric=profile_metric,
    )
    shutil.rmtree(profile_run_dir, ignore_errors=True)
    profile_run_dir.mkdir(parents=True, exist_ok=True)

    warmup_repeats = max(0, int(warmup_repeats))
    active_repeats = max(1, int(active_repeats))
    active_steps = active_repeats * len(inputs)

    maybe_sync(device)
    warmup_start = time.perf_counter()
    warmup_forward_count = 0
    for _ in range(warmup_repeats):
        for item in inputs:
            run_vision_one(model=model, item=item, device=device, vision_forward=vision_forward)
            warmup_forward_count += 1
    maybe_sync(device)
    profiler_warmup_s = time.perf_counter() - warmup_start

    schedule = npu_prof.schedule(wait=0, warmup=0, active=active_steps, repeat=1)
    rows: list[dict[str, Any]] = []
    maybe_sync(device)
    profile_context_start = time.perf_counter()
    active_loop_start: float | None = None
    active_loop_end: float | None = None
    profiler_step_times_s: list[float] = []
    with npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        schedule=schedule,
        experimental_config=npu_profiler_config(profile_metric),
        on_trace_ready=npu_prof.tensorboard_trace_handler(str(profile_run_dir), analyse_flag=True),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    ) as profiler:
        profile_step = 0
        active_loop_start = time.perf_counter()
        for repeat_idx in range(active_repeats):
            for original_idx, item in enumerate(inputs):
                idx = repeat_idx * len(inputs) + original_idx
                bucket = str(item.entry.get("crop_sample_bucket") or "crop")
                with torch.profiler.record_function(f"paddle_ocr_vl.vision_prefill_profile.{bucket}"):
                    step_start = time.perf_counter()
                    output = run_vision_one(model=model, item=item, device=device, vision_forward=vision_forward)
                    maybe_sync(device)
                    elapsed = time.perf_counter() - step_start
                rows.append(
                    make_forward_row(
                        idx,
                        item,
                        output,
                        repeat_idx=repeat_idx,
                        original_idx=original_idx,
                        elapsed_s=elapsed,
                        profile_step=profile_step,
                    )
                )
                profiler_step_start = time.perf_counter()
                profiler.step()
                profiler_step_times_s.append(float(time.perf_counter() - profiler_step_start))
                profile_step += 1
        active_loop_end = time.perf_counter()
    maybe_sync(device)
    profile_context_wall_s = time.perf_counter() - profile_context_start

    forward_sync_s = float(sum(float(row.get("elapsed_s", 0.0)) for row in rows))
    profiler_step_s = float(sum(profiler_step_times_s))
    active_loop_wall_s = float((active_loop_end or profile_context_start) - (active_loop_start or profile_context_start))
    context_non_active_s = float(max(0.0, profile_context_wall_s - active_loop_wall_s))
    active_loop_unattributed_s = float(max(0.0, active_loop_wall_s - forward_sync_s - profiler_step_s))
    profiled_mode_context_result = summarize_mode("profile_context_wall", rows=rows, total_s=profile_context_wall_s)
    profiled_mode_forward_sync_result = summarize_mode("profile_forward_sync_only", rows=rows, total_s=forward_sync_s)

    summary = {
        "enabled": True,
        "profile_dir": str(profile_run_dir),
        "profile_mode": str(profile_mode),
        "profile_metric": str(profile_metric),
        "scope": (
            "same selected real OCR crops as the benchmark mode; CPU preprocessed pixel tensor -> device transfer "
            "-> native-resolution visual encoder + post layernorm + adaptive MLP projector"
        ),
        "post_warmup": True,
        "with_stack": True,
        "record_shapes": True,
        "profile_memory": False,
        "profiler_step_contract": "one profiler.step() is called after exactly one model.get_image_features() crop forward",
        "profile_warmup_repeats": int(warmup_repeats),
        "profile_active_repeats": int(active_repeats),
        "profile_warmup_forward_count": int(warmup_forward_count),
        "profile_active_steps": int(active_steps),
        "profile_warmup_s": float(profiler_warmup_s),
        "profile_context_wall_s": float(profile_context_wall_s),
        "profile_active_loop_wall_s": float(active_loop_wall_s),
        "profile_forward_sync_sum_s": float(forward_sync_s),
        "profile_profiler_step_sum_s": float(profiler_step_s),
        "profile_context_non_active_s": float(context_non_active_s),
        "profile_active_loop_unattributed_s": float(active_loop_unattributed_s),
        "profile_profiler_step_s": stats(profiler_step_times_s),
        "profiled_context_wall_result": profiled_mode_context_result,
        "profiled_forward_sync_result": profiled_mode_forward_sync_result,
        "profiled_mode_result": profiled_mode_context_result,
        "profiled_mode_result_note": (
            "Deprecated compatibility field. This uses full profiler context wall time and may include "
            "profiler.step(), trace handling, and export/finalization overhead. Prefer profiled_forward_sync_result "
            "for forward+sync timing and profile_profiler_step_sum_s / profile_context_non_active_s for overhead."
        ),
    }
    summary_path = profile_run_dir / "vision_prefill_profile_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--page-start", type=int, default=0)
    parser.add_argument("--num-pages", type=int, default=8)
    parser.add_argument("--layout-source", default="omnidocbench_gt", choices=["omnidocbench_gt"])
    parser.add_argument("--crop-padding", type=int, default=0)
    parser.add_argument("--min-crop-side", type=int, default=4)
    parser.add_argument("--skip-labels", default="")
    parser.add_argument("--include-ignored-gt", action="store_true")
    parser.add_argument("--include-empty-gt", action="store_true")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "fp32", "float32", "bf16", "bfloat16"])
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--vision-attention", default=os.environ.get(VISION_ATTENTION_ENV, "manual"), choices=VISION_ATTENTION_CHOICES)
    parser.add_argument(
        "--vision-prompt-fa-layout",
        default=os.environ.get(VISION_PROMPT_FA_LAYOUT_ENV, "bnsd"),
        choices=VISION_PROMPT_FA_LAYOUT_CHOICES,
    )
    parser.add_argument(
        "--modes",
        default="sync_per_crop,unsynced_loop",
        help="Comma/space separated modes. Choices: sync_per_crop, unsynced_loop.",
    )
    parser.add_argument("--warmup-items", type=int, default=1)
    parser.add_argument(
        "--crop-sample",
        default="all",
        choices=CROP_SAMPLE_CHOICES,
        help="Optionally reduce real OCR crops to representative size buckets before speed/profiler runs.",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=1,
        help="Repeat the selected crop list this many times for non-profiled benchmark modes.",
    )
    parser.add_argument(
        "--vision-compile-backend",
        default="none",
        choices=VISION_COMPILE_BACKEND_CHOICES,
        help=(
            "Compile the whole single-crop model.get_image_features path before benchmark/profiler runs. "
            "Requires --crop-sample small_only because the graph is shape-specialized to one crop."
        ),
    )
    parser.add_argument(
        "--vision-compile-validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare the first compiled vision output against eager for the selected crop before timed runs.",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="Optional NPU profiler output root. Captures one post-warmup run of --profile-mode.",
    )
    parser.add_argument(
        "--profile-mode",
        default="unsynced_loop",
        choices=MODE_CHOICES,
        help="Benchmark mode to rerun under torch_npu.profiler when --profile-dir is set.",
    )
    parser.add_argument(
        "--profile-metric",
        default="pipe",
        choices=PROFILE_METRIC_CHOICES,
        help="torch_npu profiler AiC metric for --profile-dir captures.",
    )
    parser.add_argument(
        "--profile-warmup-repeats",
        type=int,
        default=2,
        help="Profiler-only warmup repeats over selected crops before entering torch_npu.profiler.",
    )
    parser.add_argument(
        "--profile-active-repeats",
        type=int,
        default=3,
        help="Profiler active repeats over selected crops. One profiler step is one crop forward.",
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=0,
        help="Optional development cap after crop extraction/input preprocessing. 0 means all selected crops.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if int(args.num_pages) <= 0:
        raise ValueError("--num-pages must be positive")
    modes = parse_modes(args.modes)
    if args.profile_dir is not None and str(args.profile_mode) not in modes:
        modes.append(str(args.profile_mode))

    os.environ[VISION_ATTENTION_ENV] = str(args.vision_attention)
    os.environ[VISION_PROMPT_FA_LAYOUT_ENV] = str(args.vision_prompt_fa_layout)

    model_dir = _resolve_model_dir(args.model)
    device = resolve_device(args.device)
    dtype = parse_vision_dtype(args.dtype)
    configure_npu_jit_compile(args.npu_jit_compile, device)

    page_load = load_pages_result(
        args.dataset_dir,
        page_start=int(args.page_start),
        num_pages=int(args.num_pages),
    )
    pages = page_load.pages

    layout_pages, layout_timing = build_omnidocbench_gt_layout_pages(
        pages,
        include_ignored=bool(args.include_ignored_gt),
        include_empty_gt=bool(args.include_empty_gt),
    )
    crops, crop_summary, crop_timing = build_detected_crops(pages=pages, layout_pages=layout_pages, args=args)
    if not crops:
        raise RuntimeError("OmniDocBench GT layout produced zero recognizer crops")
    raw_extracted_crop_count = int(len(crops))
    if int(args.max_crops) > 0:
        crops = crops[: int(args.max_crops)]
    if not crops:
        raise RuntimeError("zero crops after --max-crops")

    pre_cfg = load_preprocessor_config(model_dir)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    queue_inputs, input_build_summary = build_queue_inputs_from_crops(
        crops=crops,
        tokenizer=tokenizer,
        pre_cfg=pre_cfg,
        prompt_override=args.prompt,
    )
    if not queue_inputs:
        raise RuntimeError("zero queue inputs after --max-crops")
    raw_queue_input_count_before_crop_sample = int(len(queue_inputs))
    queue_inputs, crop_sample_summary = select_profile_crop_sample(queue_inputs, strategy=str(args.crop_sample))
    if not queue_inputs:
        raise RuntimeError("zero queue inputs after --crop-sample")

    setup_timing: dict[str, float] = {}
    maybe_sync(device)
    start = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, device=device)
    maybe_sync(device)
    setup_timing["recognizer_model_load_s"] = time.perf_counter() - start

    vision_forward: Callable[[torch.Tensor], torch.Tensor] | None = None
    vision_compile_summary: dict[str, Any]
    if str(args.vision_compile_backend) != "none" and len(queue_inputs) != 1:
        raise ValueError(
            "--vision-compile-backend is shape-specialized and currently requires exactly one selected crop. "
            "Use --crop-sample small_only for the requested small-crop compile/profiler experiment."
        )
    if str(args.vision_compile_backend) != "none":
        target_item = queue_inputs[0]
        eager_ref: torch.Tensor | None = None
        if bool(args.vision_compile_validate):
            maybe_sync(device)
            eager_start = time.perf_counter()
            eager_ref = run_vision_one(model=model, item=target_item, device=device)
            maybe_sync(device)
            eager_ref_s = time.perf_counter() - eager_start
        else:
            eager_ref_s = None

        vision_forward, vision_compile_summary = compile_single_crop_vision_forward(
            model=model,
            item=target_item,
            device=device,
            backend_name=str(args.vision_compile_backend),
        )
        maybe_sync(device)
        first_call_start = time.perf_counter()
        compiled_first = run_vision_one(model=model, item=target_item, device=device, vision_forward=vision_forward)
        maybe_sync(device)
        compiled_first_call_s = time.perf_counter() - first_call_start
        vision_compile_summary["compiled_first_call_s"] = float(compiled_first_call_s)
        if eager_ref is not None:
            diff = (eager_ref.float() - compiled_first.float()).abs()
            vision_compile_summary["validation"] = {
                "enabled": True,
                "eager_reference_s": float(eager_ref_s),
                "max_abs_diff": float(diff.max().detach().cpu().item()),
                "mean_abs_diff": float(diff.mean().detach().cpu().item()),
                "allclose_atol_5e_2_rtol_5e_2": bool(
                    torch.allclose(eager_ref.float(), compiled_first.float(), atol=5e-2, rtol=5e-2)
                ),
                "eager_shape": [int(dim) for dim in eager_ref.shape],
                "compiled_shape": [int(dim) for dim in compiled_first.shape],
            }
        else:
            vision_compile_summary["validation"] = {"enabled": False}
        setup_timing["vision_compile_wrapper_s"] = float(vision_compile_summary.get("compile_wrapper_s", 0.0))
        setup_timing["vision_compile_first_call_s"] = float(compiled_first_call_s)
    else:
        vision_forward, vision_compile_summary = compile_single_crop_vision_forward(
            model=model,
            item=queue_inputs[0],
            device=device,
            backend_name="none",
        )

    warmup = warmup_vision(
        model=model,
        inputs=queue_inputs,
        device=device,
        warmup_items=int(args.warmup_items),
        vision_forward=vision_forward,
    )

    mode_results: dict[str, Any] = {}
    for mode in modes:
        mode_results[mode] = run_mode(
            mode,
            model=model,
            inputs=queue_inputs,
            device=device,
            repeat_count=max(1, int(args.benchmark_repeats)),
            vision_forward=vision_forward,
        )

    comparisons: dict[str, Any] = {}
    if "sync_per_crop" in mode_results and "unsynced_loop" in mode_results:
        sync_s = float(mode_results["sync_per_crop"]["total_s"])
        unsynced_s = float(mode_results["unsynced_loop"]["total_s"])
        comparisons["unsynced_vs_sync_per_crop"] = {
            "speedup": (sync_s / unsynced_s) if unsynced_s > 0 else None,
            "saved_s": float(sync_s - unsynced_s),
            "sync_per_crop_total_s": sync_s,
            "unsynced_loop_total_s": unsynced_s,
            "note": (
                "This isolates per-crop device synchronization overhead for the same crop/preprocess/model path. "
                "Run order and warmup are reported because first-use kernel behavior can still affect small runs."
            ),
        }

    profiler_summary: dict[str, Any] = {"enabled": False}
    if args.profile_dir is not None:
        profiler_summary = profile_vision_mode(
            profile_root=args.profile_dir,
            profile_mode=str(args.profile_mode),
            profile_metric=str(args.profile_metric),
            model=model,
            inputs=queue_inputs,
            device=device,
            dtype=dtype,
            warmup_repeats=int(args.profile_warmup_repeats),
            active_repeats=int(args.profile_active_repeats),
            vision_forward=vision_forward,
        )
        baseline = mode_results[str(args.profile_mode)]
        profiled_context = profiler_summary["profiled_context_wall_result"]
        profiled_forward_sync = profiler_summary["profiled_forward_sync_result"]
        baseline_total_s = float(baseline["total_s"])
        profiled_context_total_s = float(profiled_context["total_s"])
        profiled_forward_sync_total_s = float(profiled_forward_sync["total_s"])
        baseline_items_per_s = float(baseline["items_per_s"])
        profiled_context_items_per_s = float(profiled_context["items_per_s"])
        profiled_forward_sync_items_per_s = float(profiled_forward_sync["items_per_s"])
        baseline_vision_tokens_per_s = float(baseline["vision_tokens_per_s"])
        profiled_context_vision_tokens_per_s = float(profiled_context["vision_tokens_per_s"])
        profiled_forward_sync_vision_tokens_per_s = float(profiled_forward_sync["vision_tokens_per_s"])
        baseline_forward_count = int(baseline["count"])
        profiled_forward_count = int(profiled_context["count"])
        comparisons[f"profiled_vs_unprofiled_{args.profile_mode}"] = {
            "mode": str(args.profile_mode),
            "baseline_forward_count": baseline_forward_count,
            "profiled_forward_count": profiled_forward_count,
            "forward_count_match": bool(baseline_forward_count == profiled_forward_count),
            "baseline_total_s": baseline_total_s,
            "profiled_context_total_s": profiled_context_total_s,
            "profiled_forward_sync_total_s": profiled_forward_sync_total_s,
            "profiled_context_total_s_over_baseline": (
                profiled_context_total_s / baseline_total_s if baseline_total_s > 0 else None
            ),
            "profiled_forward_sync_total_s_over_baseline": (
                profiled_forward_sync_total_s / baseline_total_s if baseline_total_s > 0 else None
            ),
            "baseline_items_per_s": baseline_items_per_s,
            "profiled_context_items_per_s": profiled_context_items_per_s,
            "profiled_forward_sync_items_per_s": profiled_forward_sync_items_per_s,
            "profiled_context_items_per_s_over_baseline": (
                profiled_context_items_per_s / baseline_items_per_s if baseline_items_per_s > 0 else None
            ),
            "profiled_forward_sync_items_per_s_over_baseline": (
                profiled_forward_sync_items_per_s / baseline_items_per_s if baseline_items_per_s > 0 else None
            ),
            "baseline_vision_tokens_per_s": baseline_vision_tokens_per_s,
            "profiled_context_vision_tokens_per_s": profiled_context_vision_tokens_per_s,
            "profiled_forward_sync_vision_tokens_per_s": profiled_forward_sync_vision_tokens_per_s,
            "profiled_context_vision_tokens_per_s_over_baseline": (
                profiled_context_vision_tokens_per_s / baseline_vision_tokens_per_s
                if baseline_vision_tokens_per_s > 0
                else None
            ),
            "profiled_forward_sync_vision_tokens_per_s_over_baseline": (
                profiled_forward_sync_vision_tokens_per_s / baseline_vision_tokens_per_s
                if baseline_vision_tokens_per_s > 0
                else None
            ),
            "profile_context_wall_s": profiler_summary.get("profile_context_wall_s"),
            "profile_active_loop_wall_s": profiler_summary.get("profile_active_loop_wall_s"),
            "profile_forward_sync_sum_s": profiler_summary.get("profile_forward_sync_sum_s"),
            "profile_profiler_step_sum_s": profiler_summary.get("profile_profiler_step_sum_s"),
            "profile_context_non_active_s": profiler_summary.get("profile_context_non_active_s"),
            "profile_active_loop_unattributed_s": profiler_summary.get("profile_active_loop_unattributed_s"),
            "profile_dir": profiler_summary.get("profile_dir"),
            "profile_metric": str(args.profile_metric),
            "benchmark_repeats": int(args.benchmark_repeats),
            "profile_warmup_repeats": int(args.profile_warmup_repeats),
            "profile_active_repeats": int(args.profile_active_repeats),
            "profiler_step_contract": profiler_summary.get("profiler_step_contract"),
            "note": (
                "Context timings include profiler.step(), trace handling, and possible export/finalization overhead. "
                "Forward-sync timings sum only the measured crop forward + device sync windows inside the profiler. "
                "Set --benchmark-repeats equal to --profile-active-repeats for the cleanest forward-count comparison."
            ),
        }

    output = {
        "experiment": "06_vision_prefill_only",
        "scope": (
            "full pages -> OmniDocBench GT layout boxes -> real OCR crops -> PaddleOCR-VL crop preprocessing "
            "-> vision device transfer + native-resolution visual encoder + adaptive MLP projector only"
        ),
        "not_run": [
            "document layout detector",
            "text token embedding",
            "image embedding scatter into text sequence",
            "mRoPE index construction",
            "KV cache allocation/prefill",
            "LM head prefill",
            "text decode/hot-swap",
            "OCR text validation or accuracy scoring",
        ],
        "model": str(model_dir),
        "dataset_dir": str(resolve_dataset_dir(args.dataset_dir)),
        "device": str(device),
        "dtype": str(dtype),
        "npu_jit_compile": str(args.npu_jit_compile),
        "vision_attention": get_vision_attention_impl(),
        "vision_prompt_fa_layout": get_vision_prompt_fa_layout(),
        "page_start": int(args.page_start),
        "page_count": int(len(pages)),
        "page_load": page_load_summary(page_load),
        "layout": {
            "source": "omnidocbench_gt",
            "uses_ground_truth_boxes": True,
            "include_ignored_gt": bool(args.include_ignored_gt),
            "include_empty_gt": bool(args.include_empty_gt),
        },
        "recognizer_crop_count": int(len(queue_inputs)),
        "raw_queue_input_count_before_crop_sample": int(raw_queue_input_count_before_crop_sample),
        "raw_extracted_crop_count_before_max_crops": int(raw_extracted_crop_count),
        "max_crops": None if int(args.max_crops) <= 0 else int(args.max_crops),
        "crop_sample": crop_sample_summary,
        "benchmark_repeats": int(args.benchmark_repeats),
        "prompt_override": args.prompt,
        "crop_summary": crop_summary,
        "input_summary": summarize_inputs(queue_inputs, merge_size=int(pre_cfg["merge_size"])),
        "layout_timing_s": layout_timing,
        "crop_timing_s": crop_timing,
        "input_build_summary_s": input_build_summary,
        "queue_input_timing_summary_s": aggregate_timing_dicts([item.timing_s for item in queue_inputs]),
        "setup_timing_s": setup_timing,
        "vision_compile": vision_compile_summary,
        "warmup": warmup,
        "mode_order": modes,
        "mode_order_note": (
            "Modes run sequentially in the listed order after warmup. For tiny runs, rerun once or set "
            "--warmup-items high enough to avoid first-use effects."
        ),
        "modes": mode_results,
        "profiler": profiler_summary,
        "comparisons": comparisons,
    }

    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True, default=json_default))
    else:
        summary = {
            "experiment": output["experiment"],
            "device": output["device"],
            "vision_attention": output["vision_attention"],
            "page_count": output["page_count"],
            "recognizer_crop_count": output["recognizer_crop_count"],
            "comparisons": comparisons,
            "modes": {
                key: {
                    "total_s": value["total_s"],
                    "items_per_s": value["items_per_s"],
                    "vision_tokens_per_s": value["vision_tokens_per_s"],
                }
                for key, value in mode_results.items()
            },
        }
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
