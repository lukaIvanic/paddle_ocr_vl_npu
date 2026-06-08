#!/usr/bin/env python3
"""Benchmark experiment-4 fixed-cohort batched static decode.

Experiment 4 starts with the serving-relevant part only: sequential no-padding
prefill for real crops, then true batched static decode over the filled KV
cache rows. It does not do slot hot-swapping yet.
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
class DecodeLoopResult:
    ids: torch.Tensor
    last_logits: torch.Tensor | None
    decode_calls: int
    finished: torch.Tensor | None
    eos_steps: torch.Tensor | None
    stopped_all_eos: bool


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
    crop_ids: list[str] | None,
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {batch_size}")
    if crop_ids:
        by_id = {str(entry["id"]): entry for entry in manifest}
        missing = [crop_id for crop_id in crop_ids if crop_id not in by_id]
        if missing:
            raise ValueError(f"unknown --crop-ids entries: {missing}")
        if len(crop_ids) != batch_size:
            raise ValueError(f"--batch-size={batch_size} but --crop-ids contains {len(crop_ids)} ids")
        return [by_id[crop_id] for crop_id in crop_ids]
    if batch_size > len(manifest):
        raise ValueError(f"--batch-size={batch_size} requires {batch_size} real crops, but manifest has {len(manifest)}")
    return manifest[:batch_size]


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

    if eos_mode == "overlap_event_flags":
        if next_token.device.type != "npu":
            raise ValueError("--eos-mode overlap_event_flags requires NPU tensors.")
        import torch_npu

        finished = hits_eos(next_token, int(eos_token_id))
        eos_steps = torch.full((batch_size,), -1, device=next_token.device, dtype=torch.int64)
        eos_steps = torch.where(finished, torch.zeros_like(eos_steps), eos_steps)
        async_cpu_flags = torch.zeros((max_decode_calls, batch_size), dtype=torch.bool, pin_memory=True)
        copy_stream = torch_npu.npu.Stream(device=next_token.device)

    for step in range(max_decode_calls):
        last_logits = decode_fn(next_token, cache_position, prefill.rope_deltas, *flat_cache)
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

        if eos_mode == "overlap_event_flags" and async_cpu_flags is not None and copy_stream is not None:
            eos_ready_event = torch_npu.npu.current_stream().record_event()
            copy_done_event = torch_npu.npu.Event()
            with torch_npu.npu.stream(copy_stream):
                copy_stream.wait_event(eos_ready_event)
                async_cpu_flags[step].copy_(finished, non_blocking=True)
                copy_done_event.record(copy_stream)
            if pending_eos_event is not None and pending_eos_step is not None:
                pending_eos_event.synchronize()
                if bool(async_cpu_flags[pending_eos_step].all().item()):
                    stopped_all_eos = True
                    break
            pending_eos_event = copy_done_event
            pending_eos_step = int(step)

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

    warm_prefill, warm_next_token, _warm_slots, _warm_nexts = make_batched_prefill(
        model=model,
        cohort=cohort,
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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop-ids", nargs="*", default=None, help="Optional explicit manifest ids; count must equal --batch-size.")
    parser.add_argument("--prompt", default=None, help="Optional prompt override. By default, each crop uses its manifest suggested_prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cache-length", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--backend", default="eager", choices=["eager", "aot_eager", "inductor", "default", "torchair"])
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--torchair-cache-dir", type=Path, default=DEFAULT_TORCHAIR_CACHE_DIR)
    parser.add_argument("--eos-mode", default="none", choices=EOS_MODE_CHOICES)
    parser.add_argument("--profile-dir", type=Path, default=None, help="Write one post-warmup torch_npu profiler capture for compiled batched decode.")
    parser.add_argument("--profile-metric", default="pipe", choices=PROFILE_METRIC_CHOICES)
    parser.add_argument("--json", action="store_true", help="Print a compact JSON summary instead of human-readable lines.")
    args = parser.parse_args()

    if int(args.max_new_tokens) <= 0:
        raise ValueError(f"--max-new-tokens must be positive, got {args.max_new_tokens}")

    model_dir = _resolve_model_dir(args.model)
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    configure_npu_jit_compile(args.npu_jit_compile, device)

    pre_cfg = load_preprocessor_config(model_dir)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    manifest = load_manifest(args.manifest)
    entries = select_manifest_entries(manifest, batch_size=int(args.batch_size), crop_ids=args.crop_ids)
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

    warm_prefill, warm_next_token, _warm_slots, _warm_nexts = make_batched_prefill(
        model=model,
        cohort=cohort,
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
        ),
    )
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
        ),
    )
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
        "tok_per_s": {
            "static_eager_decode_steps": tok_per_s(static_result.decode_calls, static_decode_s),
            "static_eager_raw_batch_tokens": tok_per_s(static_raw_token_calls, static_decode_s),
            "static_eager_effective_batch_tokens": tok_per_s(static_effective_token_calls, static_decode_s),
            "compiled_decode_steps": tok_per_s(compiled_result.decode_calls, compiled_decode_s),
            "compiled_raw_batch_tokens": tok_per_s(compiled_raw_token_calls, compiled_decode_s),
            "compiled_effective_batch_tokens": tok_per_s(compiled_effective_token_calls, compiled_decode_s),
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
    print("tok_per_s=" + json.dumps(summary["tok_per_s"], sort_keys=True))
    if profile_summary is not None:
        print("profile=" + json.dumps(profile_summary, sort_keys=True, default=json_default))
    print("compiled_texts=" + repr(summary["texts"]["compiled_trimmed"]))


if __name__ == "__main__":
    main()
