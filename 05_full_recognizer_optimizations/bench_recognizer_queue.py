#!/usr/bin/env python3
"""Benchmark a queued full-recognizer path for real crop inputs.

This script measures the experiment-5 serving shape:

1. Build real crop inputs from a manifest.
2. Run CPU preprocessing and prompt construction for all crops.
3. Build all per-crop NPU static-cache prefill states.
4. Decode the ready states through one compiled static decode slot.

The first implementation intentionally supports active batch size 1 only. That
keeps the scheduler simple while still measuring the important experiment-5
split between preprocessing, vision/projector/prefill, and compiled decode.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tokenizers import Tokenizer

from bench_stage_timing import load_manifest, select_manifest_entries, stats
from local_modeling_paddleocr_vl import (
    DECODE_ATTENTION,
    DECODE_CACHE_UPDATE,
    LocalPaddleOCRVLForConditionalGeneration,
    LocalPaddleOCRVLStaticCache,
    _resolve_model_dir,
    cast_decode_linear_weights_to_nz,
    get_vision_attention_impl,
    get_vision_prompt_fa_layout,
)
from probe_static_compile import DEFAULT_TORCHAIR_CACHE_DIR, compile_decode_module, maybe_sync
from run_local_recognition import (
    NPU_JIT_COMPILE_CHOICES,
    build_inputs,
    configure_npu_jit_compile,
    load_preprocessor_config,
    parse_dtype,
    resolve_device,
    smart_resize,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_CHOICES = ("raw_eager", "eager", "aot_eager", "inductor", "default", "torchair")
EOS_MODE_CHOICES = ("none", "overlap_event_flags")


@dataclass
class QueueInput:
    entry: dict[str, Any]
    crop_path: Path
    prompt: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    timing_s: dict[str, float]


@dataclass
class ReadyItem:
    input_item: QueueInput
    cache: LocalPaddleOCRVLStaticCache
    rope_deltas: torch.Tensor
    next_cache_position: torch.Tensor
    next_token: torch.Tensor
    timing_s: dict[str, float]
    vision_tokens: int
    projected_image_tokens: int


@dataclass
class DecodedDeviceItem:
    item: ReadyItem
    token_tensors: list[torch.Tensor]
    decode_calls: int


@dataclass
class DecodedItem:
    item: ReadyItem
    token_ids: list[int]
    trimmed_token_ids: list[int]
    generated_text: str
    decode_calls: int
    eos_hit: bool
    length_cap_hit: bool
    first_eos_position: int | None
    postprocess_s: float


class PhaseTimer:
    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.timings: dict[str, float] = {}

    def measure(self, name: str, fn: Callable[[], Any]) -> Any:
        if self.device is not None:
            maybe_sync(self.device)
        start = time.perf_counter()
        result = fn()
        if self.device is not None:
            maybe_sync(self.device)
        self.timings[name] = self.timings.get(name, 0.0) + (time.perf_counter() - start)
        return result


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def repo_path(path: Path) -> Path:
    path = path.expanduser()
    if path.exists():
        return path
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate
    return path


def preprocess_crop_timed(crop_path: Path, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    timing: dict[str, float] = {}
    start = time.perf_counter()
    with Image.open(crop_path) as opened:
        image = opened.copy()
    timing["image_read_s"] = time.perf_counter() - start

    start = time.perf_counter()
    if cfg["do_convert_rgb"]:
        image = image.convert("RGB")
    width, height = image.size
    patch_size = int(cfg["patch_size"])
    merge_size = int(cfg["merge_size"])
    temporal_patch_size = int(cfg["temporal_patch_size"])
    if temporal_patch_size != 1:
        raise ValueError(f"temporal_patch_size must be 1 for this recognizer path, got {temporal_patch_size}")

    resized_height, resized_width = height, width
    if cfg["do_resize"]:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=int(cfg["min_pixels"]),
            max_pixels=int(cfg["max_pixels"]),
        )
        image = image.resize((resized_width, resized_height), Image.Resampling(int(cfg["resample"])))

    array = np.asarray(image)
    if cfg["do_rescale"]:
        array = array.astype(np.float32) * float(cfg["rescale_factor"])
    else:
        array = array.astype(np.float32)
    if cfg["do_normalize"]:
        mean = np.array(cfg["image_mean"], dtype=np.float32)
        std = np.array(cfg["image_std"], dtype=np.float32)
        array = (array - mean) / std

    patches = array.transpose(2, 0, 1)[None, ...]
    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size
    patches = patches.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h,
        patch_size,
        grid_w,
        patch_size,
    )
    patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
    flatten_patches = patches.reshape(grid_t * grid_h * grid_w, channel, patch_size, patch_size)
    pixel_values = torch.from_numpy(flatten_patches)
    image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)
    timing["preprocess_cpu_s"] = time.perf_counter() - start
    return pixel_values, image_grid_thw, timing


def build_queue_inputs(
    *,
    entries: list[dict[str, Any]],
    manifest_path: Path,
    tokenizer: Tokenizer,
    pre_cfg: dict[str, Any],
    prompt_override: str | None,
) -> tuple[list[QueueInput], dict[str, Any]]:
    manifest_path = repo_path(manifest_path)
    crops_dir = manifest_path.parent
    inputs: list[QueueInput] = []
    total_start = time.perf_counter()
    for entry in entries:
        crop_path = Path(str(entry["file"]))
        if not crop_path.is_absolute():
            crop_path = crops_dir / crop_path
        if not crop_path.exists():
            raise FileNotFoundError(f"crop not found for {entry.get('id')}: {crop_path}")
        prompt = str(prompt_override if prompt_override is not None else entry.get("suggested_prompt", "OCR:"))
        pixel_values, image_grid_thw, timing = preprocess_crop_timed(crop_path, pre_cfg)
        token_start = time.perf_counter()
        input_ids, attention_mask = build_inputs(
            tokenizer,
            image_grid_thw,
            prompt,
            merge_size=int(pre_cfg["merge_size"]),
        )
        timing["token_build_s"] = time.perf_counter() - token_start
        timing["input_build_s"] = timing["image_read_s"] + timing["preprocess_cpu_s"] + timing["token_build_s"]
        inputs.append(
            QueueInput(
                entry=entry,
                crop_path=crop_path,
                prompt=prompt,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                timing_s=timing,
            )
        )
    summary = aggregate_input_build_timings(inputs)
    summary["input_build_wall_s"] = float(time.perf_counter() - total_start)
    return inputs, summary


def aggregate_timing_dicts(rows: list[dict[str, float]]) -> dict[str, Any]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    return {key: stats([float(row[key]) for row in rows if key in row]) for key in sorted(keys)}


def aggregate_input_build_timings(inputs: list[QueueInput]) -> dict[str, Any]:
    return aggregate_timing_dicts([item.timing_s for item in inputs])


def aggregate_ready_timings(items: list[ReadyItem]) -> dict[str, Any]:
    return aggregate_timing_dicts([item.timing_s for item in items])


def prompt_token_summary(inputs: list[QueueInput], *, max_new_tokens: int, cache_length: int) -> dict[str, Any]:
    prompt_lengths = [int(item.input_ids.shape[1]) for item in inputs]
    required = [length + max(0, int(max_new_tokens) - 1) for length in prompt_lengths]
    overflows = [
        {
            "id": str(item.entry.get("id")),
            "file": str(item.crop_path),
            "input_tokens": int(prompt_len),
            "required_cache_length": int(req),
        }
        for item, prompt_len, req in zip(inputs, prompt_lengths, required)
        if int(req) > int(cache_length)
    ]
    return {
        "cache_length": int(cache_length),
        "max_new_tokens": int(max_new_tokens),
        "input_tokens": {
            "min": int(min(prompt_lengths)) if prompt_lengths else 0,
            "max": int(max(prompt_lengths)) if prompt_lengths else 0,
            "stats": stats([float(value) for value in prompt_lengths]),
            "per_item": prompt_lengths,
        },
        "required_cache_length": {
            "min": int(min(required)) if required else 0,
            "max": int(max(required)) if required else 0,
            "stats": stats([float(value) for value in required]),
        },
        "overflow_count": int(len(overflows)),
        "overflow_items": overflows[:16],
    }


@torch.inference_mode()
def build_ready_item(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    item: QueueInput,
    cache_length: int,
    device: torch.device,
) -> ReadyItem:
    timer = PhaseTimer(device)
    moved = timer.measure(
        "device_transfer",
        lambda: (
            item.input_ids.to(device),
            item.attention_mask.to(device),
            item.pixel_values.to(device),
            item.image_grid_thw.to(device),
        ),
    )
    input_ids, attention_mask, pixel_values, image_grid_thw = moved

    vision_inputs = timer.measure(
        "vision_prepare",
        lambda: {
            "pixel_values": pixel_values.type(model.visual.dtype).unsqueeze(0),
            "cu_seqlens": F.pad(
                torch.repeat_interleave(
                    image_grid_thw[:, 1] * image_grid_thw[:, 2],
                    image_grid_thw[:, 0],
                ).cumsum(dim=0, dtype=torch.int32),
                (1, 0),
                value=0,
            ),
        },
    )
    vision_model = model.visual.vision_model
    vision_hidden = timer.measure(
        "vision_embeddings",
        lambda: vision_model.embeddings(vision_inputs["pixel_values"], image_grid_thw=image_grid_thw),
    )
    vision_hidden = timer.measure(
        "vision_encoder",
        lambda: vision_model.encoder(
            vision_hidden,
            cu_seqlens=vision_inputs["cu_seqlens"],
            image_grid_thw=image_grid_thw,
        ),
    )
    image_features = timer.measure("vision_post_layernorm", lambda: vision_model.post_layernorm(vision_hidden))
    image_embeds = timer.measure("adaptive_mlp_projector", lambda: model.mlp_AR(image_features, image_grid_thw))

    inputs_embeds = timer.measure("text_token_embedding", lambda: model.model.embed_tokens(input_ids))

    def scatter_image_embeds() -> torch.Tensor:
        projected = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        if inputs_embeds[image_mask].numel() != projected.numel():
            raise ValueError(
                "image features and image tokens do not match: "
                f"tokens={int((input_ids == model.config.image_token_id).sum().item())} "
                f"features={int(projected.shape[0])}"
            )
        return inputs_embeds.masked_scatter(image_mask, projected)

    inputs_embeds = timer.measure("image_embed_scatter", scatter_image_embeds)
    position_ids, rope_deltas = timer.measure(
        "mrope_index",
        lambda: model.get_rope_index(input_ids, image_grid_thw, attention_mask),
    )
    cache = timer.measure(
        "static_cache_alloc",
        lambda: model.allocate_static_cache(
            batch_size=int(inputs_embeds.shape[0]),
            cache_length=int(cache_length),
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
            init_mode="zeros",
        ),
    )
    hidden_states = timer.measure(
        "text_prefill",
        lambda: model.model.forward_prefill_static(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
        ),
    )
    logits = timer.measure("prefill_lm_head", lambda: model.lm_head(hidden_states[:, -1:, :]))
    next_token = timer.measure("prefill_argmax", lambda: torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True))
    next_cache_position = torch.full((int(input_ids.shape[0]),), int(input_ids.shape[1]), device=device, dtype=torch.int64)

    timings = {key: float(value) for key, value in timer.timings.items()}
    timings["native_resolution_visual_encoder_total"] = (
        timings["vision_embeddings"] + timings["vision_encoder"] + timings["vision_post_layernorm"]
    )
    timings["vision_projector_total"] = timings["native_resolution_visual_encoder_total"] + timings["adaptive_mlp_projector"]
    timings["vision_total"] = timings["vision_prepare"] + timings["native_resolution_visual_encoder_total"]
    timings["prefill_total_excluding_device_transfer"] = (
        timings["vision_prepare"]
        + timings["native_resolution_visual_encoder_total"]
        + timings["adaptive_mlp_projector"]
        + timings["text_token_embedding"]
        + timings["image_embed_scatter"]
        + timings["mrope_index"]
        + timings["static_cache_alloc"]
        + timings["text_prefill"]
        + timings["prefill_lm_head"]
        + timings["prefill_argmax"]
    )
    timings["ready_item_total_excluding_device_transfer"] = timings["prefill_total_excluding_device_transfer"]
    timings["ready_item_total_with_device_transfer"] = timings["device_transfer"] + timings["prefill_total_excluding_device_transfer"]
    return ReadyItem(
        input_item=item,
        cache=cache,
        rope_deltas=rope_deltas,
        next_cache_position=next_cache_position,
        next_token=next_token,
        timing_s=timings,
        vision_tokens=int(image_features.shape[0]),
        projected_image_tokens=int(image_embeds.shape[0]),
    )


def hits_eos(token_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    return token_ids.reshape(-1) == int(eos_token_id)


def trim_at_eos(row: list[int], eos_token_id: int) -> tuple[list[int], int | None]:
    try:
        first = row.index(int(eos_token_id))
    except ValueError:
        return row, None
    return row[: first + 1], int(first)


def safe_decode_tokens(tokenizer: Tokenizer, row: list[int]) -> str:
    try:
        return tokenizer.decode(row, skip_special_tokens=True)
    except Exception as exc:
        return f"<decode_error {type(exc).__name__}: {exc}>"


@torch.inference_mode()
def decode_ready_item(
    *,
    decode_fn: Callable,
    ready: ReadyItem,
    eos_token_id: int,
    max_new_tokens: int,
    eos_mode: str,
) -> DecodedDeviceItem:
    if eos_mode not in EOS_MODE_CHOICES:
        raise ValueError(f"unsupported eos_mode={eos_mode!r}")
    if int(max_new_tokens) <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

    device = ready.next_token.device
    next_token = ready.next_token
    cache_position = ready.next_cache_position
    flat_cache = ready.cache.flat_tensors()
    generated = [next_token]
    decode_calls = 0
    max_decode_calls = max(0, int(max_new_tokens) - 1)

    finished = hits_eos(next_token, int(eos_token_id)) if eos_mode == "overlap_event_flags" else None
    if eos_mode == "overlap_event_flags" and bool(finished.detach().cpu().all().item()):
        return DecodedDeviceItem(
            item=ready,
            token_tensors=generated,
            decode_calls=0,
        )

    use_npu_overlap = eos_mode == "overlap_event_flags" and device.type == "npu"
    async_cpu_flags = None
    copy_stream = None
    pending_eos_event = None
    pending_eos_step = None
    if use_npu_overlap and max_decode_calls > 0:
        import torch_npu

        async_cpu_flags = torch.zeros((max_decode_calls, 1), dtype=torch.bool, pin_memory=True)
        copy_stream = torch_npu.npu.Stream(device=device)

    stopped = False
    for step in range(max_decode_calls):
        logits = decode_fn(next_token, cache_position, ready.rope_deltas, *flat_cache)
        sampled = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
        if eos_mode == "overlap_event_flags":
            assert finished is not None
            active_before_step = ~finished
            eos_fill = torch.full_like(sampled, int(eos_token_id))
            next_token = torch.where(active_before_step.view(-1, 1), sampled, eos_fill)
            new_hits = hits_eos(next_token, int(eos_token_id)) & active_before_step
            finished = finished | new_hits
        else:
            next_token = sampled
        generated.append(next_token)
        cache_position = cache_position + 1
        decode_calls += 1

        if use_npu_overlap and async_cpu_flags is not None and copy_stream is not None:
            eos_ready_event = torch_npu.npu.current_stream().record_event()
            copy_done_event = torch_npu.npu.Event()
            with torch_npu.npu.stream(copy_stream):
                copy_stream.wait_event(eos_ready_event)
                async_cpu_flags[step].copy_(finished, non_blocking=True)
                copy_done_event.record(copy_stream)
            if pending_eos_event is not None and pending_eos_step is not None:
                pending_eos_event.synchronize()
                if bool(async_cpu_flags[pending_eos_step].all().item()):
                    stopped = True
                    break
            pending_eos_event = copy_done_event
            pending_eos_step = int(step)
        elif eos_mode == "overlap_event_flags":
            if bool(finished.detach().cpu().all().item()):
                stopped = True
                break

    if use_npu_overlap and not stopped and pending_eos_event is not None and pending_eos_step is not None:
        pending_eos_event.synchronize()
        if bool(async_cpu_flags[pending_eos_step].all().item()):
            stopped = True

    return DecodedDeviceItem(
        item=ready,
        token_tensors=generated,
        decode_calls=int(decode_calls),
    )


def materialize_decoded_item(
    *,
    device_item: DecodedDeviceItem,
    tokenizer: Tokenizer,
    eos_token_id: int,
    max_new_tokens: int,
) -> DecodedItem:
    start = time.perf_counter()
    row = [int(value) for value in torch.cat(device_item.token_tensors, dim=1)[0].detach().cpu().tolist()]
    trimmed, first_eos = trim_at_eos(row, int(eos_token_id))
    return DecodedItem(
        item=device_item.item,
        token_ids=row,
        trimmed_token_ids=trimmed,
        generated_text=safe_decode_tokens(tokenizer, trimmed),
        decode_calls=int(device_item.decode_calls),
        eos_hit=first_eos is not None,
        length_cap_hit=first_eos is None and len(row) >= int(max_new_tokens),
        first_eos_position=first_eos,
        postprocess_s=float(time.perf_counter() - start),
    )


@torch.inference_mode()
def validate_outputs(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    decoded: list[DecodedItem],
    device: torch.device,
    cache_length: int,
    max_new_tokens: int,
    eos_token_id: int,
    max_items: int,
) -> dict[str, Any]:
    if max_items == 0:
        return {
            "enabled": False,
            "all_required_checks_passed": False,
            "reason": "validation disabled",
        }
    limit = len(decoded) if int(max_items) < 0 else min(len(decoded), int(max_items))
    mismatches = []
    start = time.perf_counter()
    for idx, decoded_item in enumerate(decoded[:limit]):
        source = decoded_item.item.input_item
        reference = model.generate_ids_static(
            input_ids=source.input_ids.to(device),
            attention_mask=source.attention_mask.to(device),
            pixel_values=source.pixel_values.to(device),
            image_grid_thw=source.image_grid_thw.to(device),
            max_new_tokens=int(max_new_tokens),
            cache_length=int(cache_length),
            eos_token_id=int(eos_token_id),
        )
        ref_row = [int(value) for value in reference[0].detach().cpu().tolist()]
        ref_trimmed, _ = trim_at_eos(ref_row, int(eos_token_id))
        if ref_trimmed != decoded_item.trimmed_token_ids:
            first_mismatch = None
            for pos, (left, right) in enumerate(zip(decoded_item.trimmed_token_ids, ref_trimmed)):
                if int(left) != int(right):
                    first_mismatch = {"position": int(pos), "queue": int(left), "reference": int(right)}
                    break
            if first_mismatch is None and len(decoded_item.trimmed_token_ids) != len(ref_trimmed):
                pos = min(len(decoded_item.trimmed_token_ids), len(ref_trimmed))
                first_mismatch = {
                    "position": int(pos),
                    "queue": None if pos >= len(decoded_item.trimmed_token_ids) else int(decoded_item.trimmed_token_ids[pos]),
                    "reference": None if pos >= len(ref_trimmed) else int(ref_trimmed[pos]),
                }
            mismatches.append(
                {
                    "item": int(idx),
                    "id": str(source.entry.get("id")),
                    "first_mismatch": first_mismatch,
                }
            )
    maybe_sync(device)
    return {
        "enabled": True,
        "reference": "model.generate_ids_static_trimmed",
        "validated_items": int(limit),
        "all_required_checks_passed": bool(not mismatches and limit == len(decoded)),
        "mismatch_count": int(len(mismatches)),
        "first_mismatches": mismatches[:8],
        "elapsed_s": float(time.perf_counter() - start),
    }


def token_range_summary(rows: list[list[int]], *, vocab_size: int) -> dict[str, Any]:
    invalid_samples = []
    token_count = 0
    min_id = None
    max_id = None
    invalid_count = 0
    for row_idx, row in enumerate(rows):
        for col_idx, value in enumerate(row):
            token_count += 1
            value = int(value)
            min_id = value if min_id is None else min(min_id, value)
            max_id = value if max_id is None else max(max_id, value)
            if value < 0 or value >= int(vocab_size):
                invalid_count += 1
                if len(invalid_samples) < 8:
                    invalid_samples.append({"item": int(row_idx), "position": int(col_idx), "value": int(value)})
    return {
        "vocab_size": int(vocab_size),
        "token_count": int(token_count),
        "min_id": min_id,
        "max_id": max_id,
        "invalid_count": int(invalid_count),
        "invalid_samples": invalid_samples,
    }


def tok_per_s(tokens: int, seconds: float) -> float | None:
    if seconds <= 0:
        return None
    return float(tokens) / float(seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "crops" / "hotswap_100_manifest.json")
    parser.add_argument("--num-items", type=int, default=100)
    parser.add_argument("--crop-ids", nargs="*", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cache-length", type=int, default=1024)
    parser.add_argument("--active-batch-size", type=int, default=1)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--decode-backend", default="torchair", choices=BACKEND_CHOICES)
    parser.add_argument("--eos-mode", default="overlap_event_flags", choices=EOS_MODE_CHOICES)
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--torchair-cache-dir", type=Path, default=DEFAULT_TORCHAIR_CACHE_DIR)
    parser.add_argument(
        "--validation-items",
        type=int,
        default=-1,
        help="Number of queue outputs to validate against direct static generation; -1 validates all.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if int(args.active_batch_size) != 1:
        raise ValueError("bench_recognizer_queue.py currently supports only --active-batch-size 1")
    if int(args.num_items) <= 0:
        raise ValueError(f"--num-items must be positive, got {args.num_items}")
    if int(args.max_new_tokens) <= 0:
        raise ValueError(f"--max-new-tokens must be positive, got {args.max_new_tokens}")
    if int(args.cache_length) <= 0:
        raise ValueError(f"--cache-length must be positive, got {args.cache_length}")

    model_dir = _resolve_model_dir(args.model)
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    configure_npu_jit_compile(args.npu_jit_compile, device)

    pre_cfg = load_preprocessor_config(model_dir)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    manifest = load_manifest(args.manifest)
    entries = select_manifest_entries(
        manifest,
        num_items=int(args.num_items),
        crop_ids=args.crop_ids,
    )
    queue_inputs, input_build_summary = build_queue_inputs(
        entries=entries,
        manifest_path=args.manifest,
        tokenizer=tokenizer,
        pre_cfg=pre_cfg,
        prompt_override=args.prompt,
    )
    cache_preflight = prompt_token_summary(
        queue_inputs,
        max_new_tokens=int(args.max_new_tokens),
        cache_length=int(args.cache_length),
    )
    if int(cache_preflight["overflow_count"]) > 0:
        output = {
            "experiment": "05_full_recognizer_queue",
            "error": "cache_length_too_small",
            "num_items": int(len(queue_inputs)),
            "cache_preflight": cache_preflight,
            "input_build_summary_s": input_build_summary,
        }
        print(json.dumps(output, indent=2, sort_keys=True, default=json_default))
        return

    maybe_sync(device)
    model_load_start = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, device=device)
    maybe_sync(device)
    model_load_s = time.perf_counter() - model_load_start

    maybe_sync(device)
    weight_format_start = time.perf_counter()
    weight_format_meta = cast_decode_linear_weights_to_nz(model)
    maybe_sync(device)
    weight_format_meta["setup_s"] = time.perf_counter() - weight_format_start

    flat_decode = model.make_flat_static_decode_module().eval()
    maybe_sync(device)
    compile_wrapper_start = time.perf_counter()
    decode_fn, compile_meta = compile_decode_module(
        flat_decode,
        backend_name=str(args.decode_backend),
        device=device,
        cache_root=args.torchair_cache_dir,
        batch_size=1,
        cache_length=int(args.cache_length),
    )
    maybe_sync(device)
    compile_wrapper_s = time.perf_counter() - compile_wrapper_start

    warm_cache = model.allocate_static_cache(
        batch_size=1,
        cache_length=int(args.cache_length),
        device=device,
        dtype=dtype,
        init_mode="zeros",
    )
    warm_input = torch.zeros((1, 1), device=device, dtype=torch.int64)
    warm_position = torch.full((1,), min(cache_preflight["input_tokens"]["max"], int(args.cache_length) - 1), device=device, dtype=torch.int64)
    warm_rope = torch.zeros((1, 1), device=device, dtype=torch.int64)
    maybe_sync(device)
    compile_first_start = time.perf_counter()
    decode_fn(warm_input, warm_position, warm_rope, *warm_cache.flat_tensors())
    maybe_sync(device)
    compile_first_s = time.perf_counter() - compile_first_start

    ready_start = time.perf_counter()
    ready_items = [
        build_ready_item(
            model=model,
            item=item,
            cache_length=int(args.cache_length),
            device=device,
        )
        for item in queue_inputs
    ]
    maybe_sync(device)
    ready_bank_build_s = time.perf_counter() - ready_start

    eos_token_id = int(model.config.eos_token_id)
    decode_start = time.perf_counter()
    decoded_device_items = [
        decode_ready_item(
            decode_fn=decode_fn,
            ready=ready,
            eos_token_id=eos_token_id,
            max_new_tokens=int(args.max_new_tokens),
            eos_mode=str(args.eos_mode),
        )
        for ready in ready_items
    ]
    maybe_sync(device)
    decode_queue_s = time.perf_counter() - decode_start

    postprocess_start = time.perf_counter()
    decoded_items = [
        materialize_decoded_item(
            device_item=item,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            max_new_tokens=int(args.max_new_tokens),
        )
        for item in decoded_device_items
    ]
    decode_output_postprocess_s = time.perf_counter() - postprocess_start

    validation = validate_outputs(
        model=model,
        decoded=decoded_items,
        device=device,
        cache_length=int(args.cache_length),
        max_new_tokens=int(args.max_new_tokens),
        eos_token_id=eos_token_id,
        max_items=int(args.validation_items),
    )
    trimmed_rows = [item.trimmed_token_ids for item in decoded_items]
    token_summary = token_range_summary(trimmed_rows, vocab_size=int(model.config.text_config.vocab_size))
    total_decode_calls = sum(int(item.decode_calls) for item in decoded_items)
    effective_decode_token_calls = sum(max(0, len(item.trimmed_token_ids) - 1) for item in decoded_items)
    generated_new_tokens = sum(len(item.trimmed_token_ids) for item in decoded_items)
    input_build_wall_s = float(input_build_summary.get("input_build_wall_s", 0.0))
    total_excluding_setup_s = input_build_wall_s + ready_bank_build_s + decode_queue_s + decode_output_postprocess_s

    output = {
        "experiment": "05_full_recognizer_queue",
        "model": str(model_dir),
        "device": str(device),
        "dtype": str(dtype),
        "decode_backend": str(args.decode_backend),
        "decode_attention": DECODE_ATTENTION if device.type == "npu" else "manual",
        "decode_cache_update": DECODE_CACHE_UPDATE if device.type == "npu" else "per_row_copy",
        "eos_mode": str(args.eos_mode),
        "eos_token_id": eos_token_id,
        "npu_jit_compile": str(args.npu_jit_compile),
        "vision_attention": get_vision_attention_impl(),
        "vision_prompt_fa_layout": get_vision_prompt_fa_layout(),
        "active_batch_size": int(args.active_batch_size),
        "scheduler": "single_slot_ready_state_queue",
        "num_items": int(len(decoded_items)),
        "max_new_tokens": int(args.max_new_tokens),
        "cache_length": int(args.cache_length),
        "cache_preflight": cache_preflight,
        "setup_timing_s": {
            "model_load": float(model_load_s),
            "decode_weight_format": float(weight_format_meta.get("setup_s", 0.0) or 0.0),
            "compile_wrapper": float(compile_wrapper_s),
            "compile_first_call": float(compile_first_s),
        },
        "linear_weight_format": weight_format_meta,
        "compile": compile_meta,
        "input_build_summary_s": input_build_summary,
        "ready_item_timing_summary_s": aggregate_ready_timings(ready_items),
        "phase_timing_s": {
            "input_build_wall": input_build_wall_s,
            "ready_bank_build": float(ready_bank_build_s),
            "decode_queue": float(decode_queue_s),
            "decode_output_postprocess": float(decode_output_postprocess_s),
            "total_excluding_setup": float(total_excluding_setup_s),
            "validation": float(validation.get("elapsed_s", 0.0) or 0.0),
        },
        "throughput": {
            "items_per_s_excluding_setup": tok_per_s(len(decoded_items), total_excluding_setup_s),
            "items_per_s_decode_only": tok_per_s(len(decoded_items), decode_queue_s),
            "decode_calls_per_s": tok_per_s(total_decode_calls, decode_queue_s),
            "effective_decode_tokens_per_s": tok_per_s(effective_decode_token_calls, decode_queue_s),
            "generated_new_tokens_per_s_decode_only": tok_per_s(generated_new_tokens, decode_queue_s),
        },
        "decode_summary": {
            "decode_calls": int(total_decode_calls),
            "effective_decode_token_calls": int(effective_decode_token_calls),
            "generated_new_tokens": int(generated_new_tokens),
            "eos_hit_count": int(sum(1 for item in decoded_items if item.eos_hit)),
            "length_cap_hit_count": int(sum(1 for item in decoded_items if item.length_cap_hit)),
            "trimmed_new_tokens": stats([float(len(item.trimmed_token_ids)) for item in decoded_items]),
            "raw_new_tokens": stats([float(len(item.token_ids)) for item in decoded_items]),
        },
        "token_id_range": token_summary,
        "correctness": {
            **validation,
            "invalid_token_count": int(token_summary["invalid_count"]),
            "all_required_checks_passed": bool(
                validation.get("all_required_checks_passed", False)
                and int(token_summary["invalid_count"]) == 0
            ),
        },
        "items": [
            {
                "idx": int(idx),
                "id": str(decoded.item.input_item.entry.get("id")),
                "file": str(decoded.item.input_item.crop_path),
                "category_type": decoded.item.input_item.entry.get("category_type"),
                "prompt": decoded.item.input_item.prompt,
                "crop_size": decoded.item.input_item.entry.get("crop_size"),
                "input_tokens": int(decoded.item.input_item.input_ids.shape[1]),
                "vision_tokens": int(decoded.item.vision_tokens),
                "projected_image_tokens": int(decoded.item.projected_image_tokens),
                "generated_tokens_raw": int(len(decoded.token_ids)),
                "generated_tokens_trimmed": int(len(decoded.trimmed_token_ids)),
                "decode_calls": int(decoded.decode_calls),
                "eos_hit": bool(decoded.eos_hit),
                "length_cap_hit": bool(decoded.length_cap_hit),
                "first_eos_position": decoded.first_eos_position,
                "generated_text": decoded.generated_text,
                "generated_ids_trimmed": decoded.trimmed_token_ids,
                "decode_output_postprocess_s": float(decoded.postprocess_s),
                "preprocess_timing_s": decoded.item.input_item.timing_s,
                "ready_timing_s": decoded.item.timing_s,
                "ground_truth_source": decoded.item.input_item.entry.get("ground_truth_source"),
                "ground_truth": decoded.item.input_item.entry.get("ground_truth"),
            }
            for idx, decoded in enumerate(decoded_items)
        ],
        "texts": {
            "trimmed": [item.generated_text for item in decoded_items],
            "sample": [item.generated_text for item in decoded_items[: min(8, len(decoded_items))]],
        },
        "stage_notes": {
            "input_build_wall": "CPU crop image read/decode, resize/normalize/patchify, and prompt token construction for all selected crops.",
            "ready_bank_build": "Sequential per-crop device transfer plus vision/projector/text prefill into NPU static cache states.",
            "decode_queue": "Single active compiled decode slot over already-prefilled states; no vision/prefill work occurs inside this phase.",
            "validation": "Direct local static generation is run after measured phases and is not included in throughput.",
            "cache_length": "Static KV cache is preflighted against input_tokens + max_new_tokens - 1; overflow is a hard error.",
        },
    }
    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True, default=json_default))
        return

    print(f"experiment={output['experiment']} device={output['device']} dtype={output['dtype']} backend={output['decode_backend']}")
    print(f"num_items={output['num_items']} cache_length={output['cache_length']} max_new_tokens={output['max_new_tokens']}")
    print("cache_preflight=" + json.dumps(output["cache_preflight"], sort_keys=True))
    print("phase_timing_s=" + json.dumps(output["phase_timing_s"], sort_keys=True))
    print("throughput=" + json.dumps(output["throughput"], sort_keys=True))
    print("correctness=" + json.dumps(output["correctness"], sort_keys=True))
    print("texts_sample=" + repr(output["texts"]["sample"]))


if __name__ == "__main__":
    main()
