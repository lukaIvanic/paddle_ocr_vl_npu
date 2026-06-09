#!/usr/bin/env python3
"""Measure full recognizer stage costs for experiment 5.

This benchmark keeps layout detection and image preprocessing outside the model
timing. It uses real crop inputs, then breaks the recognition model into the
native-resolution vision transformer, adaptive MLP connector, text prefill, LM
head, and static decode stages.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from local_modeling_paddleocr_vl import (
    DECODE_ATTENTION,
    DECODE_CACHE_UPDATE,
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
    cast_decode_linear_weights_to_nz,
    get_vision_attention_impl,
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


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_CHOICES = ("raw_eager", "eager", "aot_eager", "inductor", "default", "torchair")


@dataclass
class CohortInput:
    entry: dict[str, Any]
    crop_path: Path
    prompt: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor


class StageTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.timings: dict[str, float] = {}

    def measure(self, name: str, fn: Callable[[], Any]) -> Any:
        maybe_sync(self.device)
        start = time.perf_counter()
        result = fn()
        maybe_sync(self.device)
        self.timings[name] = self.timings.get(name, 0.0) + (time.perf_counter() - start)
        return result


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def resolve_repo_path(path: Path) -> Path:
    path = path.expanduser()
    if path.exists():
        return path
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate
    return path


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return json.loads(resolve_repo_path(path).read_text(encoding="utf-8"))


def select_manifest_entries(
    manifest: list[dict[str, Any]],
    *,
    num_items: int,
    crop_ids: list[str] | None,
) -> list[dict[str, Any]]:
    if crop_ids:
        by_id = {str(entry["id"]): entry for entry in manifest}
        missing = [crop_id for crop_id in crop_ids if crop_id not in by_id]
        if missing:
            raise ValueError(f"unknown --crop-ids entries: {missing}")
        if len(crop_ids) != int(num_items):
            raise ValueError(f"--num-items={num_items} but --crop-ids contains {len(crop_ids)} ids")
        return [by_id[crop_id] for crop_id in crop_ids]
    if int(num_items) <= 0:
        raise ValueError(f"--num-items must be positive, got {num_items}")
    if int(num_items) > len(manifest):
        raise ValueError(f"requested {num_items} crops, but manifest has {len(manifest)}")
    return manifest[: int(num_items)]


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


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "sum": 0.0, "avg": None, "min": None, "p50": None, "p90": None, "max": None}
    ordered = sorted(float(value) for value in values)

    def percentile(q: float) -> float:
        idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
        return float(ordered[idx])

    return {
        "count": int(len(ordered)),
        "sum": float(sum(ordered)),
        "avg": float(sum(ordered) / len(ordered)),
        "min": float(ordered[0]),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "max": float(ordered[-1]),
    }


def aggregate_item_timings(items: list[dict[str, Any]]) -> dict[str, Any]:
    keys: set[str] = set()
    for item in items:
        keys.update((item.get("timing_s") or {}).keys())
    return {
        key: stats([float(item["timing_s"][key]) for item in items if key in item.get("timing_s", {})])
        for key in sorted(keys)
    }


def timed_static_decode(
    *,
    decode_fn: Callable,
    next_token: torch.Tensor,
    cache_position: torch.Tensor,
    rope_deltas: torch.Tensor,
    flat_cache: tuple[torch.Tensor, ...],
    max_new_tokens: int,
    device: torch.device,
    trace_decode_steps: bool,
) -> dict[str, Any]:
    generated = [next_token]
    decode_calls = 0
    last_logits = None
    decode_step_wall_s: list[float] = []
    for _ in range(max(0, int(max_new_tokens) - 1)):
        if trace_decode_steps:
            maybe_sync(device)
            step_start = time.perf_counter()
        last_logits = decode_fn(next_token, cache_position, rope_deltas, *flat_cache)
        next_token = torch.argmax(last_logits[:, -1, :].float(), dim=-1, keepdim=True)
        if trace_decode_steps:
            maybe_sync(device)
            decode_step_wall_s.append(time.perf_counter() - step_start)
        generated.append(next_token)
        cache_position = cache_position + 1
        decode_calls += 1
    ids = torch.cat(generated, dim=1)
    result = {
        "ids": ids,
        "decode_calls": int(decode_calls),
        "generated_tokens": int(ids.shape[1]),
        "decode_mode": "fixed_steps_no_eos_check",
        "last_logits_shape": None if last_logits is None else [int(v) for v in last_logits.shape],
    }
    if trace_decode_steps:
        result["decode_step_wall_s"] = decode_step_wall_s
    return result


def first_mismatch(left: list[int], right: list[int]) -> dict[str, Any] | None:
    for idx, (left_value, right_value) in enumerate(zip(left, right)):
        if int(left_value) != int(right_value):
            return {
                "position": int(idx),
                "staged": int(left_value),
                "direct_static": int(right_value),
            }
    if len(left) != len(right):
        return {
            "position": int(min(len(left), len(right))),
            "staged": None if len(left) <= len(right) else int(left[min(len(left), len(right))]),
            "direct_static": None if len(right) <= len(left) else int(right[min(len(left), len(right))]),
        }
    return None


@torch.inference_mode()
def direct_static_fixed_steps(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    cache_length: int,
    max_new_tokens: int,
) -> torch.Tensor:
    outputs = model.forward_static_prefill(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        cache_length=int(cache_length),
        logits_to_keep=1,
    )
    cache = outputs.cache
    rope_deltas = outputs.rope_deltas
    cache_position = outputs.next_cache_position
    next_token = torch.argmax(outputs.logits[:, -1, :].float(), dim=-1, keepdim=True)
    generated = [next_token]
    for _ in range(max(0, int(max_new_tokens) - 1)):
        outputs_decode = model.forward_static_decode(
            input_ids=next_token,
            cache=cache,
            cache_position=cache_position,
            rope_deltas=rope_deltas,
            logits_to_keep=1,
        )
        next_token = torch.argmax(outputs_decode.logits[:, -1, :].float(), dim=-1, keepdim=True)
        generated.append(next_token)
        cache_position = cache_position + 1
    return torch.cat(generated, dim=1)


@torch.inference_mode()
def run_item_stage_timing(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    decode_fn: Callable,
    item: CohortInput,
    cache_length: int,
    max_new_tokens: int,
    device: torch.device,
    trace_decode_steps: bool,
) -> dict[str, Any]:
    timer = StageTimer(device)

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
    cache_position = torch.full((int(input_ids.shape[0]),), int(input_ids.shape[1]), device=device, dtype=torch.int64)

    decode_result = timer.measure(
        "static_decode_total",
        lambda: timed_static_decode(
            decode_fn=decode_fn,
            next_token=next_token,
            cache_position=cache_position,
            rope_deltas=rope_deltas,
            flat_cache=cache.flat_tensors(),
            max_new_tokens=int(max_new_tokens),
            device=device,
            trace_decode_steps=bool(trace_decode_steps),
        ),
    )
    staged_ids = decode_result["ids"]

    validation_timer = StageTimer(device)
    direct_ids = validation_timer.measure(
        "direct_static_fixed_steps",
        lambda: direct_static_fixed_steps(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            cache_length=int(cache_length),
            max_new_tokens=int(max_new_tokens),
        ),
    )
    staged_row = [int(value) for value in staged_ids[0].detach().cpu().tolist()]
    direct_row = [int(value) for value in direct_ids[0].detach().cpu().tolist()]
    tokens_match = staged_row == direct_row

    model_prefill_total = (
        timer.timings["vision_prepare"]
        + timer.timings["vision_embeddings"]
        + timer.timings["vision_encoder"]
        + timer.timings["vision_post_layernorm"]
        + timer.timings["adaptive_mlp_projector"]
        + timer.timings["text_token_embedding"]
        + timer.timings["image_embed_scatter"]
        + timer.timings["mrope_index"]
        + timer.timings["static_cache_alloc"]
        + timer.timings["text_prefill"]
        + timer.timings["prefill_lm_head"]
        + timer.timings["prefill_argmax"]
    )
    timer.timings["native_resolution_visual_encoder_total"] = (
        timer.timings["vision_embeddings"]
        + timer.timings["vision_encoder"]
        + timer.timings["vision_post_layernorm"]
    )
    timer.timings["vision_total"] = timer.timings["vision_prepare"] + timer.timings["native_resolution_visual_encoder_total"]
    timer.timings["prefill_total_excluding_device_transfer"] = model_prefill_total
    timer.timings["model_total_excluding_device_transfer"] = model_prefill_total + timer.timings["static_decode_total"]

    grid = [int(value) for value in image_grid_thw.reshape(-1).detach().cpu().tolist()]
    return {
        "id": str(item.entry.get("id")),
        "file": str(item.crop_path),
        "category_type": item.entry.get("category_type"),
        "prompt": item.prompt,
        "crop_size": item.entry.get("crop_size"),
        "input_tokens": int(input_ids.shape[1]),
        "image_grid_thw": grid,
        "vision_tokens": int(image_features.shape[0]),
        "projected_image_tokens": int(image_embeds.shape[0]),
        "generated_tokens": int(decode_result["generated_tokens"]),
        "decode_calls": int(decode_result["decode_calls"]),
        "decode_mode": str(decode_result["decode_mode"]),
        "decode_step_wall_s": [float(value) for value in decode_result.get("decode_step_wall_s", [])],
        "correctness": {
            "reference": "direct_local_static_fixed_steps",
            "tokens_match": bool(tokens_match),
            "first_mismatch": first_mismatch(staged_row, direct_row),
            "validation_timing_s": {
                key: float(value)
                for key, value in sorted(validation_timer.timings.items())
            },
        },
        "timing_s": {key: float(value) for key, value in sorted(timer.timings.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "crops" / "hotswap_100_manifest.json")
    parser.add_argument("--num-items", type=int, default=8)
    parser.add_argument("--crop-ids", nargs="*", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cache-length", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--decode-backend", default="raw_eager", choices=BACKEND_CHOICES)
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--torchair-cache-dir", type=Path, default=DEFAULT_TORCHAIR_CACHE_DIR)
    parser.add_argument(
        "--warmup-items",
        type=int,
        default=0,
        help="Run this many leading cohort items through the full staged path before measured items.",
    )
    parser.add_argument(
        "--decode-step-timing",
        action="store_true",
        help="Synchronize and record every fixed-step decode call inside static_decode_total.",
    )
    parser.add_argument("--json", action="store_true")
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
    entries = select_manifest_entries(
        manifest,
        num_items=int(args.num_items),
        crop_ids=args.crop_ids,
    )
    cohort = build_cohort_inputs(
        entries=entries,
        manifest_path=args.manifest,
        tokenizer=tokenizer,
        pre_cfg=pre_cfg,
        prompt_override=args.prompt,
    )
    prompt_tokens = [int(item.input_ids.shape[1]) for item in cohort]
    max_prompt_tokens = int(max(prompt_tokens))
    min_cache_length = max_prompt_tokens + max(0, int(args.max_new_tokens) - 1)
    cache_length = int(args.cache_length if args.cache_length is not None else max_prompt_tokens + int(args.max_new_tokens))
    if cache_length < min_cache_length:
        raise ValueError(
            f"--cache-length={cache_length} is too small for max prompt length {max_prompt_tokens} "
            f"and --max-new-tokens={args.max_new_tokens}; need at least {min_cache_length}"
        )

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
    compile_start = time.perf_counter()
    decode_fn, compile_meta = compile_decode_module(
        flat_decode,
        backend_name=args.decode_backend,
        device=device,
        cache_root=args.torchair_cache_dir,
        batch_size=1,
        cache_length=int(cache_length),
    )
    maybe_sync(device)
    compile_wrapper_s = time.perf_counter() - compile_start

    warm_cache = model.allocate_static_cache(
        batch_size=1,
        cache_length=int(cache_length),
        device=device,
        dtype=dtype,
        init_mode="zeros",
    )
    warm_input = torch.zeros((1, 1), device=device, dtype=torch.int64)
    warm_decode_position = min(int(max_prompt_tokens), int(cache_length) - 1)
    warm_position = torch.full((1,), int(warm_decode_position), device=device, dtype=torch.int64)
    warm_rope = torch.zeros((1, 1), device=device, dtype=torch.int64)
    maybe_sync(device)
    compile_first_start = time.perf_counter()
    decode_fn(warm_input, warm_position, warm_rope, *warm_cache.flat_tensors())
    maybe_sync(device)
    compile_first_s = time.perf_counter() - compile_first_start

    warmup_items: list[dict[str, Any]] = []
    stage_warmup_s = 0.0
    if int(args.warmup_items) < 0:
        raise ValueError(f"--warmup-items must be non-negative, got {args.warmup_items}")
    if int(args.warmup_items) > 0:
        warmup_count = min(int(args.warmup_items), len(cohort))
        maybe_sync(device)
        warmup_start = time.perf_counter()
        warmup_items = [
            run_item_stage_timing(
                model=model,
                decode_fn=decode_fn,
                item=item,
                cache_length=int(cache_length),
                max_new_tokens=int(args.max_new_tokens),
                device=device,
                trace_decode_steps=False,
            )
            for item in cohort[:warmup_count]
        ]
        maybe_sync(device)
        stage_warmup_s = time.perf_counter() - warmup_start

    items = [
        run_item_stage_timing(
            model=model,
            decode_fn=decode_fn,
            item=item,
            cache_length=int(cache_length),
            max_new_tokens=int(args.max_new_tokens),
            device=device,
            trace_decode_steps=bool(args.decode_step_timing),
        )
        for item in cohort
    ]

    timing_summary = aggregate_item_timings(items)
    mismatches = [
        {
            "item": int(idx),
            "id": str(item["id"]),
            "first_mismatch": item["correctness"]["first_mismatch"],
        }
        for idx, item in enumerate(items)
        if not bool(item["correctness"]["tokens_match"])
    ]
    warmup_mismatches = [
        {
            "item": int(idx),
            "id": str(item["id"]),
            "first_mismatch": item["correctness"]["first_mismatch"],
        }
        for idx, item in enumerate(warmup_items)
        if not bool(item["correctness"]["tokens_match"])
    ]
    output = {
        "experiment": "05_full_recognizer_optimizations",
        "model": str(model_dir),
        "device": str(device),
        "dtype": str(dtype),
        "decode_backend": str(args.decode_backend),
        "npu_jit_compile": args.npu_jit_compile,
        "vision_attention": get_vision_attention_impl(),
        "decode_attention": DECODE_ATTENTION if device.type == "npu" else "manual",
        "decode_cache_update": DECODE_CACHE_UPDATE if device.type == "npu" else "per_row_copy",
        "linear_weight_format": weight_format_meta,
        "compile": compile_meta,
        "num_items": int(len(items)),
        "max_new_tokens": int(args.max_new_tokens),
        "cache_length": int(cache_length),
        "warmup_items": int(len(warmup_items)),
        "decode_step_timing": bool(args.decode_step_timing),
        "prompt_tokens": {
            "min": int(min(prompt_tokens)),
            "max": int(max(prompt_tokens)),
            "per_item": prompt_tokens,
        },
        "setup_timing_s": {
            "model_load": float(model_load_s),
            "decode_weight_format": float(weight_format_meta.get("setup_s", 0.0) or 0.0),
            "compile_wrapper": float(compile_wrapper_s),
            "compile_first_call": float(compile_first_s),
            "compile_first_call_cache_position": int(warm_decode_position),
        },
        "stage_warmup": {
            "count": int(len(warmup_items)),
            "elapsed_s": float(stage_warmup_s),
            "item_ids": [str(item["id"]) for item in warmup_items],
            "mismatch_count": int(len(warmup_mismatches)),
            "first_mismatches": warmup_mismatches[:8],
            "timing_summary_s": aggregate_item_timings(warmup_items) if warmup_items else {},
        },
        "correctness": {
            "reference": "direct_local_static_fixed_steps",
            "all_required_checks_passed": bool(not mismatches and not warmup_mismatches),
            "mismatch_count": int(len(mismatches)),
            "first_mismatches": mismatches[:8],
            "warmup_mismatch_count": int(len(warmup_mismatches)),
            "warmup_first_mismatches": warmup_mismatches[:8],
        },
        "stage_timing_summary_s": timing_summary,
        "items": items,
        "stage_notes": {
            "preprocessing": "PIL resize/normalize/patchify is outside model timing; device_transfer is reported separately.",
            "vision_total": "vision_prepare + patch/position embeddings + native-resolution vision encoder + post layernorm.",
            "native_resolution_visual_encoder_total": "patch/position embeddings + native-resolution vision encoder + post layernorm, excluding cu_seqlens/pixel reshape preparation.",
            "adaptive_mlp_projector": "2x2 spatial merge and MLP connector into text hidden size.",
            "prefill_total_excluding_device_transfer": "vision + projector + text embeddings/scatter/mRoPE/cache allocation/text prefill/lm_head/argmax.",
            "model_total_excluding_device_transfer": "prefill_total_excluding_device_transfer + static_decode_total.",
            "static_decode_total": "Fixed-step decode with no per-token EOS host sync; use experiment 4 for scheduler/EOS throughput.",
            "decode_step_timing": "Optional diagnostic mode that synchronizes every decode step and should not be used for clean throughput.",
            "stage_warmup": "Optional leading full-path item warmup that is recorded separately and excluded from measured item summaries.",
            "correctness": "The staged path is compared against direct local static fixed-step generation outside the timed stage measurements.",
        },
    }
    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True, default=json_default))
        return

    print(f"experiment={output['experiment']} device={output['device']} dtype={output['dtype']} decode_backend={output['decode_backend']}")
    print(f"num_items={output['num_items']} max_new_tokens={output['max_new_tokens']} cache_length={output['cache_length']}")
    for key in (
        "vision_total",
        "native_resolution_visual_encoder_total",
        "vision_encoder",
        "adaptive_mlp_projector",
        "text_prefill",
        "static_decode_total",
        "model_total_excluding_device_transfer",
    ):
        stat = timing_summary.get(key, {})
        print(
            f"{key}: p50={stat.get('p50')} avg={stat.get('avg')} "
            f"max={stat.get('max')} sum={stat.get('sum')}"
        )


if __name__ == "__main__":
    main()
